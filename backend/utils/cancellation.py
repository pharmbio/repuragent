'''Cooperative cancellation, so Stop reaches inside a running tool.

Closing the event stream stops the UI immediately and abandons the graph, but it
cannot kill work already inside a synchronous tool: LangChain runs sync tools in a
thread pool and a thread cannot be interrupted from outside. That matters here
more than in most agents — building a disease knowledge graph walks hundreds of
sequential ChEMBL/UniProt/OpenTargets requests, and a CPSign prediction is a Java
subprocess — so "stopped" could otherwise mean the browser goes quiet while the
machine keeps working for another ten minutes.

The contract is deliberately tiny:

* the UI calls `request_cancel(conversation_id)`;
* long-running tool loops call `raise_if_cancelled()` at a point where abandoning
  the work is safe (top of a per-item loop, between subprocess polls);
* `raise_if_cancelled()` is a dictionary lookup on a `threading.Event` when
  nothing has been cancelled, so it is free to call often.

Nothing depends on cancellation being observed: a tool that never checks simply
runs to completion, and the run is discarded either way.
'''

from __future__ import annotations

import threading
from typing import Optional

from backend.utils.output_paths import DEFAULT_CONVERSATION, get_current_conversation_id


class ExecutionCancelled(Exception):
    '''Raised inside a tool when the user has stopped the run.'''

_events: dict[str, threading.Event] = {}
_guard = threading.Lock()


def scope_key(conversation_id: Optional[str] = None) -> str:
    return conversation_id or get_current_conversation_id() or DEFAULT_CONVERSATION


def _event_for(key: str) -> threading.Event:
    with _guard:
        event = _events.get(key)
        if event is None:
            event = threading.Event()
            _events[key] = event
        return event


def request_cancel(conversation_id: Optional[str] = None) -> None:
    '''Ask every tool running for this conversation to stop at its next checkpoint.

    Parameters:
    ---------
    conversation_id (str): the run to stop, defaulting to the ambient scope.
    '''

    _event_for(scope_key(conversation_id)).set()


def clear_cancel(conversation_id: Optional[str] = None) -> None:
    '''Reset the flag. Call this when a new run starts, not when one ends.

    Parameters:
    ---------
    conversation_id (str): the conversation whose flag to reset, defaulting to the ambient scope.
    '''

    key = scope_key(conversation_id)
    with _guard:
        event = _events.get(key)
    if event is not None:
        event.clear()


def is_cancelled(conversation_id: Optional[str] = None) -> bool:
    key = scope_key(conversation_id)
    with _guard:
        event = _events.get(key)
    return bool(event and event.is_set())


def raise_if_cancelled(conversation_id: Optional[str] = None, *, detail: str = "") -> None:
    '''Abort the current tool call if the user has stopped this conversation's run.

    Parameters:
    ---------
    conversation_id (str): the run to check, defaulting to the ambient scope.
    detail (str): what was in progress, named in the raised error.
    '''

    if not is_cancelled(conversation_id):
        return
    suffix = f" ({detail})" if detail else ""
    raise ExecutionCancelled(f"The user stopped this run{suffix}.")


def cancel_event(conversation_id: Optional[str] = None) -> threading.Event:
    '''The raw event, for code that wants to wait on it (e.g. subprocess polling).

    Parameters:
    ---------
    conversation_id (str): the run to watch, defaulting to the ambient scope.

    Returns:
    ----------
    event (threading.Event): the raw flag, for code that must wait on it rather than poll — the CPSign subprocess especially.
    '''

    return _event_for(scope_key(conversation_id))


def forget(conversation_id: Optional[str] = None) -> None:
    '''Drop bookkeeping for a conversation that no longer exists.

    Parameters:
    ---------
    conversation_id (str): the conversation to drop bookkeeping for.
    '''

    key = scope_key(conversation_id)
    with _guard:
        _events.pop(key, None)


# --- Making existing loops cancellable ----------------------------------------


def cancellable_tqdm(iterable=None, *args, **kwargs):
    '''A `tqdm` drop-in that checks for cancellation on every item.

    The knowledge-graph and prediction tools already wrap every long loop in
    `tqdm`, because each iteration is a network call. Swapping the import
    (`from backend.utils.cancellation import cancellable_tqdm as tqdm`) therefore
    turns *all* of them into cancellation checkpoints without touching a single
    loop body — and a loop that was not worth a progress bar was not worth a
    checkpoint either.

    The scope is resolved once, when iteration starts, so a worker thread that
    inherited the run's context keeps checking the right conversation.

    Parameters:
    ---------
    iterable (iterable): what to iterate, as `tqdm` takes it.
    *args: passed straight through to `tqdm`.
    **kwargs: passed straight through to `tqdm`.

    Returns:
    ----------
    iterator (generator): the same items, checking for cancellation on each one. Imported *as* `tqdm` in the knowledge-graph modules, so every long loop there becomes a checkpoint without touching a loop body.
    '''

    from tqdm.auto import tqdm as _tqdm

    if iterable is None:
        # Manual-update usage: nothing to hook, hand back the real bar.
        return _tqdm(*args, **kwargs)

    key = scope_key()

    def _iterate():
        event = _event_for(key)
        for item in _tqdm(iterable, *args, **kwargs):
            if event.is_set():
                raise ExecutionCancelled("The user stopped this run.")
            yield item

    return _iterate()


__all__ = [
    "ExecutionCancelled",
    "cancellable_tqdm",
    "cancel_event",
    "clear_cancel",
    "forget",
    "is_cancelled",
    "raise_if_cancelled",
    "request_cancel",
    "scope_key",
]
