'''Driving one run and reflecting it in the UI.

This module owns four things that are easy to get subtly wrong:

**The output scope.** It is pinned into an explicit `contextvars.Context` that every
task of the run is created in. `asyncio.Task` copies the ambient context *at
creation*, and the run loop wraps each `__anext__` in a Task so it can race the
file-refresh tick — so a scope merely set inside the streaming generator lives only
in the first task's copy, and from the second event onward every tool call falls back
to `anonymous-user`/`default-thread`. That sent artifacts to a directory the
conversation never reads.

**Stop.** Two things happen, because one is not enough. The event loop breaks and
closes the stream, which abandons the graph mid-run; and a cancellation flag is set
that long tool loops check, because a knowledge-graph traversal already inside a
tool cannot be interrupted from outside — LangChain runs sync tools in a thread pool.
Without the flag, "stopped" meant the browser went quiet while the machine kept
working for another ten minutes.

**Whether this message resumes an approval.** Always asked of the graph, never of
session state. Sending plain input to an interrupted thread makes LangGraph restart
from `START` and re-plan, which used to trap users in an approval loop.

**Runs the user is not watching.** They write into an in-memory buffer flushed on a
timer, rather than a load-rebuild-save round trip per event.
'''

from __future__ import annotations

import asyncio
import contextvars
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Dict, Optional, Tuple

import gradio as gr

from app import timeline_store
from app.app_config import AppRunConfig
from app.config import FILE_LIST_REFRESH_INTERVAL_SECONDS, logger
from app.conversation_store import update_thread_title
from app.files import refresh_thread_files
from app.langgraph_runner import (
    build_stream_input,
    read_pending_approval,
    stream_langgraph_events,
)
from app.state import UIState
from app.ui.approval import APPROVE_TEXT, REQUEST_CHANGES_HINT
from app.ui.chat_timeline import (
    append_error_block,
    append_notice_block,
    append_user_message,
    finalize_active_blocks,
    process_chunk,
)
from app.ui.conversation_panel import append_file_paths
from app.ui.projection import render
from backend.utils import cancellation
from backend.utils.output_paths import set_current_conversation_id


# --- Applying events ----------------------------------------------------------


def parse_complete_payload(
    payload: Any,
) -> Tuple[bool, Optional[Dict[str, Any]], Optional[datetime]]:
    if isinstance(payload, dict):
        interrupted = bool(payload.get("interrupted"))
        approval = payload.get("approval") if interrupted else None
        completed_at = payload.get("completed_at")
    else:
        interrupted, approval, completed_at = bool(payload), None, None

    completed = None
    if isinstance(completed_at, (int, float)):
        completed = datetime.fromtimestamp(completed_at, tz=timezone.utc)
    elif isinstance(completed_at, str):
        try:
            completed = datetime.fromisoformat(completed_at)
        except ValueError:
            completed = None
    if interrupted and not isinstance(approval, dict):
        approval = {"type": "plan_review"}
    return interrupted, approval, completed


def apply_stream_event(event_type: str, payload: Any, state: UIState) -> bool:
    '''Fold one streamed event into a timeline. True when it changed.

    Parameters:
    ---------
    event_type (str): `chunk`, `token` or `complete`.
    payload (Any): the event's contents.
    state (UIState): the state whose timeline to fold it into.

    Returns:
    ----------
    changed (boolean): True when the timeline actually changed, so a render is only sent when it must be.
    '''

    if event_type in ("chunk", "token"):
        return process_chunk(state, payload)
    if event_type == "complete":
        _, approval, _ = parse_complete_payload(payload)
        state.pending_approval = approval
        # Resolve the live spinner once the run settles, whether it finished or
        # paused for approval.
        finalize_active_blocks(state)
        return True
    return False


def record_stream_error(state: UIState, exc: Exception) -> bool:
    state.pending_approval = None
    return append_error_block(
        state,
        "The run stopped before finishing because a tool raised an unhandled error. "
        "Your conversation is preserved — send another message to adjust the request "
        "and continue from here.",
        title="Run interrupted",
        detail=f"{type(exc).__name__}: {exc}",
    )


# --- Concurrency --------------------------------------------------------------

_thread_locks: Dict[str, asyncio.Lock] = {}
_thread_locks_guard = asyncio.Lock()


async def _get_thread_lock(thread_id: str) -> asyncio.Lock:
    async with _thread_locks_guard:
        lock = _thread_locks.get(thread_id)
        if lock is None:
            lock = asyncio.Lock()
            _thread_locks[thread_id] = lock
        return lock


@asynccontextmanager
async def thread_execution_lock(thread_id: Optional[str]):
    '''One run at a time per conversation.

    Parameters:
    ---------
    thread_id (str): the conversation to lock.

    Returns:
    ----------
    lock (asyncio.Lock): that conversation's lock — one run at a time, so a second submit cannot interleave with the first.
    '''

    if not thread_id:
        yield
        return
    lock = await _get_thread_lock(thread_id)
    await lock.acquire()
    try:
        yield
    finally:
        lock.release()


def build_conversation_context(conversation_id: Optional[str]) -> contextvars.Context:
    '''A context with this run's output scope already bound into it.

    Returned rather than entered: the scope has to survive the generator's yield
    boundaries, and `contextvars` mutations do not, because the loop below creates a
    Task per `__anext__` and each Task starts from its own copy of the ambient
    context. Spawning the run's tasks *in* this context sidesteps that entirely.

    Parameters:
    ---------
    conversation_id (str): the conversation being run.

    Returns:
    ----------
    context (contextvars.Context): a context with the output scope already bound. Load-bearing: `asyncio.Task` copies the ambient context *at creation*, and the run loop wraps each `__anext__` in a Task, so a scope merely set inside the streaming generator would live only in the first task's copy.
    '''

    context = contextvars.copy_context()
    context.run(set_current_conversation_id, conversation_id)
    return context


def _spawn(coro, context: Optional[contextvars.Context]):
    if context is not None:
        try:
            return asyncio.get_running_loop().create_task(coro, context=context)
        except TypeError:
            # `context=` needs Python 3.11+. Older runtimes lose the pinned scope,
            # which the graph state still carries.
            logger.debug("create_task(context=...) unsupported; falling back")
    return asyncio.ensure_future(coro)


async def _events_with_ticks(
    stream: AsyncIterator[Tuple[str, Any]],
    interval: float,
    *,
    context: Optional[contextvars.Context] = None,
) -> AsyncIterator[Tuple[str, Any]]:
    '''Merge the event stream with a periodic tick.

    A single knowledge-graph build can take minutes without emitting anything, and
    files written during it — figures especially — should appear while it runs rather
    than all at once when it returns. Isolating the two-task race here keeps the run
    loop below a flat `async for`.

    Parameters:
    ---------
    stream (AsyncIterator[Tuple[str, Any]]): the graph's event stream.
    interval (float): seconds between ticks.
    context (contextvars.Context): the pinned context every task of the run is created in.

    Returns:
    ----------
    events (async iterator): the graph's events merged with a periodic tick that refreshes the file list.
    '''

    iterator = stream.__aiter__()
    pending = _spawn(iterator.__anext__(), context)
    timer = asyncio.ensure_future(asyncio.sleep(interval))
    try:
        while True:
            done, _ = await asyncio.wait({pending, timer}, return_when=asyncio.FIRST_COMPLETED)
            if timer in done:
                timer = asyncio.ensure_future(asyncio.sleep(interval))
                yield ("tick", None)
            if pending in done:
                try:
                    event = pending.result()
                except StopAsyncIteration:
                    return
                yield ("event", event)
                pending = _spawn(iterator.__anext__(), context)
    finally:
        for task in (pending, timer):
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task


# --- The run ------------------------------------------------------------------


async def _prepare_submission(prompt: str, state: UIState) -> Tuple[Optional[str], bool]:
    '''Record the user's message and return what to send to the graph.

    Returns `(payload, resume)`, with a None payload when there is nothing to run.

    Parameters:
    ---------
    prompt (str): what the user typed.
    state (UIState): the state to record the message into.

    Returns:
    ----------
    submission (tuple): `(message to send, whether this resumes a paused approval)`.
    '''

    thread_id = state.current_thread_id
    if not thread_id:
        return None, False

    prompt = (prompt or "").strip()
    if not prompt:
        return None, False

    # Whether this resumes a pending approval is decided by the graph, never by
    # session state: the UI's copy is reset by thread switches and page reloads, and
    # is never set at all when the interrupt landed while the user was elsewhere.
    resume = await read_pending_approval(thread_id) is not None

    final_prompt = prompt if resume else append_file_paths(prompt, state)
    append_user_message(state, prompt)
    await timeline_store.persist(thread_id, state)

    # Name the conversation after its first request.
    if len([message for message in state.messages if message.role == "user"]) == 1:
        title = prompt[:60].strip()
        await update_thread_title(thread_id, title)
        for thread in state.thread_ids:
            if thread.get("thread_id") == thread_id:
                thread["title"] = title
                break

    state.pending_approval = None
    state.current_app_config = AppRunConfig(
        user_request=final_prompt,
        conversation_id=thread_id,
        use_episodic_learning=state.use_episodic_learning,
    )
    return (prompt if resume else final_prompt), resume


async def _stream_run(prompt: str, state: UIState):
    thread_id = state.current_thread_id
    final_prompt, resume = await _prepare_submission(prompt, state)
    if final_prompt is None:
        yield render(state)
        return

    state.selected_thread_id = thread_id
    state.running_threads.add(thread_id)
    state.stop_signals[thread_id] = False
    # Clear any flag left by a previous stop, or this run would abort immediately.
    cancellation.clear_cancel(thread_id)
    yield render(state, clear_input=True)

    writer = timeline_store.DetachedTimelineWriter(thread_id)
    attached = True
    stopped = False

    run_context = build_conversation_context(thread_id)

    stream = stream_langgraph_events(
        state.current_app_config,
        build_stream_input(
            final_prompt,
            conversation_id=thread_id,
            use_episodic_learning=state.use_episodic_learning,
            resume=resume,
        ),
        thread_id,
        check_for_interrupts=True,
    )

    try:
        async for kind, item in _events_with_ticks(
            stream, FILE_LIST_REFRESH_INTERVAL_SECONDS, context=run_context
        ):
            if state.stop_signals.get(thread_id):
                stopped = True
                break

            viewing = (state.selected_thread_id or state.current_thread_id) == thread_id
            if viewing and not attached:
                # The user came back. Adopt whatever the detached buffer recorded
                # while they were away, then write to session state again.
                await _reattach(state, thread_id, writer)
                attached = True
                yield render(state)
            elif not viewing and attached:
                attached = False
                state.stale_threads.add(thread_id)

            if kind == "tick":
                if attached:
                    if refresh_thread_files(state, thread_id):
                        yield render(state)
                else:
                    await writer.maybe_flush()
                continue

            event_type, payload = item
            if event_type == "complete":
                _, _, completed_at = parse_complete_payload(payload)
                if not (isinstance(payload, dict) and payload.get("interrupted")):
                    state.last_run_at[thread_id] = completed_at or datetime.now(timezone.utc)

            if attached:
                if apply_stream_event(event_type, payload, state):
                    # Token events are display-only: the completed message arrives
                    # moments later and is what gets persisted. Writing a snapshot
                    # and rescanning files every ~120 ms during generation would
                    # cost more than it shows.
                    is_token = event_type == "token"
                    if not is_token:
                        await timeline_store.persist(thread_id, state)
                        refresh_thread_files(state, thread_id)
                    yield render(state, live=is_token)
            elif event_type != "token":
                # Nobody is watching, so tokens are pure cost.
                detached = await writer.state()
                if apply_stream_event(event_type, payload, detached):
                    writer.mark_dirty()
                    await writer.maybe_flush()
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        logger.exception("Run failed for thread %s", thread_id)
        target = state if attached else await writer.state()
        record_stream_error(target, exc)
        if attached:
            await timeline_store.persist(thread_id, state)
            yield render(state)
        else:
            writer.mark_dirty()
            await writer.maybe_flush(force=True)
    finally:
        with suppress(Exception):
            await stream.aclose()
        state.running_threads.discard(thread_id)
        state.stop_signals.pop(thread_id, None)
        cancellation.clear_cancel(thread_id)
        await writer.maybe_flush(force=True)

    if stopped:
        await _record_stop(state, thread_id, writer, attached=attached)
        yield render(state)
        return

    viewing = (state.selected_thread_id or state.current_thread_id) == thread_id
    if not viewing:
        state.stale_threads.add(thread_id)
    else:
        refresh_thread_files(state, thread_id)
    yield render(state)


async def _reattach(
    state: UIState, thread_id: str, writer: timeline_store.DetachedTimelineWriter
) -> None:
    '''Pull the detached buffer's work into the state the user can see.

    Parameters:
    ---------
    state (UIState): the state the user can see.
    thread_id (str): the conversation being reattached.
    writer (DetachedTimelineWriter): the buffer that recorded the run while nobody was watching.
    '''

    await writer.maybe_flush(force=True)
    writer.discard()
    from app.conversation_store import load_timeline

    timeline_store.apply_snapshot(state, await load_timeline(thread_id))
    state.stale_threads.discard(thread_id)
    refresh_thread_files(state, thread_id)


async def _record_stop(
    state: UIState,
    thread_id: str,
    writer: timeline_store.DetachedTimelineWriter,
    *,
    attached: bool,
) -> None:
    '''Leave a visible mark that the user stopped this run.

    Stopping used to just resolve the spinner, which is indistinguishable from the
    run finishing normally. It also matters that the graph checkpoint is left
    mid-run: the next message continues from there rather than starting clean, and
    the user should know that.

    Parameters:
    ---------
    state (UIState): the state to append the notice to.
    thread_id (str): the conversation that was stopped.
    writer (DetachedTimelineWriter): the detached buffer, written to when the run is not attached.
    attached (boolean): whether the user is currently watching this thread.
    '''

    target = state if attached else await writer.state()
    target.pending_approval = None
    finalize_active_blocks(target)
    append_notice_block(
        target,
        "You stopped this run. Work already finished is saved, the plan file records "
        "how far it got, and any files it produced are listed in the sidebar. Send "
        "another message to continue from here.",
        title="Run stopped",
    )
    if attached:
        state.pending_approval = None
        state.current_app_config = None
        await timeline_store.persist(thread_id, state)
        refresh_thread_files(state, thread_id)
    else:
        writer.mark_dirty()
        await writer.maybe_flush(force=True)


async def run_user_message(prompt: str, state: UIState):
    async with thread_execution_lock(state.current_thread_id):
        async for update in _stream_run(prompt, state):
            yield update


# --- Handlers -----------------------------------------------------------------


async def on_send_message(prompt: str, state: UIState):
    state = state or UIState()
    async for update in run_user_message(prompt, state):
        yield update


async def on_approve_plan(state: UIState):
    '''Approve the paused plan without making the user phrase it.

    Parameters:
    ---------
    state (UIState): the state whose thread is paused for review.

    Returns:
    ----------
    updates (tuple): the render for the approved run, without making the user phrase the approval.
    '''

    state = state or UIState()
    async for update in run_user_message(APPROVE_TEXT, state):
        yield update


def on_request_changes(state: UIState):
    '''Focus the conversation on revising the plan.

    No graph call: a plan is revised by describing the change, so this only clears
    the gate's buttons and hands the user back the textbox with a prompt saying what
    to write.

    Parameters:
    ---------
    state (UIState): the state whose thread is paused for review.

    Returns:
    ----------
    updates (tuple): the render that focuses the conversation on revising the plan.
    '''

    state = state or UIState()
    if state.pending_approval is not None:
        payload = dict(state.pending_approval)
        payload["message"] = REQUEST_CHANGES_HINT
        state.pending_approval = payload
    return render(state)


async def on_stop_run(state: UIState):
    '''Stop the run on the thread the user is looking at.

    Sets both signals: the loop flag, which ends streaming at the next event
    boundary, and the cancellation token, which the long tool loops check so work
    already inside a tool is abandoned too.

    Parameters:
    ---------
    state (UIState): the state whose thread should stop.

    Returns:
    ----------
    updates (tuple): the render after requesting the stop. Stopping appends a visible notice, because resolving the spinner alone is indistinguishable from finishing.
    '''

    state = state or UIState()
    thread_id = state.current_thread_id
    if thread_id and thread_id in state.running_threads:
        state.stop_signals[thread_id] = True
        cancellation.request_cancel(thread_id)
        gr.Info("Stopping — finishing the current tool call, then wrapping up.")
    else:
        gr.Info("Nothing is running on this conversation.")
    return render(state)


__all__ = [
    "apply_stream_event",
    "build_conversation_context",
    "on_approve_plan",
    "on_request_changes",
    "on_send_message",
    "on_stop_run",
    "parse_complete_payload",
    "run_user_message",
    "thread_execution_lock",
]
