'''Stopping a run, and what the user is left with afterwards.

Stop has to do two separate things, so both are asserted here: end the event loop and
close the stream (which abandons the graph), and raise the cancellation flag that
long tool loops check (because work already inside a synchronous tool cannot be
interrupted from outside).

The graph is replaced by a scripted event stream, so this tests the controller rather
than the agents.
'''

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage

from tests.harness import scoped_results_root


class FakeStream:
    '''A slow event stream that records whether it was closed.'''

    def __init__(self, events: List[Tuple[str, Any]], *, delay: float = 0.05) -> None:
        self._events = list(events)
        self._delay = delay
        self.closed = False
        self.emitted = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._events:
            raise StopAsyncIteration
        await asyncio.sleep(self._delay)
        self.emitted += 1
        return self._events.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _install_fakes(monkey: Dict[str, Any], stream: FakeStream) -> None:
    '''Replace the graph, the database and the title write with stubs.

    Parameters:
    ---------
    monkey (dict): the patches to install, undone by the caller.
    stream (FakeStream): the scripted event stream to run instead of the graph.
    '''

    import app.run_controller as controller
    import app.timeline_store as timeline_store

    async def fake_stream(*_args, **_kwargs):
        return stream

    monkey["stream"] = controller.stream_langgraph_events
    monkey["approval"] = controller.read_pending_approval
    monkey["title"] = controller.update_thread_title
    monkey["persist"] = timeline_store.persist

    controller.stream_langgraph_events = lambda *a, **k: stream
    controller.read_pending_approval = _async_none
    controller.update_thread_title = _async_noop
    timeline_store.persist = _async_noop


async def _async_none(*_args, **_kwargs):
    return None


async def _async_noop(*_args, **_kwargs):
    return None


def _restore(monkey: Dict[str, Any]) -> None:
    import app.run_controller as controller
    import app.timeline_store as timeline_store

    controller.stream_langgraph_events = monkey["stream"]
    controller.read_pending_approval = monkey["approval"]
    controller.update_thread_title = monkey["title"]
    timeline_store.persist = monkey["persist"]


def _state():
    from app.state import UIState

    state = UIState(current_thread_id="conv")
    state.thread_ids = [{"thread_id": "conv", "title": "AML"}]
    return state


def _notice_texts(state) -> List[str]:
    return [
        item.get("message", "")
        for block in state.agent_blocks.values()
        for item in block["items"]
        if item.get("type") == "notice"
    ]


def test_stop_ends_the_run_and_says_so() -> None:
    import app.run_controller as controller
    from backend.utils import cancellation

    events: List[Tuple[str, Any]] = [
        ("chunk", {"supervisor": {"messages": [AIMessage(content=f"step {i}", name="supervisor", id=f"m{i}")]}})
        for i in range(20)
    ]
    events.append(("complete", {"interrupted": False, "completed_at": 0}))
    stream = FakeStream(events)
    monkey: Dict[str, Any] = {}

    async def scenario():
        state = _state()
        cancelled_during_run = False
        updates = 0
        async for _ in controller.run_user_message("Find AML candidates", state):
            updates += 1
            if updates == 3:
                await controller.on_stop_run(state)
                # The flag a tool would see must be raised while the run is live.
                cancelled_during_run = cancellation.is_cancelled("conv")
        return state, updates, cancelled_during_run

    with scoped_results_root():
        _install_fakes(monkey, stream)
        try:
            state, updates, cancelled_during_run = asyncio.run(scenario())
        finally:
            _restore(monkey)

    # Stopped well before the stream ran out.
    assert stream.emitted < 20, stream.emitted
    assert stream.closed, "the stream must be closed so the graph is abandoned"
    assert cancelled_during_run, "long tool loops must see the cancellation flag"

    # The user is told, rather than being left with a run that looks finished.
    notices = _notice_texts(state)
    assert notices and "You stopped this run" in notices[0]
    assert "continue from here" in notices[0]

    # And the thread is left ready for the next message.
    assert "conv" not in state.running_threads
    assert "conv" not in state.stop_signals
    assert not cancellation.is_cancelled("conv"), "the flag must be cleared afterwards"
    assert state.pending_approval is None


def test_a_new_run_is_not_killed_by_the_previous_stop() -> None:
    '''The cancellation flag is per conversation and must be cleared on start.'''

    import app.run_controller as controller
    from backend.utils import cancellation

    events = [("complete", {"interrupted": False, "completed_at": 0})]
    stream = FakeStream(events, delay=0.0)
    monkey: Dict[str, Any] = {}

    async def scenario():
        state = _state()
        cancellation.request_cancel("conv")  # left over from an earlier stop
        updates = 0
        async for _ in controller.run_user_message("Try again", state):
            updates += 1
        return state, updates

    with scoped_results_root():
        _install_fakes(monkey, stream)
        try:
            state, updates = asyncio.run(scenario())
        finally:
            _restore(monkey)

    assert updates >= 2, updates
    assert not _notice_texts(state), "the run must not report itself as stopped"
    assert stream.emitted == 1


def test_an_unhandled_tool_error_is_surfaced_not_swallowed() -> None:
    import app.run_controller as controller

    class ExplodingStream(FakeStream):
        async def __anext__(self):
            await asyncio.sleep(0)
            raise RuntimeError("ChEMBL returned 503")

    stream = ExplodingStream([])
    monkey: Dict[str, Any] = {}

    async def scenario():
        state = _state()
        async for _ in controller.run_user_message("Find AML candidates", state):
            pass
        return state

    with scoped_results_root():
        _install_fakes(monkey, stream)
        try:
            state = asyncio.run(scenario())
        finally:
            _restore(monkey)

    errors = [
        item
        for block in state.agent_blocks.values()
        for item in block["items"]
        if item.get("type") == "error"
    ]
    assert errors, "the failure must be visible in the transcript"
    assert "ChEMBL returned 503" in errors[0]["detail"]
    assert "conv" not in state.running_threads


def test_a_submission_with_nothing_to_run_never_reaches_the_graph() -> None:
    '''Whitespace, or no conversation selected: one render, and the graph is untouched.

    The web build guards a third case here — a read-only demo conversation — which
    this build has no equivalent of. These two remain, and they are the ones that
    would otherwise start a run whose output has nowhere to go.
    '''

    import app.run_controller as controller

    for prompt, blank_thread in (("   ", False), ("Find AML candidates", True)):
        stream = FakeStream([], delay=0.0)
        monkey: Dict[str, Any] = {}

        async def scenario():
            state = _state()
            if blank_thread:
                state.current_thread_id = None
            updates = 0
            async for _ in controller.run_user_message(prompt, state):
                updates += 1
            return state, updates

        with scoped_results_root():
            _install_fakes(monkey, stream)
            try:
                state, updates = asyncio.run(scenario())
            finally:
                _restore(monkey)

        assert updates == 1, (prompt, blank_thread, "one render, then nothing")
        assert stream.emitted == 0, "the graph must never be invoked"
        assert not state.running_threads


def test_complete_payload_is_parsed_into_the_approval_gate() -> None:
    from app.run_controller import apply_stream_event, parse_complete_payload

    interrupted, approval, completed = parse_complete_payload(
        {"interrupted": True, "approval": {"type": "plan_review", "plan": "p"}, "completed_at": 1_700_000_000}
    )
    assert interrupted and approval["plan"] == "p" and completed is not None

    # An interrupt with no readable payload is still an approval gate.
    interrupted, approval, _ = parse_complete_payload({"interrupted": True, "approval": None})
    assert interrupted and approval == {"type": "plan_review"}

    state = _state()
    apply_stream_event("complete", {"interrupted": True, "approval": {"type": "plan_review"}}, state)
    assert state.pending_approval == {"type": "plan_review"}
    apply_stream_event("complete", {"interrupted": False}, state)
    assert state.pending_approval is None


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
