'''The UI layer: tool lines, the timeline, the panels, and the output contract.

The most valuable assertions here are the two that a refactor breaks silently: the
render tuple matching the wired outputs, and id de-duplication surviving a subgraph's
re-emission of the whole transcript.
'''

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from tests.harness import scoped_results_root


# --- One line per tool call ---------------------------------------------------


def test_tool_lines_name_the_action_not_the_function() -> None:
    from app.ui import tool_display as display

    view = display.describe_call(
        "transfer_to_research_agent",
        {
            "objective": "Build the AML knowledge graph",
            "inputs": "MONDO:0018874",
            "expected_output": "the pickle path and counts",
        },
    )
    assert view.label == "Delegated to Research Agent — Build the AML knowledge graph"

    # A conversation recorded before the structured brief still renders its objective.
    legacy = display.describe_call("transfer_to_data_agent", {"task": "Rank the candidates"})
    assert legacy.label == "Delegated to Data Agent — Rank the candidates"

    view = display.describe_call("hERG_classifier", {"smiles_input": ["CCO", "CCC", "c1ccccc1"]})
    assert view.label == "Predicted hERG inhibition · 3 compounds"

    view = display.describe_call("create_knowledge_graph", {"disease_id": "MONDO:0018874"})
    assert view.label == "Built knowledge graph · MONDO:0018874"

    view = display.describe_call(
        "python_executor", {"code": "# rank candidates by composite score\nimport pandas as pd"}
    )
    assert view.label == "Ran Python — rank candidates by composite score"

    view = display.describe_call("literature_search_litsense", {"query": "venetoclax resistance"})
    assert view.label == "Searched literature — venetoclax resistance"


def test_failures_are_visible_without_expanding() -> None:
    from app.ui import tool_display as display

    # LangChain serializes a dict-returning tool into a string, so the envelope has
    # to be decoded or every failure would render with a success tick.
    serialized = (
        '{"ok": false, "error_type": "InterpreterError", '
        '"error": "The variable `df` is not defined.", "recovery": "Load the CSV first."}'
    )
    status, note = display.describe_result("python_executor", serialized)
    assert status == "error"
    assert note.startswith("InterpreterError:")
    body = display.render_result_body("python_executor", serialized)
    assert "Load the CSV first." in body

    status, note = display.describe_result("hERG_classifier", "Error: CPSign failed for hERG")
    assert status == "error" and "CPSign failed" in note

    entry = display.render_tool_entry(
        display.ToolView(label="Ran Python", status="error", note="InterpreterError: x")
    )
    # Collapsed by default, with the outcome on the summary line.
    assert "<details" in entry and "open" not in entry.split(">")[0]
    assert "tool-entry--error" in entry and "✕" in entry


def test_prediction_shortfalls_reach_the_summary_line() -> None:
    '''A model that silently drops compounds must not look like a clean run.'''

    from app.ui import tool_display as display

    result = (
        "/tmp/u/c/hERG_results.csv\n\n[warning] 18 of 20 structures were predicted. "
        "CPSign could not featurize 2: CCO, C."
    )
    status, note = display.describe_result("hERG_classifier", result)
    assert status == "ok"
    assert "18 of 20" in note


def test_plan_traffic_is_suppressed() -> None:
    from app.ui.tool_display import SUPPRESSED_TOOLS

    assert SUPPRESSED_TOOLS == {"plan_update", "plan_status"}


# --- The timeline -------------------------------------------------------------


def _state():
    from app.state import UIState

    return UIState(current_thread_id="conv")


def test_a_call_and_its_result_render_as_one_entry() -> None:
    from app.ui.chat_timeline import process_chunk

    state = _state()
    call = AIMessage(
        content="Delegating.",
        name="supervisor",
        id="m1",
        tool_calls=[{"name": "transfer_to_research_agent", "args": {"task": "kg"}, "id": "c1"}],
    )
    assert process_chunk(state, {"supervisor": {"messages": [call]}})
    block = state.agent_blocks[state.last_agent_block_id]
    entries = [item for item in block["items"] if item["type"] == "tool_call"]
    assert len(entries) == 1 and entries[0]["status"] == "running"

    # The result is committed by the subgraph, under a different node name, and must
    # still land on the supervisor's call — attribution is by tool_call_id.
    result = ToolMessage(content="Delegated", name="transfer_to_research_agent", tool_call_id="c1", id="m2")
    assert process_chunk(state, {"execution": {"messages": [result]}})
    entries = [item for item in block["items"] if item["type"] == "tool_call"]
    assert len(entries) == 1 and entries[0]["status"] == "ok"


def test_a_reemitted_transcript_renders_once() -> None:
    '''A node wrapping a subgraph re-emits every message the subgraph produced.'''

    from app.ui.chat_timeline import process_chunk

    state = _state()
    messages = [
        HumanMessage(content="Find candidates", id="h1"),
        AIMessage(content="Working on it.", name="supervisor", id="m1"),
        AIMessage(content="Graph built.", name="research_agent", id="m2"),
    ]
    process_chunk(state, {"supervisor": {"messages": [messages[1]]}})
    process_chunk(state, {"research_agent": {"messages": [messages[2]]}})
    blocks_before = len(state.agent_blocks)
    texts_before = sum(
        1
        for block in state.agent_blocks.values()
        for item in block["items"]
        if item["type"] == "message"
    )

    # The aggregate arrives under the wrapping node's name.
    changed = process_chunk(state, {"execution": {"messages": messages}})
    texts_after = sum(
        1
        for block in state.agent_blocks.values()
        for item in block["items"]
        if item["type"] == "message"
    )
    assert not changed
    assert len(state.agent_blocks) == blocks_before
    assert texts_after == texts_before


def test_suppressed_nodes_stay_suppressed_inside_an_aggregate() -> None:
    '''Skipping a message is not enough; its id must be recorded.'''

    from app.ui.chat_timeline import process_chunk

    state = _state()
    ledger = AIMessage(content="PLAN · run 1 · 0/4 steps resolved", name="plan_init", id="p1")
    assert not process_chunk(state, {"plan_init": {"messages": [ledger]}})
    assert not state.agent_blocks

    # The same message arrives again inside the execution aggregate.
    assert not process_chunk(state, {"execution": {"messages": [ledger]}})
    rendered = [
        item
        for block in state.agent_blocks.values()
        for item in block["items"]
        if item.get("type") == "message"
    ]
    assert not rendered, "the plan ledger must never reach the transcript"


def test_streamed_text_is_replaced_by_the_completed_message() -> None:
    from langchain_core.messages import AIMessageChunk

    from app.ui.chat_timeline import process_chunk

    state = _state()
    process_chunk(state, {"report_agent_complex": {"messages": [AIMessageChunk(content="Twenty ", id="r1")]}})
    process_chunk(state, {"report_agent_complex": {"messages": [AIMessageChunk(content="candid", id="r1")]}})
    process_chunk(
        state,
        {"report_agent_complex": {"messages": [AIMessage(content="Twenty candidates, ranked.", name="report_agent_complex", id="r1")]}},
    )
    block = state.agent_blocks[state.last_agent_block_id]
    texts = [item["content"] for item in block["items"] if item["type"] == "message"]
    assert texts == ["Twenty candidates, ranked."]


def test_the_deliverable_is_styled_as_a_document() -> None:
    from app.ui.chat_timeline import process_chunk

    state = _state()
    process_chunk(
        state,
        {
            "report_agent_complex": {
                "messages": [
                    AIMessage(
                        content="# Response Summary\n\n## Answer\nTwenty candidates.",
                        name="report_agent_complex",
                        id="r1",
                    )
                ]
            }
        },
    )
    html = state.messages[-1].content.value
    assert "agent-block-content--report" in html
    assert "agent-message-section--report" in html

    # A report's evidence usually arrives as a markdown table, and `table` is a
    # GFM extension the strict commonmark preset leaves off — without it the
    # pipes render verbatim and the table styling never matches anything.
    state = _state()
    process_chunk(
        state,
        {
            "report_agent_complex": {
                "messages": [
                    AIMessage(
                        content="| Drug | hERG |\n| --- | --- |\n| Midostaurin | 0.5 |",
                        name="report_agent_complex",
                        id="r-table",
                    )
                ]
            }
        },
    )
    assert "<table>" in state.messages[-1].content.value

    state = _state()
    process_chunk(state, {"planning_agent": {"messages": [AIMessage(content="📋 **PLAN:** x", name="planning_agent", id="p9")]}})
    assert "agent-block-content--plan" in state.messages[-1].content.value


def test_snapshot_round_trip_preserves_tool_outcomes() -> None:
    from app.ui.chat_timeline import (
        export_timeline_snapshot,
        process_chunk,
        rebuild_from_timeline_snapshot,
    )

    state = _state()
    process_chunk(
        state,
        {
            "research_agent": {
                "messages": [
                    AIMessage(
                        content="Building.",
                        name="research_agent",
                        id="m1",
                        tool_calls=[{"name": "create_knowledge_graph", "args": {"disease_id": "MONDO:1"}, "id": "c1"}],
                    )
                ]
            }
        },
    )
    process_chunk(
        state,
        {"research_agent": {"messages": [ToolMessage(content="/tmp/kg.pkl", name="create_knowledge_graph", tool_call_id="c1", id="m2")]}},
    )
    snapshot = export_timeline_snapshot(state)

    restored = _state()
    assert rebuild_from_timeline_snapshot(restored, snapshot)
    block = next(iter(restored.agent_blocks.values()))
    entry = next(item for item in block["items"] if item["type"] == "tool_call")
    assert entry["status"] == "ok"
    assert entry["label"].startswith("Built knowledge graph")
    # A block that was mid-stream must not come back with a live spinner.
    assert all(
        (message.metadata or {}).get("status") != "pending"
        for message in restored.messages
        if message.role == "assistant"
    )


# --- Panels and the output contract -------------------------------------------


def test_plan_panel_renders_the_file() -> None:
    from app.ui.progress_panel import progress_markup
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        plan_store.start_run(
            goal="Find AML candidates",
            steps=[
                plan_store.PlanStep(number=1, title="Build the graph", agent="research_agent"),
                plan_store.PlanStep(number=2, title="Screen ADMET", agent="prediction_agent"),
            ],
            kind="complex",
            constraints=["Phase 3 or later"],
            **scope,
        )
        plan_store.update_step(step_number=1, status="completed", note="412 drugs", **scope)

        markup = progress_markup("conv")
        assert "Task monitor" in markup
        assert "1/2" in markup
        assert "Find AML candidates" in markup
        assert "Step 2 of 2" in markup
        assert "plan-step--done" in markup
        assert "Research Agent" in markup and "412 drugs" in markup
        assert "Phase 3 or later" in markup

        # Nothing to show for a conversation with no plan.
        assert progress_markup("other") == ""


def test_plan_panel_pulses_only_while_the_run_is_in_flight() -> None:
    '''`in_progress` in the file outlives the run; the dot must not.'''

    from app.ui.progress_panel import progress_markup
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        plan_store.start_run(
            goal="Find AML candidates",
            steps=[plan_store.PlanStep(number=1, title="Build the graph", agent="research_agent")],
            kind="complex",
            **scope,
        )
        plan_store.update_step(step_number=1, status="in_progress", **scope)

        assert "plan-panel__pulse" in progress_markup("conv", running=True)
        assert "plan-panel__pulse" not in progress_markup("conv")


def test_the_stylesheet_survives_gradios_prefixer() -> None:
    '''Two traps that cost an afternoon each, pinned so they cannot come back.

    Gradio rewrites every selector in `css=` to `.gradio-container ... .contain
    <selector>`, and it rewrites the arguments *inside* `:is()` as it goes — which
    turns a bare tag list into a class chain and quietly lifts that rule above the
    specific ones written after it. And CSS comments do not nest: an inner marker
    ends the block early, after which the explanatory prose is parsed as CSS and
    swallows whole rules — that is how `#chatbot-panel`'s own font-size vanished.
    '''

    import re

    from app.ui.theme import APP_CSS

    position = 0
    while True:
        opened = APP_CSS.find("/*", position)
        if opened < 0:
            break
        closed = APP_CSS.find("*/", opened + 2)
        assert closed > 0, "unterminated CSS comment"
        inner = APP_CSS.find("/*", opened + 2)
        assert not (0 <= inner < closed), f"nested CSS comment near {APP_CSS[opened:opened + 70]!r}"
        position = closed + 2

    rules = re.sub(r"/\*.*?\*/", "", APP_CSS, flags=re.DOTALL)
    assert ":is(" not in rules, "`:is()` in a selector is mangled by gradio's prefixer"

    # Sizes come from the scale, so neighbouring elements cannot drift a
    # hundredth of a rem apart again.
    literals = [
        value
        for value in re.findall(r"font-size:\s*([^;!}]+?)\s*(?:!important)?\s*[;}]", rules)
        if not value.startswith("var(") and value not in {"inherit", "0.9em"}
    ]
    assert not literals, f"font sizes outside the scale: {sorted(set(literals))}"


def test_a_checkbox_keeps_the_tick_gradio_paints() -> None:
    '''The episodic-memory toggle has to look toggled.

    Gradio renders a checkbox with `appearance: none` and paints the tick with a
    `background-image` on `:checked`. The sidebar's field styling used the
    `background` SHORTHAND with `!important`, which resets `background-image` to
    none in every state — so the box looked identical checked and unchecked, and
    the toggle read as a dead control even though it was flipping state.
    '''

    import re

    from app.ui.theme import APP_CSS

    rules = re.sub(r"/\*.*?\*/", "", APP_CSS, flags=re.DOTALL)
    assert '#sidebar-column input:not([type="checkbox"])' in rules, (
        "the text-field rule must exclude checkboxes"
    )
    for block in re.finditer(r"([^{}]*)\{([^}]*)\}", rules):
        selector, body = block.group(1).strip(), block.group(2)
        targets_checkbox = 'input[type="checkbox"]' in selector.replace(
            ':not([type="checkbox"])', ""
        )
        if not targets_checkbox:
            continue
        assert not re.search(r"(^|;)\s*background\s*:", body), (
            f"`background` shorthand on a checkbox erases its tick: {selector}"
        )


def test_panels_are_only_sent_when_they_change() -> None:
    import gradio as gr

    from app.ui.conversation_panel import conversation_panel_update
    from app.ui.progress_panel import progress_update

    with scoped_results_root():
        state = _state()
        state.thread_ids = [{"thread_id": "conv", "title": "AML"}]

        first = conversation_panel_update(state)
        assert not isinstance(first, gr.skip().__class__) or first != gr.skip()
        second = conversation_panel_update(state)
        assert second == gr.skip()

        assert progress_update(state) is not None
        assert progress_update(state) == gr.skip()


def test_render_matches_the_wired_outputs() -> None:
    '''`render` and `layout`'s output list must stay the same length.

    `render` is the *only* projection in this build. The web build has a second one
    that appends the sign-in controls, and every handler had to pick the right one;
    with no sign-in there is one shape and nothing to pick.
    '''

    import inspect

    from app.ui import layout
    from app.ui.projection import render

    state = _state()
    with scoped_results_root():
        workspace = render(state)

    source = inspect.getsource(layout.build_demo)
    body = source.split("workspace_outputs = [", 1)[1].split("]", 1)[0]
    wired = [line.strip().rstrip(",") for line in body.strip().splitlines() if line.strip()]
    assert len(workspace) == len(wired), (len(workspace), wired)


def test_every_conversation_can_be_deleted() -> None:
    '''There are no shared read-only conversations here, so every card gets a bin.'''

    from app.ui.conversation_panel import conversation_panel_markup

    state = _state()
    state.thread_ids = [
        {"thread_id": "conv", "title": "My task"},
        {"thread_id": "other", "title": "Another task"},
    ]
    with scoped_results_root():
        markup = conversation_panel_markup(state)
    assert markup.count("data-delete-thread") == 2


def test_stop_and_send_are_mutually_exclusive() -> None:
    from app.ui.projection import control_updates

    state = _state()
    send, stop = control_updates(state)
    assert send["interactive"] is True and stop["interactive"] is False

    state.running_threads.add("conv")
    send, stop = control_updates(state)
    assert send["interactive"] is False and stop["interactive"] is True


def test_approval_gate_is_visible_only_while_paused() -> None:
    from app.ui.approval import approval_updates

    banner, approve, changes = approval_updates(None)
    assert banner["visible"] is False and approve["visible"] is False and changes["visible"] is False

    banner, approve, changes = approval_updates({"type": "plan_review", "message": "Review it."})
    assert banner["visible"] is True and approve["visible"] is True and changes["visible"] is True
    assert "Waiting for your approval" in banner["value"]
    assert "Review it." in banner["value"]


def test_build_demo_constructs() -> None:
    from app.ui.layout import build_demo

    demo = build_demo()
    assert demo is not None


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
