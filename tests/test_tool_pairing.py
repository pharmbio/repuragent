'''Tool calls and their results must stay paired in every view the models are handed.

The bug this module exists for: a handoff returns
`Command(goto=<specialist>, graph=Command.PARENT)`, which *abandons* the supervisor's
own agent graph. The parent therefore receives the `Command`'s `update` and nothing
else — so an update carrying only the rendered brief left the parent holding a
`ToolMessage` whose `AIMessage`, the one with the `transfer_to_<agent>` `tool_call`,
existed only in the graph that was abandoned. The specialist ran fine; the run then
died on the *supervisor's next model call*, which is why it read as "the work cannot go
back to the supervisor":

    BadRequestError: 400 - Invalid parameter: messages with role 'tool' must be a
    response to a preceeding message with 'tool_calls'.  (param: messages.[6].role)

A scripted model accepts any message list, so nothing in the suite noticed.
`tests.harness.tool_pairing_problems` applies the provider's own two rules instead,
and these tests run it over every list the middleware assembles.
'''

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import scripted, tool_call
from tests.harness import (
    assert_every_model_call_is_valid,
    assert_tool_pairing,
    tool_pairing_problems,
)

PLAN = """📋 **PLAN:** Find repurposing candidates for acute myeloid leukemia

**TASK BREAKDOWN:**
  [1] **Build the AML knowledge graph**
      **Agent:** research_agent
      **Details:** Resolve the MONDO id and build the disease graph.
      **Depends on:** none

  [2] **Screen the shortlist for hERG risk**
      **Agent:** prediction_agent
      **Details:** Run the hERG classifier over the candidate SMILES.
      **Depends on:** 1

**ASSUMPTIONS AND OPEN QUESTIONS:**
- None
"""


def _delegate(agent: str, call_id: str, *, said: str = "Delegating.") -> AIMessage:
    '''One supervisor turn that hands a complete brief to `agent`.

    Parameters:
    ---------
    agent (str): the specialist to delegate to.
    call_id (str): the `tool_call` id, which the brief must come back answering.
    said (str): what the supervisor says alongside the call.

    Returns:
    ----------
    message (AIMessage): the delegating turn, with a brief that passes `validate_brief`.
    '''

    return AIMessage(
        content=said,
        tool_calls=[
            tool_call(
                f"transfer_to_{agent}",
                {
                    "objective": f"Complete the assigned step and report what {agent} found",
                    "inputs": "disease: acute myeloid leukemia; hERG cutoff 0.5",
                    "expected_output": "the counts, and the full path of anything written",
                },
                call_id,
            )
        ],
    )


def _install(category: str) -> None:
    '''Point the three decision-only model calls at fixed answers.

    Parameters:
    ---------
    category (str): the route the classifier should always choose.
    '''

    from core.agents import agentic_system as system

    system.install_helper_models(
        task_classifier=RunnableLambda(lambda _p: system.TaskClassification(category=category)),
        approval_judge=RunnableLambda(
            lambda _p: system.PlanFeedbackVerdict(decision="approve", constraints=[])
        ),
        context_summarizer=RunnableLambda(lambda _p: system.ContextDigest(summary="Ran the plan.")),
    )


def _run(
    models: Dict[str, Any],
    *,
    thread_id: str,
    resume: str = "approved",
    saver: Any = None,
) -> Path:
    '''Drive a full complex run — classify, plan, approve, execute, report.

    Parameters:
    ---------
    models (dict): the scripted model per role, whose `calls` are what gets checked.
    thread_id (str): the conversation this run belongs to.
    resume (str): what the user replies at the approval gate.
    saver (BaseCheckpointSaver): the checkpointer to use, so a caller can send a second turn down the same thread.

    Returns:
    ----------
    workdir (Path): the temporary results root, for the caller to clean up.
    '''

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-pairing-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    config.RESULTS_ROOT = workdir
    output_paths.RESULTS_ROOT = workdir

    from core.agents import agentic_system as system

    system.reset_model_cache()
    _install("complex")
    graph = system.build_graph(saver if saver is not None else InMemorySaver(), models=models)
    thread = {"configurable": {"thread_id": thread_id}, "recursion_limit": 100}

    async def drive() -> None:
        payload = {
            "messages": [HumanMessage(content="Find repurposing candidates for AML")],
            "conversation_id": thread_id,
        }
        async for _ in graph.astream(
            payload, config=thread, stream_mode=["updates"], subgraphs=True
        ):
            pass
        async for _ in graph.astream(
            Command(resume=resume), config=thread, stream_mode=["updates"], subgraphs=True
        ):
            pass

    asyncio.run(drive())
    return workdir


def _restore_roots(originals) -> None:
    import app.config as config
    import backend.utils.output_paths as output_paths

    config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals


def _roots():
    import app.config as config
    import backend.utils.output_paths as output_paths

    return (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)


# --- The end-to-end regression -------------------------------------------------


def test_a_delegation_leaves_no_orphaned_tool_message() -> None:
    '''The whole bug, end to end: two delegations, specialists that call tools.

    Every message list every agent is handed must satisfy the provider's rules — and
    the supervisor's list is the one that used to fail, on the call *after* the
    specialist reported back.
    '''

    originals = _roots()
    supervisor = scripted(
        _delegate("research_agent", "h1", said="Delegating the graph build."),
        AIMessage(
            content="",
            tool_calls=[
                tool_call("plan_update", {"step": 1, "status": "completed", "note": "412 drugs"}, "p1")
            ],
        ),
        _delegate("prediction_agent", "h2", said="Delegating the hERG screen."),
        AIMessage(
            content="",
            tool_calls=[
                tool_call("plan_update", {"step": 2, "status": "completed", "note": "3 flagged"}, "p2")
            ],
        ),
        AIMessage(content="Both steps are resolved."),
        label="supervisor",
    )
    # Both specialists do real tool traffic, so a span has something to collapse.
    research = scripted(
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r1")]),
        AIMessage(content="Knowledge graph built: 412 drugs, 88 proteins."),
        label="research",
    )
    prediction = scripted(
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r2")]),
        AIMessage(content="hERG run complete; 3 of 20 flagged."),
        label="prediction",
    )
    models = {
        "supervisor": supervisor,
        "research": research,
        "prediction": prediction,
        "data": scripted(AIMessage(content="unused"), label="data"),
        "planning": scripted(AIMessage(content=PLAN), label="planning"),
        "report": scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nTwo candidates survive."),
            label="report",
        ),
    }
    workdir = None
    try:
        workdir = _run(models, thread_id="pairing")

        assert research.calls and prediction.calls, "a specialist never ran"
        # The supervisor must have been called again after each specialist reported —
        # this is the call the BadRequestError killed.
        assert len(supervisor.calls) == 5, [len(c) for c in supervisor.calls]
        assert_every_model_call_is_valid(models)

        # And the delegating turn is really there, immediately before its brief.
        final = supervisor.calls[-1]
        briefs = [
            index
            for index, message in enumerate(final)
            if isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")).startswith("transfer_to_")
        ]
        assert len(briefs) == 2, [getattr(m, "name", None) for m in final]
        for index in briefs:
            caller = final[index - 1]
            assert isinstance(caller, AIMessage), caller
            assert [call["id"] for call in caller.tool_calls] == [final[index].tool_call_id]
    finally:
        _restore_roots(originals)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def test_a_plan_update_beside_a_handoff_stays_paired() -> None:
    '''A parallel call in the delegating turn is the mirror-image failure.

    The sibling `ToolMessage` is written into the graph the handoff abandons, so
    carrying the `AIMessage` up whole would leave *its* `tool_call` unanswered — the
    same rejection from the other side. Only the answered call may travel.
    '''

    originals = _roots()
    supervisor = scripted(
        AIMessage(
            content="Marking step 1 started and handing it over.",
            tool_calls=[
                tool_call("plan_update", {"step": 1, "status": "in_progress", "note": "starting"}, "p0"),
                tool_call(
                    "transfer_to_research_agent",
                    {
                        "objective": "Build the AML knowledge graph and report the counts",
                        "inputs": "disease: acute myeloid leukemia",
                        "expected_output": "the pickle path and the drug count",
                    },
                    "h1",
                ),
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                tool_call("plan_update", {"step": 1, "status": "completed", "note": "412 drugs"}, "p1")
            ],
        ),
        AIMessage(content="Step 1 is resolved."),
        label="supervisor",
    )
    models = {
        "supervisor": supervisor,
        "research": scripted(AIMessage(content="Graph built: 412 drugs."), label="research"),
        "prediction": scripted(AIMessage(content="unused"), label="prediction"),
        "data": scripted(AIMessage(content="unused"), label="data"),
        "planning": scripted(AIMessage(content=PLAN), label="planning"),
        "report": scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nDone."), label="report"
        ),
    }
    workdir = None
    try:
        workdir = _run(models, thread_id="parallel")
        assert len(supervisor.calls) >= 2, len(supervisor.calls)
        assert_every_model_call_is_valid(models)

        # The delegating turn arrived, carrying only the call the brief answers.
        final = supervisor.calls[-1]
        brief = next(
            index
            for index, message in enumerate(final)
            if isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")) == "transfer_to_research_agent"
        )
        caller = final[brief - 1]
        assert isinstance(caller, AIMessage), caller
        assert [call["id"] for call in caller.tool_calls] == ["h1"], caller.tool_calls
    finally:
        _restore_roots(originals)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def test_two_delegations_to_the_same_specialist_stay_paired() -> None:
    '''The second brief must not reopen the first one's span.'''

    originals = _roots()
    supervisor = scripted(
        _delegate("research_agent", "h1", said="First pass."),
        _delegate("research_agent", "h2", said="Second pass, now the pathways."),
        AIMessage(content="Both passes are done."),
        label="supervisor",
    )
    research = scripted(
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r1")]),
        AIMessage(content="Drugs extracted: 412."),
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r2")]),
        AIMessage(content="Pathways extracted: 31."),
        label="research",
    )
    models = {
        "supervisor": supervisor,
        "research": research,
        "prediction": scripted(AIMessage(content="unused"), label="prediction"),
        "data": scripted(AIMessage(content="unused"), label="data"),
        "planning": scripted(AIMessage(content=PLAN), label="planning"),
        "report": scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nDone."), label="report"
        ),
    }
    workdir = None
    try:
        workdir = _run(models, thread_id="twice")
        assert len(research.calls) == 4, len(research.calls)
        assert_every_model_call_is_valid(models)
    finally:
        _restore_roots(originals)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


# --- The units -----------------------------------------------------------------


def test_the_handoff_carries_its_delegating_message_up() -> None:
    '''`handoff_messages` is what survives `Command.PARENT` abandoning the inner graph.'''

    from core.agents.handoff import handoff_messages

    delegating = AIMessage(
        content="Handing over.",
        name="supervisor",
        id="ai-1",
        tool_calls=[tool_call("transfer_to_data_agent", {}, "h1")],
    )
    history = [HumanMessage(content="rank these", id="u-1"), delegating]
    brief = ToolMessage(content="TASK BRIEF", name="transfer_to_data_agent", tool_call_id="h1")

    carried = handoff_messages(history, brief, "h1")
    assert [message.id for message in carried[:2]] == ["u-1", "ai-1"]
    assert carried[-1] is brief
    assert_tool_pairing(carried, label="handoff update")

    # An empty inner list still yields something the parent can apply.
    assert handoff_messages([], brief, "h1") == [brief]


def test_the_handoff_drops_the_calls_its_brief_cannot_answer() -> None:
    '''A sibling call's result dies with the abandoned graph, so its call must not travel.'''

    from core.agents.handoff import handoff_messages

    delegating = AIMessage(
        content="",
        name="supervisor",
        id="ai-1",
        additional_kwargs={
            "tool_calls": [
                {"id": "p0", "type": "function", "function": {"name": "plan_update", "arguments": "{}"}},
                {
                    "id": "h1",
                    "type": "function",
                    "function": {"name": "transfer_to_data_agent", "arguments": "{}"},
                },
            ]
        },
        tool_calls=[
            tool_call("plan_update", {"step": 1, "status": "in_progress"}, "p0"),
            tool_call("transfer_to_data_agent", {}, "h1"),
        ],
    )
    brief = ToolMessage(content="TASK BRIEF", name="transfer_to_data_agent", tool_call_id="h1")

    carried = handoff_messages([delegating], brief, "h1")
    assert [call["id"] for call in carried[0].tool_calls] == ["h1"]
    assert [raw["id"] for raw in carried[0].additional_kwargs["tool_calls"]] == ["h1"]
    assert_tool_pairing(carried, label="parallel handoff update")
    # The original is untouched: state is not ours to mutate.
    assert [call["id"] for call in delegating.tool_calls] == ["p0", "h1"]


def test_repair_drops_a_tool_message_whose_call_is_not_in_the_view() -> None:
    '''The exact shape the API rejected, repaired rather than raised.'''

    from core.agents.context import repair_tool_pairing

    orphan = [
        HumanMessage(content="find candidates"),
        ToolMessage(content="TASK BRIEF", name="transfer_to_research_agent", tool_call_id="h1"),
        AIMessage(content="Graph built.", name="research_agent"),
    ]
    assert tool_pairing_problems(orphan), "the fixture should be the broken shape"

    repaired = repair_tool_pairing(orphan)
    assert_tool_pairing(repaired, label="repaired")
    assert [type(message).__name__ for message in repaired] == ["HumanMessage", "AIMessage"]


def test_repair_strips_a_tool_call_nothing_answers() -> None:
    '''The same defect from the other side, and the assistant's words are kept.'''

    from core.agents.context import repair_tool_pairing

    dangling = [
        HumanMessage(content="rank these"),
        AIMessage(
            content="Running the ranking.",
            tool_calls=[tool_call("python_executor", {"code": "x"}, "t1")],
        ),
    ]
    repaired = repair_tool_pairing(dangling)
    assert_tool_pairing(repaired, label="repaired")
    assert len(repaired) == 2
    assert repaired[1].tool_calls == []
    assert repaired[1].content == "Running the ranking."

    # With nothing said and nothing left to call, the message goes entirely.
    silent = [
        HumanMessage(content="rank these"),
        AIMessage(content="", tool_calls=[tool_call("python_executor", {"code": "x"}, "t1")]),
    ]
    assert len(repair_tool_pairing(silent)) == 1


def test_repair_keeps_a_healthy_pair_untouched() -> None:
    '''It must not "fix" a view that was already correct.'''

    from core.agents.context import repair_tool_pairing

    healthy: List[Any] = [
        HumanMessage(content="rank these"),
        AIMessage(
            content="",
            tool_calls=[
                tool_call("read_files", {"file_path": "a.csv"}, "t1"),
                tool_call("read_files", {"file_path": "b.csv"}, "t2"),
            ],
        ),
        ToolMessage(content="a", name="read_files", tool_call_id="t1"),
        ToolMessage(content="b", name="read_files", tool_call_id="t2"),
        AIMessage(content="Both read."),
    ]
    repaired = repair_tool_pairing(healthy)
    assert repaired == healthy
    assert_tool_pairing(repaired, label="healthy")


def test_repair_keeps_the_first_answer_when_a_result_is_duplicated() -> None:
    '''A re-emitted `ToolMessage` answers nothing the second time.'''

    from core.agents.context import repair_tool_pairing

    duplicated = [
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "a.csv"}, "t1")]),
        ToolMessage(content="a", name="read_files", tool_call_id="t1"),
        ToolMessage(content="a", name="read_files", tool_call_id="t1"),
        AIMessage(content="Read."),
    ]
    repaired = repair_tool_pairing(duplicated)
    assert_tool_pairing(repaired, label="deduplicated")
    assert sum(1 for message in repaired if isinstance(message, ToolMessage)) == 1


def test_a_conversation_already_poisoned_by_the_bug_can_continue() -> None:
    '''The safety net is what rescues threads whose checkpoints already hold the orphan.

    Fixing the handoff stops *new* delegations losing their `AIMessage`, but a thread
    that hit this before the fix has the orphaned `ToolMessage` written into its
    checkpoint permanently — every later message on it would fail again. The repair
    pass is what makes those conversations usable, so this test builds the poisoned
    shape by hand rather than through a run.
    '''

    from core.agents.context import build_llm_input_messages

    poisoned = {
        "messages": [
            HumanMessage(content="Find repurposing candidates for AML", id="u-1"),
            AIMessage(content="📋 **PLAN:** ...", name="planning_agent", id="a-1"),
            AIMessage(content="PLAN · run 1 · 0/2 steps resolved", name="plan_init", id="a-2"),
            # The AIMessage that called transfer_to_research_agent is simply not here.
            ToolMessage(
                content="TASK BRIEF for research_agent",
                name="transfer_to_research_agent",
                tool_call_id="h1",
                id="t-1",
            ),
            AIMessage(content="Graph built: 412 drugs.", name="research_agent", id="a-3"),
            HumanMessage(content="now rank those by hERG risk", id="u-2"),
        ],
        "conversation_id": "poisoned",
    }
    assert tool_pairing_problems(poisoned["messages"]), "the fixture should be poisoned"

    for collapse in (True, False):
        view = build_llm_input_messages(poisoned, collapse_delegations=collapse)
        assert_tool_pairing(view, label=f"recovered view (collapse={collapse})")
        # The specialist's result is what mattered in that span, and it survives.
        assert any("412 drugs" in str(message.content) for message in view), view


def test_a_follow_up_turn_stays_paired_through_compression() -> None:
    '''The second turn reaches the anchored-and-compressed path, spans and all.'''

    originals = _roots()
    supervisor = scripted(
        _delegate("research_agent", "h1"),
        AIMessage(
            content="",
            tool_calls=[
                tool_call("plan_update", {"step": 1, "status": "completed", "note": "412"}, "p1")
            ],
        ),
        AIMessage(content="Step 1 resolved."),
        # --- the follow-up turn ---
        _delegate("data_agent", "h2", said="Ranking what we already have."),
        AIMessage(content="Ranked."),
        label="supervisor",
    )
    research = scripted(
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r1")]),
        AIMessage(content="Graph built: 412 drugs."),
        label="research",
    )
    data = scripted(
        AIMessage(content="", tool_calls=[tool_call("read_files", {"file_path": "README.md"}, "r2")]),
        AIMessage(content="Top five written to ranked.csv."),
        label="data",
    )
    models = {
        "supervisor": supervisor,
        "research": research,
        "prediction": scripted(AIMessage(content="unused"), label="prediction"),
        "data": data,
        "planning": scripted(AIMessage(content=PLAN), label="planning"),
        "report": scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nDone."),
            AIMessage(content="# Response Summary\n\n## Answer\nRanked."),
            label="report",
        ),
    }
    workdir = None
    try:
        saver = InMemorySaver()
        workdir = _run(models, thread_id="followup", saver=saver)

        # Second turn down the *same* thread, so the follow-up route sees the history
        # the first run left: collapsed spans, anchored turns and a carried summary.
        import app.config as config
        import backend.utils.output_paths as output_paths

        config.RESULTS_ROOT = workdir
        output_paths.RESULTS_ROOT = workdir
        from core.agents import agentic_system as system

        _install("follow_up")
        graph = system.build_graph(saver, models=models)
        thread = {"configurable": {"thread_id": "followup"}, "recursion_limit": 100}

        async def follow_up() -> None:
            payload = {
                "messages": [HumanMessage(content="now rank those by hERG risk")],
                "conversation_id": "followup",
            }
            async for _ in graph.astream(
                payload, config=thread, stream_mode=["updates"], subgraphs=True
            ):
                pass

        asyncio.run(follow_up())

        assert data.calls, "the follow-up never reached the data agent"
        # The follow-up's own view must hold the first turn's collapsed delegation.
        assert len(supervisor.calls) >= 4, len(supervisor.calls)
        assert_every_model_call_is_valid(models)
    finally:
        _restore_roots(originals)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


def test_only_the_first_of_two_handoffs_in_one_turn_is_delivered() -> None:
    '''Two `transfer_to_*` calls in one turn cannot both be delivered.

    Each returns its own `Command` carrying its own copy of the same `AIMessage`, and
    both copies share its id — so `add_messages` keeps whichever lands last and the
    other brief is left answering a call that is no longer there. The first proceeds,
    the rest stand down, and the plan file still shows their steps as unresolved.
    '''

    from core.agents.handoff import superseded_handoff

    delegating = AIMessage(
        content="Running both steps now.",
        name="supervisor",
        id="ai-1",
        tool_calls=[
            tool_call("transfer_to_research_agent", {}, "h1"),
            tool_call("transfer_to_data_agent", {}, "h2"),
        ],
    )
    assert superseded_handoff([delegating], "h1") is None
    stood_down = superseded_handoff([delegating], "h2")
    assert stood_down and "only the first (research_agent)" in stood_down

    # Written order decides, not the order the tool node happens to run them in.
    reversed_turn = delegating.model_copy(
        update={
            "tool_calls": [
                tool_call("transfer_to_data_agent", {}, "h2"),
                tool_call("transfer_to_research_agent", {}, "h1"),
            ]
        }
    )
    assert superseded_handoff([reversed_turn], "h2") is None
    assert superseded_handoff([reversed_turn], "h1")

    # A single handoff, a non-handoff sibling, and an empty list all pass through.
    single = delegating.model_copy(
        update={"tool_calls": [tool_call("transfer_to_research_agent", {}, "h1")]}
    )
    assert superseded_handoff([single], "h1") is None
    beside_plan_update = delegating.model_copy(
        update={
            "tool_calls": [
                tool_call("plan_update", {"step": 1, "status": "in_progress"}, "p0"),
                tool_call("transfer_to_research_agent", {}, "h1"),
            ]
        }
    )
    assert superseded_handoff([beside_plan_update], "h1") is None
    assert superseded_handoff([], "h1") is None
    assert superseded_handoff([HumanMessage(content="hi")], "h1") is None


def test_a_double_delegation_run_stays_paired_and_resolves() -> None:
    '''End to end: the supervisor asks for two at once, then delegates the second.'''

    originals = _roots()
    supervisor = scripted(
        AIMessage(
            content="Running both steps now.",
            tool_calls=[
                tool_call(
                    "transfer_to_research_agent",
                    {
                        "objective": "Build the AML knowledge graph and report the counts",
                        "inputs": "disease: acute myeloid leukemia",
                        "expected_output": "the pickle path and the drug count",
                    },
                    "h1",
                ),
                tool_call(
                    "transfer_to_prediction_agent",
                    {
                        "objective": "Screen the shortlisted candidates for hERG risk",
                        "inputs": "hERG cutoff 0.5",
                        "expected_output": "the results path and how many were flagged",
                    },
                    "h2",
                ),
            ],
        ),
        # research_agent reported; the supervisor now delegates step 2 on its own turn.
        _delegate("prediction_agent", "h3", said="Now the hERG screen."),
        AIMessage(content="Both steps are resolved."),
        label="supervisor",
    )
    research = scripted(AIMessage(content="Graph built: 412 drugs."), label="research")
    prediction = scripted(AIMessage(content="hERG done; 3 flagged."), label="prediction")
    models = {
        "supervisor": supervisor,
        "research": research,
        "prediction": prediction,
        "data": scripted(AIMessage(content="unused"), label="data"),
        "planning": scripted(AIMessage(content=PLAN), label="planning"),
        "report": scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nDone."), label="report"
        ),
    }
    workdir = None
    try:
        workdir = _run(models, thread_id="double")

        assert research.calls, "the first handoff never landed"
        assert prediction.calls, "the second step never got delegated"
        assert_every_model_call_is_valid(models)

        # Exactly one brief reached each specialist — the superseded one was not delivered.
        final = supervisor.calls[-1]
        briefs = [
            str(getattr(message, "name", ""))
            for message in final
            if isinstance(message, ToolMessage)
            and str(getattr(message, "name", "")).startswith("transfer_to_")
        ]
        assert briefs == ["transfer_to_research_agent", "transfer_to_prediction_agent"], briefs
    finally:
        _restore_roots(originals)
        if workdir:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    import sys

    from tests.harness import run_module_tests

    passed, failed = run_module_tests(dict(globals()), label="tests.test_tool_pairing.")
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
