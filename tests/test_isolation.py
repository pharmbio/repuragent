'''Specialist context isolation.

The claim being tested is narrow and checkable: a specialist's model is given the brief
the supervisor wrote and its own working messages, and **nothing else** — no user
request, no plan, no other specialist's output. The supervisor, conversely, keeps the
conversation but sees each specialist's reported result rather than its tool traffic.

The end-to-end tests assert on what the scripted model was actually handed, which is the
only way to know the middleware did what it claims.
'''

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import List

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import scripted, tool_call
from tests.harness import scoped_results_root

USER_REQUEST = "Find repurposing candidates for acute myeloid leukemia"
PLAN_SECRET = "SECRET-PLAN-TEXT-ONLY-THE-PLANNER-WROTE"
# Appears only in the specialist's own tool traffic, so finding it anywhere else is
# unambiguous evidence of a leak.
TOOL_SENTINEL = "SENTINEL-SPECIALIST-TOOL-TRAFFIC"


def _brief(objective: str, inputs: str, expected: str) -> str:
    from core.agents.handoff import render_brief

    return render_brief(
        agent_name="research_agent",
        objective=objective,
        inputs=inputs,
        expected_output=expected,
    )


# --- The two views, in isolation ----------------------------------------------


def test_a_specialist_sees_only_its_brief_and_its_own_work() -> None:
    from core.agents.context import build_specialist_input_messages

    brief = _brief("Build the AML graph", "MONDO:0018874", "the pickle path and counts")
    messages = [
        HumanMessage(content=USER_REQUEST),
        AIMessage(content=PLAN_SECRET, name="planning_agent"),
        AIMessage(
            content="Delegating.",
            name="supervisor",
            tool_calls=[{"name": "transfer_to_research_agent", "args": {}, "id": "h1"}],
        ),
        ToolMessage(content=brief, name="transfer_to_research_agent", tool_call_id="h1"),
        # The specialist's own working messages.
        AIMessage(
            content="",
            name="research_agent",
            tool_calls=[{"name": "search_disease_id", "args": {"disease_name": "AML"}, "id": "t1"}],
        ),
        ToolMessage(content="MONDO:0018874", name="search_disease_id", tool_call_id="t1"),
    ]

    view = build_specialist_input_messages({"messages": messages}, agent_name="research_agent")
    rendered = "\n".join(str(getattr(message, "content", "")) for message in view)

    assert "TASK BRIEF" in rendered and "MONDO:0018874" in rendered
    assert USER_REQUEST not in rendered, "the conversation must not reach a specialist"
    assert PLAN_SECRET not in rendered, "the plan must not reach a specialist"
    assert "Delegating." not in rendered, "the supervisor's own turn must not reach it"
    # Its own tool traffic is present, so its react loop still works.
    assert any(isinstance(message, ToolMessage) for message in view)
    # The window opens with the brief, so it never starts on an orphaned tool call.
    assert isinstance(view[0], HumanMessage)


def test_a_second_delegation_does_not_carry_the_first() -> None:
    from core.agents.context import build_specialist_input_messages

    first = _brief("Build the graph", "MONDO:1", "counts")
    second = _brief("Extract the drugs", "/tmp/kg.pkl", "a CSV path")
    messages = [
        HumanMessage(content=USER_REQUEST),
        ToolMessage(content=first, name="transfer_to_research_agent", tool_call_id="h1"),
        AIMessage(content="Graph built with 412 drugs.", name="research_agent"),
        AIMessage(content="Next step.", name="supervisor"),
        ToolMessage(content=second, name="transfer_to_research_agent", tool_call_id="h2"),
    ]
    view = build_specialist_input_messages({"messages": messages}, agent_name="research_agent")
    rendered = "\n".join(str(getattr(message, "content", "")) for message in view)

    assert "Extract the drugs" in rendered
    assert "Build the graph" not in rendered, "an earlier brief must not leak forward"
    assert "412 drugs" not in rendered, "an earlier result must be re-stated, not inherited"


def test_a_specialist_without_a_brief_still_gets_context() -> None:
    '''Degrading to more context beats handing a model an empty prompt.'''

    from core.agents.context import build_specialist_input_messages

    messages = [HumanMessage(content=USER_REQUEST), AIMessage(content="hi", name="supervisor")]
    view = build_specialist_input_messages({"messages": messages}, agent_name="research_agent")
    assert view, "an unbriefed specialist must not be given nothing"
    assert USER_REQUEST in "\n".join(str(m.content) for m in view)


def test_the_supervisor_sees_results_not_tool_dumps() -> None:
    from core.agents.context import collapse_specialist_spans

    dump = "X" * 40000
    messages = [
        HumanMessage(content=USER_REQUEST),
        AIMessage(
            content="Delegating.",
            name="supervisor",
            tool_calls=[{"name": "transfer_to_research_agent", "args": {}, "id": "h1"}],
        ),
        ToolMessage(content="TASK BRIEF for research_agent", name="transfer_to_research_agent", tool_call_id="h1"),
        AIMessage(
            content="",
            name="research_agent",
            tool_calls=[{"name": "create_knowledge_graph", "args": {}, "id": "t1"}],
        ),
        ToolMessage(content=dump, name="create_knowledge_graph", tool_call_id="t1"),
        AIMessage(content="Graph built: 412 drugs, 88 proteins. Path /tmp/kg.pkl", name="research_agent"),
        AIMessage(content="Recording step 1.", name="supervisor"),
    ]

    collapsed = collapse_specialist_spans(messages)
    rendered = "\n".join(str(getattr(message, "content", "")) for message in collapsed)

    assert "412 drugs, 88 proteins" in rendered, "the reported result must survive"
    assert "TASK BRIEF" in rendered, "the brief stays, so the supervisor recalls what it asked"
    assert dump not in rendered, "the specialist's tool payload must not reach the supervisor"
    # And the span leaves no orphaned tool call behind.
    call_ids = {
        call["id"]
        for message in collapsed
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    }
    answered = {
        message.tool_call_id for message in collapsed if isinstance(message, ToolMessage)
    }
    assert call_ids <= answered, (call_ids, answered)


def test_a_specialist_span_with_no_report_is_kept_intact() -> None:
    '''Never silently empty a span: without a summary the raw work is the evidence.'''

    from core.agents.context import collapse_specialist_spans

    messages = [
        ToolMessage(content="brief", name="transfer_to_data_agent", tool_call_id="h1"),
        AIMessage(
            content="",
            name="data_agent",
            tool_calls=[{"name": "python_executor", "args": {}, "id": "t1"}],
        ),
        ToolMessage(content="ranked 20 rows", name="python_executor", tool_call_id="t1"),
    ]
    collapsed = collapse_specialist_spans(messages)
    assert len(collapsed) == 3
    assert "ranked 20 rows" in "\n".join(str(m.content) for m in collapsed)


# --- End to end ----------------------------------------------------------------


def _system_and_messages(calls: List[List]) -> List[str]:
    '''Everything one model was handed, per invocation, flattened to text.

    Parameters:
    ---------
    calls (list): the recorded invocations of one scripted model.

    Returns:
    ----------
    texts (list): everything that model was handed per invocation, flattened, so the isolation assertions can search it.
    '''

    rendered = []
    for call in calls:
        rendered.append("\n".join(str(getattr(message, "content", "")) for message in call))
    return rendered


def test_isolation_holds_through_a_real_run() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="repuragent-isolation-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    originals = (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)
    try:
        config.RESULTS_ROOT = workdir
        output_paths.RESULTS_ROOT = workdir
        from core.agents import agentic_system as system

        plan = (
            "📋 **PLAN:** Find AML candidates\n\n"
            "**TASK BREAKDOWN:**\n"
            "  [1] **Build the AML knowledge graph**\n"
            "      **Agent:** research_agent\n"
            f"      **Details:** {PLAN_SECRET}\n"
        )

        supervisor = scripted(
            AIMessage(
                content="Delegating the graph build.",
                tool_calls=[
                    tool_call(
                        "transfer_to_research_agent",
                        {
                            "objective": "Build the AML disease knowledge graph",
                            "inputs": "disease: acute myeloid leukemia",
                            "expected_output": "the pickle path and the drug count",
                        },
                        "h1",
                    )
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("plan_update", {"step": 1, "status": "completed", "note": "412 drugs"}, "p1")
                ],
            ),
            AIMessage(content="Step 1 is resolved."),
        )
        research = scripted(
            AIMessage(
                content="",
                tool_calls=[tool_call("search_disease_id", {"disease_name": TOOL_SENTINEL}, "t1")],
            ),
            AIMessage(content="Graph built at /tmp/kg.pkl with 412 drugs."),
        )
        report = scripted(AIMessage(content="# Response Summary\n\n## Answer\nDone."))

        system.reset_model_cache()
        system.install_helper_models(
            task_classifier=RunnableLambda(lambda _p: system.TaskClassification(category="complex")),
            approval_judge=RunnableLambda(
                lambda _p: system.PlanFeedbackVerdict(decision="approve", constraints=[])
            ),
            context_summarizer=RunnableLambda(lambda _p: system.ContextDigest(summary="s")),
        )
        graph = system.build_graph(
            InMemorySaver(),
            models={
                "supervisor": supervisor,
                "research": research,
                "prediction": scripted(AIMessage(content="unused")),
                "data": scripted(AIMessage(content="unused")),
                "planning": scripted(AIMessage(content=plan)),
                "report": report,
            },
        )
        thread = {"configurable": {"thread_id": "iso"}, "recursion_limit": 80}

        async def run():
            payload = {
                "messages": [HumanMessage(content=USER_REQUEST)],
                "conversation_id": "iso",
            }
            async for _ in graph.astream(payload, config=thread, stream_mode=["updates"], subgraphs=True):
                pass
            async for _ in graph.astream(
                Command(resume="approved"), config=thread, stream_mode=["updates"], subgraphs=True
            ):
                pass

        asyncio.run(run())

        # --- what the research agent was actually handed ---------------------
        assert research.calls, "the research agent never ran"
        research_context = "\n\n".join(_system_and_messages(research.calls))
        assert "TASK BRIEF for research_agent" in research_context
        assert "Build the AML disease knowledge graph" in research_context
        assert USER_REQUEST not in research_context, "the user's request reached a specialist"
        assert PLAN_SECRET not in research_context, "the plan reached a specialist"
        assert "Delegating the graph build." not in research_context
        # It is told why its context is narrow, so it reports gaps instead of guessing.
        assert "WHAT YOU CAN SEE" in research_context
        # It still has its operating scope, or it could not write anything.
        assert "Active output scope" in research_context

        # --- what the supervisor was handed ---------------------------------
        supervisor_context = "\n\n".join(_system_and_messages(supervisor.calls))
        assert USER_REQUEST in supervisor_context, "the supervisor must hold the conversation"
        assert "Graph built at /tmp/kg.pkl with 412 drugs." in supervisor_context
        # The specialist's own tool traffic must not reach the supervisor; only the
        # result it reported. The sentinel exists solely so this needle cannot collide
        # with the supervisor's prompt, which names a MONDO id as an example brief.
        assert TOOL_SENTINEL not in supervisor_context, (
            "specialist tool traffic reached the supervisor"
        )

        # --- the report still has the full evidence -------------------------
        report_context = "\n\n".join(_system_and_messages(report.calls))
        assert "Graph built at /tmp/kg.pkl" in report_context
        assert USER_REQUEST in report_context

        # --- and the run still completed normally ---------------------------
        plan_file = workdir / "iso" / "plan.md"
        assert "_Outcome: All 1 steps completed._" in plan_file.read_text()
    finally:
        config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals
        shutil.rmtree(workdir, ignore_errors=True)


def test_the_brief_is_rendered_completely() -> None:
    from core.agents.handoff import render_brief

    brief = render_brief(
        agent_name="prediction_agent",
        objective="Run the hERG classifier",
        inputs="/tmp/candidates.csv (column `smiles`)",
        expected_output="the results path and the flagged count",
        context="Step 2 produced 20 candidates.",
    )
    for fragment in (
        "TASK BRIEF for prediction_agent",
        "Objective: Run the hERG classifier",
        "Inputs: /tmp/candidates.csv",
        "Expected output: the results path",
        "Background: Step 2 produced 20 candidates.",
    ):
        assert fragment in brief, fragment

    # Background is omitted rather than left as an empty label.
    assert "Background:" not in render_brief(
        agent_name="data_agent", objective="a", inputs="b", expected_output="c"
    )


def test_the_handoff_tool_requires_a_complete_brief() -> None:
    '''A free-text `task` let the supervisor be vague; these fields cannot be.'''

    from core.agents.handoff import build_handoff_tools

    for handoff in build_handoff_tools():
        # `tool_call_schema` is what the model is asked for; `args_schema` also holds
        # the injected call id, which the model never supplies.
        schema = handoff.tool_call_schema.model_json_schema()
        required = set(schema.get("required") or [])
        properties = set(schema.get("properties") or {})
        assert {"objective", "inputs", "expected_output"} <= required, (handoff.name, required)
        assert "context" in properties and "context" not in required
        assert "tool_call_id" not in properties




# --- The brief must be usable ---------------------------------------------------


def test_a_dangling_reference_is_refused() -> None:
    '''"The file from the previous step" resolves to nothing for an isolated agent.'''

    from core.agents.handoff import validate_brief

    for bad in (
        "the file from the previous step",
        "the candidates you found earlier",
        "same as above",
        "output of the prior run",
        "the shortlist that we produced",
    ):
        problem = validate_brief(
            agent_name="data_agent",
            objective="Rank the candidates by composite score",
            inputs=bad,
            expected_output="the ranked CSV path",
        )
        assert problem is not None, bad
        assert "cannot see any earlier step" in problem, (bad, problem)

    # The same phrasing is legitimate prose in `context`, which is not checked.
    assert (
        validate_brief(
            agent_name="data_agent",
            objective="Rank the candidates by composite score",
            inputs="score column: composite_score",
            expected_output="the ranked CSV path",
        )
        is None
    )


def test_an_artifact_must_say_what_it_holds() -> None:
    import tempfile
    from pathlib import Path

    from core.agents.handoff import validate_brief

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as handle:
        real = Path(handle.name)
    try:
        base = dict(
            agent_name="data_agent",
            objective="Rank the shortlisted candidates",
            inputs="none",
            expected_output="the ranked CSV path",
        )
        # A bare path says where, not what.
        problem = validate_brief(**base, artifacts=[str(real)])
        assert problem and "what it holds" in problem

        # With a description it passes.
        assert validate_brief(**base, artifacts=[f"{real} — 20 candidates with SMILES"]) is None

        # A path that does not exist is caught here rather than a delegation later.
        problem = validate_brief(**base, artifacts=["/nope/ghost.csv — 20 candidates"])
        assert problem and "does not exist" in problem

        # A relative path is the specialist's to resolve, so it is left alone.
        assert validate_brief(**base, artifacts=["candidates.csv — 20 candidates"]) is None
    finally:
        real.unlink(missing_ok=True)


def test_a_thin_objective_or_missing_output_is_refused() -> None:
    from core.agents.handoff import validate_brief

    problem = validate_brief(
        agent_name="research_agent",
        objective="do step 2",
        inputs="none",
        expected_output="the counts",
    )
    assert problem and "too short to act on" in problem

    problem = validate_brief(
        agent_name="research_agent",
        objective="Build the AML disease knowledge graph and report the counts",
        inputs="none",
        expected_output="",
    )
    assert problem and "`expected_output` is missing" in problem


def test_the_brief_renders_artifacts_and_constraints() -> None:
    from core.agents.handoff import render_brief

    brief = render_brief(
        agent_name="prediction_agent",
        objective="Screen the 20 shortlisted candidates for cardiotoxicity risk",
        inputs="none beyond the file below",
        expected_output="the results CSV path and the flagged count",
        artifacts=["/x/master_candidates.csv — 20 shortlisted drugs, SMILES in `smiles`"],
        constraints=["hERG is the endpoint that matters here"],
        context="Step 2 narrowed 412 candidates to 20.",
    )
    for fragment in (
        "TASK BRIEF for prediction_agent",
        "Objective: Screen the 20 shortlisted candidates",
        "Files produced by earlier steps, and what they hold:",
        "  - /x/master_candidates.csv — 20 shortlisted drugs",
        "Constraints for this step:",
        "  - hERG is the endpoint that matters here",
        "Expected output: the results CSV path",
        "Background: Step 2 narrowed 412 candidates to 20.",
    ):
        assert fragment in brief, fragment

    # Empty optional sections are omitted, not left as bare labels.
    minimal = render_brief(
        agent_name="data_agent",
        objective="Rank the candidates by composite score",
        inputs="none",
        expected_output="the ranked CSV path",
        artifacts=[],
        constraints=[""],
    )
    assert "Files produced" not in minimal and "Constraints" not in minimal
    assert "Background" not in minimal


def test_a_refused_brief_comes_back_to_the_supervisor() -> None:
    '''The supervisor gets the reason and can rewrite it — the specialist never runs.'''

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.types import Command

    from core.agents.handoff import build_handoff_tools
    from tests.fakes import tool_call

    handoff = next(tool for tool in build_handoff_tools() if tool.name == "transfer_to_data_agent")

    refused = handoff.invoke(
        {
            "name": "transfer_to_data_agent",
            "type": "tool_call",
            "id": "h1",
            "args": {
                "objective": "Rank the candidates by composite score",
                "inputs": "the file from the previous step",
                "expected_output": "the ranked CSV path",
                "state": {"messages": []},
            },
        }
    )
    content = getattr(refused, "content", refused)
    assert not isinstance(content, Command)
    assert "Brief not delivered to data_agent" in str(content)
    assert "cannot see any earlier step" in str(content)

    # The supervisor's own turn, as the inner agent state holds it when the tool runs.
    delegating = AIMessage(
        content="Handing the ranking over.",
        name="supervisor",
        id="ai-delegate",
        tool_calls=[tool_call("transfer_to_data_agent", {}, "h2")],
    )
    accepted = handoff.invoke(
        {
            "name": "transfer_to_data_agent",
            "type": "tool_call",
            "id": "h2",
            "args": {
                "objective": "Rank the 20 shortlisted candidates by composite score",
                "inputs": "score weights: 0.5 target coverage, 0.5 ADMET",
                "expected_output": "the ranked CSV path and the top five names",
                "artifacts": ["candidates.csv — 20 shortlisted drugs with ADMET columns"],
                "state": {
                    "messages": [
                        HumanMessage(content="rank the shortlist", id="h-0"),
                        delegating,
                    ]
                },
            },
        }
    )
    assert isinstance(accepted, Command)
    assert accepted.goto == "data_agent"

    carried = accepted.update["messages"]
    brief = carried[-1]
    assert "Files produced by earlier steps" in brief.content

    # The delegating AIMessage must travel up with the brief. `Command.PARENT` abandons
    # the supervisor's own graph, so an update of the brief alone leaves the parent
    # holding a `tool` message answering a `tool_call` it cannot see, and the next model
    # call is rejected with "messages with role 'tool' must be a response to a
    # preceeding message with 'tool_calls'".
    assert carried[-2] is delegating or carried[-2].id == delegating.id, carried
    assert [call["id"] for call in carried[-2].tool_calls] == ["h2"]
    assert brief.tool_call_id == "h2"


def test_the_handoff_tool_asks_for_every_part_of_the_brief() -> None:
    from core.agents.handoff import build_handoff_tools

    for handoff in build_handoff_tools():
        schema = handoff.tool_call_schema.model_json_schema()
        properties = set(schema.get("properties") or {})
        assert {
            "objective",
            "inputs",
            "expected_output",
            "artifacts",
            "constraints",
            "context",
        } <= properties, (handoff.name, sorted(properties))
        # The half of brief quality no validation can check is the level of detail, so
        # the guidance has to reach the model through the field descriptions themselves.
        objective_help = str(schema["properties"]["objective"].get("description", "")).lower()
        assert "must be true" in objective_help, objective_help
        assert "specialist's decision" in objective_help, objective_help
        artifacts_help = str(schema["properties"]["artifacts"].get("description", "")).lower()
        assert "what it contains" in artifacts_help, artifacts_help


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
