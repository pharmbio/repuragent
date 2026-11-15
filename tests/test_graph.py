'''End-to-end: classify -> plan -> approve -> delegate -> track -> report.

The one test that proves the architecture holds together. Every model call is
scripted, so it runs offline, and the assertions are about the things a refactor
could plausibly break: node order, the handoff actually reaching a specialist,
`plan.md` contents after the run, and the interrupt/resume cycle.
'''

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from tests.fakes import scripted, tool_call

INTERNAL_NODES = {"model", "tools", "agent", "__start__", "__end__", "__pregel_pull"}

PLAN_TEXT = """📋 **PLAN:** Find repurposing candidates for acute myeloid leukemia

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


def _resolve_node(namespace, metadata=None, default=None) -> str:
    node = (metadata or {}).get("langgraph_node")
    if node and node not in INTERNAL_NODES:
        return node
    for entry in reversed(tuple(namespace or ())):
        candidate = str(entry).split(":", 1)[0]
        if candidate and candidate not in INTERNAL_NODES:
            return candidate
    return node or default or "agent"


async def _drive(app, payload, config) -> Tuple[List[str], List[Dict[str, Any]]]:
    '''Run the graph, returning the node order and any interrupt payloads.

    Parameters:
    ---------
    app (CompiledGraph): the compiled graph to run.
    payload (dict): the graph input.
    config (dict): the run config, carrying the thread id.

    Returns:
    ----------
    result (tuple): `(node order, interrupt payloads)`, which is what the route assertions are made against.
    '''

    order: List[str] = []
    interrupts: List[Dict[str, Any]] = []
    async for namespace, mode, data in app.astream(
        payload, config=config, stream_mode=["messages", "updates"], subgraphs=True
    ):
        if mode != "updates" or not isinstance(data, dict):
            continue
        for node, update in data.items():
            if node == "__interrupt__":
                interrupts.extend(
                    getattr(item, "value", item) for item in tuple(update or ())
                )
                continue
            order.append(_resolve_node(namespace, None, node))
    return order, interrupts


def test_full_complex_run() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="repuragent-graph-"))
    try:
        import app.config as config

        original_results = config.RESULTS_ROOT
        config.RESULTS_ROOT = workdir
        import backend.utils.output_paths as output_paths

        original_op_results = output_paths.RESULTS_ROOT
        output_paths.RESULTS_ROOT = workdir

        from core.agents import agentic_system as system

        # The specialists and the supervisor share one script per role.
        supervisor = scripted(
            AIMessage(
                content="Delegating the graph build.",
                tool_calls=[tool_call(
                    "transfer_to_research_agent",
                    {
                        "objective": "Build the AML disease knowledge graph and report the counts",
                        "inputs": "disease: acute myeloid leukemia (resolve the MONDO id first)",
                        "expected_output": "the pickle path plus drug/protein/pathway counts",
                    },
                    "h1",
                )],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "plan_update",
                        {"step": 1, "status": "completed", "note": "412 drugs, 88 proteins"},
                        "p1",
                    )
                ],
            ),
            AIMessage(
                content="Delegating the hERG screen.",
                tool_calls=[tool_call(
                    "transfer_to_prediction_agent",
                    {
                        "objective": "Run the hERG classifier over the candidate SMILES",
                        "inputs": "/tmp/candidates.csv (column `smiles`)",
                        "expected_output": "the results CSV path and how many were flagged",
                    },
                    "h2",
                )],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    tool_call("plan_update", {"step": 2, "status": "completed", "note": "3 flagged"}, "p2")
                ],
            ),
            AIMessage(content="Both steps are resolved."),
        )
        research = scripted(AIMessage(content="Knowledge graph built: 412 drugs, 88 proteins."))
        prediction = scripted(AIMessage(content="hERG run complete; 3 of 20 flagged."))
        data = scripted(AIMessage(content="(not used in this run)"))
        planning = scripted(AIMessage(content=PLAN_TEXT))
        report = scripted(
            AIMessage(content="# Response Summary\n\n## Answer\nTwo candidates survive.")
        )

        system.reset_model_cache()
        system.install_helper_models(
            task_classifier=RunnableLambda(
                lambda _payload: system.TaskClassification(category="complex")
            ),
            approval_judge=RunnableLambda(
                lambda _payload: system.PlanFeedbackVerdict(
                    decision="approve", constraints=["Only clinical phase 3 or later"]
                )
            ),
            context_summarizer=RunnableLambda(
                lambda _payload: system.ContextDigest(
                    summary="Built the AML graph and screened for hERG risk."
                )
            ),
        )

        app_graph = system.build_graph(
            InMemorySaver(),
            models={
                "supervisor": supervisor,
                "research": research,
                "prediction": prediction,
                "data": data,
                "planning": planning,
                "report": report,
            },
        )
        thread = {"configurable": {"thread_id": "conv"}, "recursion_limit": 80}

        # --- first turn: plan, then pause for approval ------------------------
        order, interrupts = asyncio.run(
            _drive(
                app_graph,
                {
                    "messages": [HumanMessage(content="Find repurposing candidates for AML")],
                    "conversation_id": "conv",
                },
                thread,
            )
        )
        assert "planning_agent" in order, order
        assert len(interrupts) == 1, interrupts
        assert interrupts[0]["type"] == "plan_review"
        assert "TASK BREAKDOWN" in interrupts[0]["plan"]

        snapshot = asyncio.run(app_graph.aget_state(thread))
        assert "human_chat" in tuple(snapshot.next or ()), snapshot.next

        # --- second turn: approve and run ------------------------------------
        order, interrupts = asyncio.run(
            _drive(app_graph, Command(resume="approved, phase 3 only"), thread)
        )
        assert not interrupts, interrupts

        # Delegation actually reached both specialists, and in plan order.
        for expected in ("plan_init", "supervisor", "research_agent", "prediction_agent",
                         "plan_finalize", "report_agent_complex"):
            assert expected in order, f"{expected} missing from {order}"
        assert order.index("research_agent") < order.index("prediction_agent"), order
        assert order.index("plan_init") < order.index("supervisor"), order
        assert order.index("plan_finalize") > order.index("prediction_agent"), order
        assert "report_agent_simple" not in order, order

        final = asyncio.run(app_graph.aget_state(thread))
        assert final.next == (), final.next
        assert final.values["task_category"] == "complex"
        assert final.values["approval_constraints"] == ["Only clinical phase 3 or later"]

        # --- the plan file is the record -------------------------------------
        plan_file = workdir / "conv" / "plan.md"
        assert plan_file.exists(), sorted(workdir.rglob("*"))
        document = plan_file.read_text()
        assert "## Run 1 · complex" in document
        assert "**Goal:** Find repurposing candidates for acute myeloid leukemia" in document
        assert "Only clinical phase 3 or later" in document
        assert "- [x] **1.** Build the AML knowledge graph · @research_agent · `completed`" in document
        assert "- [x] **2.** Screen the shortlist for hERG risk · @prediction_agent · `completed`" in document
        assert "Note: 412 drugs, 88 proteins" in document
        assert "_Progress: 2/2 steps resolved._" in document
        assert "_Outcome: All 2 steps completed._" in document

        # The report is the last thing the user sees.
        last = final.values["messages"][-1]
        assert last.name == "context_summary" or last.name == "report_agent_complex", last.name
        reports = [
            message
            for message in final.values["messages"]
            if getattr(message, "name", "") == "report_agent_complex"
        ]
        assert reports and reports[-1].content.startswith("# Response Summary"), reports

        # Every list the middleware assembled has to be one a provider would accept.
        # A scripted model takes anything, so without this the run can look healthy
        # while the supervisor's own view holds a `tool` message answering a
        # `tool_call` that is not in it. See tests/test_tool_pairing.py.
        from tests.harness import assert_every_model_call_is_valid

        assert_every_model_call_is_valid(
            {
                "supervisor": supervisor,
                "research": research,
                "prediction": prediction,
                "planning": planning,
                "report": report,
            }
        )
    finally:
        import app.config as config
        import backend.utils.output_paths as output_paths

        config.RESULTS_ROOT = original_results
        output_paths.RESULTS_ROOT = original_op_results
        shutil.rmtree(workdir, ignore_errors=True)


def test_graph_shape() -> None:
    '''The node set and the wiring that carries the deterministic plan flow.'''

    from core.agents.agentic_system import build_graph

    fake = scripted()
    app_graph = build_graph(
        None,
        models={key: fake for key in ("supervisor", "research", "prediction", "data", "planning", "report")},
    )
    graph = app_graph.get_graph()
    nodes = {name for name in graph.nodes if not name.startswith("__")}
    assert {
        "task_classifier",
        "planning_agent",
        "human_chat",
        "approval_ack",
        "plan_init",
        "execution",
        "plan_finalize",
        "report_agent_complex",
        "report_agent_simple",
        "report_agent_meta",
    } <= nodes, nodes

    edges = {(edge.source, edge.target) for edge in graph.edges}
    assert ("plan_init", "execution") in edges
    assert ("execution", "plan_finalize") in edges
    assert ("approval_ack", "plan_init") in edges
    # The supervisor must not be reachable from anywhere except plan_init, and the
    # execution subgraph must be the only thing feeding plan_finalize.
    assert not any(source == "supervisor" for source, _ in edges), edges


def _build_for_route(category: str, workdir: Path, *, supervisor_script=None):
    '''A graph whose classifier always answers `category`.

    Parameters:
    ---------
    category (str): the task category the stub classifier always returns.
    workdir (Path): the temporary scope this graph writes `plan.md` into.
    supervisor_script (list): scripted supervisor turns, or None for the default.

    Returns:
    ----------
    app (CompiledGraph): a graph pinned to that route.
    '''

    import app.config as config
    import backend.utils.output_paths as output_paths

    config.RESULTS_ROOT = workdir
    output_paths.RESULTS_ROOT = workdir

    from core.agents import agentic_system as system

    supervisor = supervisor_script or scripted(
        AIMessage(
            content="",
            tool_calls=[tool_call("plan_update", {"step": 1, "status": "completed", "note": "done"}, "p1")],
        ),
        AIMessage(content="Finished."),
    )
    system.reset_model_cache()
    system.install_helper_models(
        task_classifier=RunnableLambda(lambda _p: system.TaskClassification(category=category)),
        approval_judge=RunnableLambda(
            lambda _p: system.PlanFeedbackVerdict(decision="approve", constraints=[])
        ),
        context_summarizer=RunnableLambda(lambda _p: system.ContextDigest(summary="s")),
    )
    graph = system.build_graph(
        InMemorySaver(),
        models={
            "supervisor": supervisor,
            "research": scripted(AIMessage(content="research result")),
            "prediction": scripted(AIMessage(content="prediction result")),
            "data": scripted(AIMessage(content="data result")),
            "planning": scripted(AIMessage(content=PLAN_TEXT)),
            "report": scripted(AIMessage(content="# Response Summary\n## Answer\nok")),
        },
    )
    return graph


def test_simple_route_skips_planning_and_approval() -> None:
    '''A short request should not make the user approve a one-step plan.'''

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-simple-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    originals = (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)
    try:
        graph = _build_for_route("simple", workdir)
        thread = {"configurable": {"thread_id": "simple"}, "recursion_limit": 60}
        order, interrupts = asyncio.run(
            _drive(
                graph,
                {
                    "messages": [HumanMessage(content="What is the hERG prediction for aspirin?")],
                    "conversation_id": "simple",
                },
                thread,
            )
        )
        assert not interrupts, "a simple request must not stop for approval"
        assert "planning_agent" not in order, order
        assert "human_chat" not in order, order
        for expected in ("plan_init", "supervisor", "plan_finalize", "report_agent_simple"):
            assert expected in order, f"{expected} missing from {order}"
        assert "report_agent_complex" not in order, order

        # Even a plan-less route is recorded, so the document covers the whole
        # conversation and a follow-up can read it.
        document = (workdir / "simple" / "plan.md").read_text()
        assert "## Run 1 · simple" in document
        assert "What is the hERG prediction for aspirin?" in document
        assert "_Progress: 1/1 steps resolved._" in document
    finally:
        config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals
        shutil.rmtree(workdir, ignore_errors=True)


def test_meta_query_answers_without_executing() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="repuragent-meta-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    originals = (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)
    try:
        graph = _build_for_route("meta_query", workdir)
        thread = {"configurable": {"thread_id": "meta"}, "recursion_limit": 40}
        order, interrupts = asyncio.run(
            _drive(
                graph,
                {
                    "messages": [HumanMessage(content="Which ADMET models do you have?")],
                    "conversation_id": "meta",
                },
                thread,
            )
        )
        assert not interrupts
        assert "report_agent_meta" in order, order
        # No execution machinery at all, and no plan file.
        for forbidden in ("plan_init", "execution", "supervisor", "plan_finalize"):
            assert forbidden not in order, (forbidden, order)
        assert not (workdir / "meta" / "plan.md").exists()
    finally:
        config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals
        shutil.rmtree(workdir, ignore_errors=True)


def test_a_blocked_step_is_recorded_not_hidden() -> None:
    '''The plan file must show a step that could not be completed.'''

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-blocked-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    originals = (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)
    try:
        supervisor = scripted(
            AIMessage(
                content="",
                tool_calls=[
                    tool_call(
                        "plan_update",
                        {"step": 1, "status": "blocked", "note": "No SMILES could be resolved"},
                        "p1",
                    )
                ],
            ),
            AIMessage(content="Blocked; stopping here."),
        )
        graph = _build_for_route("simple", workdir, supervisor_script=supervisor)
        thread = {"configurable": {"thread_id": "blocked"}, "recursion_limit": 60}
        asyncio.run(
            _drive(
                graph,
                {
                    "messages": [HumanMessage(content="Predict hERG for compound X")],
                    "conversation_id": "blocked",
                },
                thread,
            )
        )
        document = (workdir / "blocked" / "plan.md").read_text()
        assert "`blocked`" in document
        assert "Note: No SMILES could be resolved" in document
        # A blocked step is terminal, so the run *is* resolved — but the outcome must
        # not read as success. "All N steps completed" is reserved for exactly that.
        assert "_Outcome: 0/1 steps completed; blocked: 1._" in document
        assert "_Progress: 1/1 steps resolved._" in document
    finally:
        config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals
        shutil.rmtree(workdir, ignore_errors=True)


def test_revision_loop_creates_no_spurious_runs() -> None:
    '''Revising a plan several times must not leave draft runs in the document.'''

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-revise-"))
    import app.config as config
    import backend.utils.output_paths as output_paths

    originals = (config.RESULTS_ROOT, output_paths.RESULTS_ROOT)
    try:
        config.RESULTS_ROOT = workdir
        output_paths.RESULTS_ROOT = workdir
        from core.agents import agentic_system as system

        verdicts = iter(
            [
                system.PlanFeedbackVerdict(decision="revise", constraints=[]),
                system.PlanFeedbackVerdict(decision="approve", constraints=[]),
            ]
        )
        system.reset_model_cache()
        system.install_helper_models(
            task_classifier=RunnableLambda(lambda _p: system.TaskClassification(category="complex")),
            approval_judge=RunnableLambda(lambda _p: next(verdicts)),
            context_summarizer=RunnableLambda(lambda _p: system.ContextDigest(summary="s")),
        )
        graph = system.build_graph(
            InMemorySaver(),
            models={
                "supervisor": scripted(AIMessage(content="Nothing to do.")),
                "research": scripted(AIMessage(content="r")),
                "prediction": scripted(AIMessage(content="p")),
                "data": scripted(AIMessage(content="d")),
                "planning": scripted(
                    AIMessage(content=PLAN_TEXT), AIMessage(content=PLAN_TEXT.replace("Screen", "Rescreen"))
                ),
                "report": scripted(AIMessage(content="# Response Summary\n## Answer\nok")),
            },
        )
        thread = {"configurable": {"thread_id": "revise"}, "recursion_limit": 80}

        asyncio.run(
            _drive(
                graph,
                {
                    "messages": [HumanMessage(content="Find AML candidates")],
                    "conversation_id": "revise",
                },
                thread,
            )
        )
        plan_file = workdir / "revise" / "plan.md"
        assert not plan_file.exists(), "no plan file before approval"

        # First reply asks for changes: the planner runs again and pauses again.
        _, interrupts = asyncio.run(_drive(graph, Command(resume="use the STEL instead"), thread))
        assert interrupts, "a revision must come back for another review"
        assert not plan_file.exists(), "a revision must not create a run"

        # Second reply approves.
        asyncio.run(_drive(graph, Command(resume="approved"), thread))
        document = plan_file.read_text()
        assert document.count("## Run ") == 1, document
        assert "Rescreen" in document, "the approved revision is what got written"
    finally:
        config.RESULTS_ROOT, output_paths.RESULTS_ROOT = originals
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
