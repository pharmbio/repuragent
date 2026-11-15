'''The plan file: parsing, round-tripping, validation, accounting.

This is the module the deterministic-tracking change rests on, so the tests are
about the properties the supervisor and the UI panel both rely on: a plan parses the
same way every time, what is written can be read back exactly, an invalid update is
refused with a usable message, and progress is arithmetic rather than narration.
'''

from __future__ import annotations

from pathlib import Path

from tests.harness import scoped_results_root

PLAN = """📋 **PLAN:** Identify repurposing candidates for acute myeloid leukemia

**TASK BREAKDOWN:**
  [1] **Build the AML knowledge graph**
      **Agent:** Research Agent
      **Details:** Resolve the MONDO id, then build the disease graph.
      **Depends on:** none

  [2] **Extract drug, protein and pathway candidates**
      **Agent:** research agent
      **Details:** Pull candidates from the pickled graph.
      **Depends on:** 1

  [3] **Screen the shortlist across the ADMET panel**
      **Agent:** prediction
      **Details:** hERG, Ames, BBB and the CYP panel.

  [4] **Rank and visualise**
      **Agent:** data analysis
      **Details:** Composite score plus a target-coverage heatmap.

**ASSUMPTIONS AND OPEN QUESTIONS:**
- Candidate count is unknown until the graph is built.
"""


def test_parses_the_canonical_format() -> None:
    from backend.utils import plan_store

    steps = plan_store.parse_plan_steps(PLAN)
    assert [step.number for step in steps] == [1, 2, 3, 4]
    # Whatever wording the planner used for the agent, it maps to a real node name,
    # because the UI labels steps by it.
    assert [step.agent for step in steps] == [
        "research_agent",
        "research_agent",
        "prediction_agent",
        "data_agent",
    ]
    assert steps[0].details.startswith("Resolve the MONDO id")
    assert steps[1].depends_on == "1"
    assert plan_store.parse_plan_goal(PLAN).startswith("Identify repurposing candidates")


def test_falls_back_rather_than_failing() -> None:
    '''A differently phrased plan must still execute, with a degraded ledger.'''

    from backend.utils import plan_store

    numbered = plan_store.parse_plan_steps("1. Build the graph\n2) Screen the candidates")
    assert [step.title for step in numbered] == ["Build the graph", "Screen the candidates"]

    catch_all = plan_store.parse_plan_steps("Just find me some candidates for AML.")
    assert len(catch_all) == 1
    assert catch_all[0].title == "Just find me some candidates for AML."

    # Steps are renumbered densely, so a step number is always addressable.
    sparse = plan_store.parse_plan_steps("  [3] **Third**\n  [9] **Ninth**")
    assert [step.number for step in sparse] == [1, 2]


def test_round_trips_through_the_file() -> None:
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        run = plan_store.start_run(
            goal=plan_store.parse_plan_goal(PLAN),
            steps=plan_store.parse_plan_steps(PLAN),
            kind="complex",
            constraints=["Only clinical phase 3 or later"],
            **scope,
        )
        assert run.run_id == 1

        plan_store.update_step(step_number=1, status="completed", note="412 drugs, 88 proteins", **scope)
        plan_store.update_step(step_number=2, status="in_progress", **scope)
        plan_store.update_step(step_number=3, status="Blocked", note="No SMILES resolved", **scope)

        reloaded = plan_store.load_document(**scope).active
        assert [step.status for step in reloaded.steps] == [
            "completed",
            "in_progress",
            "blocked",
            "pending",
        ]
        assert [step.agent for step in reloaded.steps] == [
            "research_agent",
            "research_agent",
            "prediction_agent",
            "data_agent",
        ]
        assert reloaded.steps[0].note == "412 drugs, 88 proteins"
        assert reloaded.steps[0].details.startswith("Resolve the MONDO id")
        assert reloaded.constraints == ["Only clinical phase 3 or later"]
        assert reloaded.goal.startswith("Identify repurposing candidates")
        assert reloaded.progress() == (2, 4)


def test_rejects_an_invalid_update_with_a_usable_message() -> None:
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        plan_store.start_run(
            goal="g", steps=plan_store.parse_plan_steps(PLAN), kind="complex", **scope
        )

        ok, message, _ = plan_store.update_step(step_number=9, status="completed", **scope)
        assert not ok
        # The message must name the valid steps, so the model can correct itself.
        assert "Available steps: 1, 2, 3, 4" in message

        ok, message, _ = plan_store.update_step(step_number=1, status="finished", **scope)
        assert not ok and "Unknown status" in message and "completed" in message

        # Nothing was written by either failure.
        assert all(step.status == "pending" for step in plan_store.load_document(**scope).active.steps)


def test_missing_plan_is_reported_not_crashed() -> None:
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "none"}
        assert plan_store.load_document(**scope).active is None
        ok, message, run = plan_store.update_step(step_number=1, status="completed", **scope)
        assert not ok and run is None and "No execution plan" in message


def test_each_run_is_appended_so_the_file_is_a_work_log() -> None:
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        first = plan_store.start_run(
            goal="Initial plan", steps=plan_store.parse_plan_steps(PLAN), kind="complex", **scope
        )
        plan_store.update_step(step_number=1, status="completed", **scope)
        plan_store.set_outcome(
            outcome=plan_store.summarize_outcome(plan_store.load_document(**scope).active),
            run_id=first.run_id,
            **scope,
        )

        second = plan_store.start_run(
            goal="Follow-up: re-rank by hERG",
            steps=[plan_store.PlanStep(number=1, title="Re-rank")],
            kind="follow_up",
            **scope,
        )
        assert second.run_id == 2

        document = plan_store.load_document(**scope)
        assert len(document.runs) == 2
        # The earlier run keeps its own state and outcome.
        assert document.runs[0].progress() == (1, 4)
        assert document.runs[0].outcome.startswith("1/4 steps completed")
        assert document.runs[0].kind == "complex"
        assert document.active.kind == "follow_up"

        # An update with no run id targets the newest run only.
        plan_store.update_step(step_number=1, status="completed", **scope)
        document = plan_store.load_document(**scope)
        assert document.runs[0].progress() == (1, 4)
        assert document.runs[1].progress() == (1, 1)


def test_progress_line_names_the_next_step() -> None:
    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        run = plan_store.start_run(
            goal="g", steps=plan_store.parse_plan_steps(PLAN), kind="complex", **scope
        )
        line = plan_store.progress_line(run)
        assert "0/4 steps resolved" in line and "next: step 1" in line

        plan_store.update_step(step_number=1, status="completed", **scope)
        _, _, run = plan_store.update_step(step_number=2, status="skipped", **scope)
        line = plan_store.progress_line(run)
        assert "2/4 steps resolved" in line and "next: step 3" in line

        for number in (3, 4):
            _, _, run = plan_store.update_step(step_number=number, status="completed", **scope)
        # Step 2 was skipped, so this is not "all completed".
        assert plan_store.summarize_outcome(run) == "3/4 steps completed; skipped: 2."
        assert "next" not in plan_store.progress_line(run)


def test_the_written_document_is_human_readable() -> None:
    '''The file is also read by people, so its shape is part of the contract.'''

    from backend.utils import plan_store

    with scoped_results_root():
        scope = {"conversation_id": "conv"}
        plan_store.start_run(
            goal="Find AML candidates",
            steps=plan_store.parse_plan_steps(PLAN),
            kind="complex",
            constraints=["Phase 3 or later"],
            **scope,
        )
        plan_store.update_step(step_number=1, status="completed", note="412 drugs", **scope)
        text = Path(plan_store.plan_file_path(**scope)).read_text()

        assert text.startswith("# Execution plan")
        assert "## Run 1 · complex" in text
        assert "**Goal:** Find AML candidates" in text
        assert "**Approval conditions:**" in text and "- Phase 3 or later" in text
        assert "- [x] **1.** Build the AML knowledge graph · @research_agent · `completed`" in text
        assert "      Note: 412 drugs" in text
        assert "_Progress: 1/4 steps resolved._" in text


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
