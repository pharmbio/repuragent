'''Reading and advancing the on-disk execution plan.

These two tools are what replaced the supervisor's prose tracking block. The
supervisor supplies only a step number and a status; the file mutation,
validation, timestamping and the ledger rendered back all happen in code, so
reported progress cannot drift from what actually happened.

The ledger returned by `plan_update` **is** the progress display. Do not add
narration alongside it.
'''

from __future__ import annotations

from typing import Annotated, Any, Dict, Optional

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from backend.utils import plan_store
from backend.utils.output_paths import DEFAULT_CONVERSATION, get_current_conversation_id

_STEP_MARKERS = {
    plan_store.COMPLETED: "[x]",
    plan_store.BLOCKED: "[!]",
    plan_store.IN_PROGRESS: "[~]",
    plan_store.SKIPPED: "[-]",
}


def _scope(state: Optional[Dict[str, Any]] = None) -> dict:
    '''Which conversation's plan file to touch.

    Read from **graph state first**, because that is what `plan_init` used when it
    wrote the file. Tools elsewhere resolve their scope from contextvars, and when
    the two disagree the plan is written in one place and looked for in another —
    which surfaces as `plan_update` reporting "No execution plan exists for this
    conversation yet" while `plan.md` sits there, correctly written. Taking the
    scope from the same source as the writer removes that failure mode entirely;
    contextvars remain as the fallback for a caller outside the graph.

    Parameters:
    ---------
    state (dict): graph state for this run, read before contextvars so the plan is looked for where `plan_init` wrote it.

    Returns:
    ----------
    scope (dict): the `conversation_id` this conversation's plan file is keyed by.
    '''

    state = state or {}
    return {
        "conversation_id": (
            state.get("conversation_id") or get_current_conversation_id() or DEFAULT_CONVERSATION
        ),
    }


def render_ledger(run: Optional[plan_store.PlanRun], *, path: str = "") -> str:
    '''Compact, code-generated view of a run: the deterministic tracking block.

    Parameters:
    ---------
    run (PlanRun): the run to render, or None when no plan has been created yet.
    path (str): the plan file's location, appended so the reader can open it.

    Returns:
    ----------
    ledger (str): the compact progress block, one line per step.
    '''

    if run is None or not run.steps:
        return "No execution plan has been created for this conversation yet."

    done, total = run.progress()
    lines = [f"PLAN · run {run.run_id} · {done}/{total} steps resolved"]
    if run.goal:
        lines.append(f"Goal: {run.goal}")
    if run.constraints:
        lines.append("Approval conditions (these override the corresponding steps):")
        lines.extend(f"  - {item}" for item in run.constraints)
    lines.append("")

    for step in run.steps:
        marker = _STEP_MARKERS.get(step.status, "[ ]")
        suffix = f" · {step.agent}" if step.agent else ""
        lines.append(f"{marker} {step.number}. {step.title}{suffix}")
        if step.details:
            lines.append(f"       {step.details}")
        if step.depends_on and step.depends_on.lower() not in {"none", "-", ""}:
            lines.append(f"       depends on: {step.depends_on}")
        if step.note:
            lines.append(f"       note: {step.note}")

    unresolved = run.unresolved
    lines.append("")
    lines.append(
        f"Next unresolved: step {unresolved[0].number} — {unresolved[0].title}"
        if unresolved
        else "All steps resolved."
    )
    if path:
        lines.append(f"File: {path}")
    return "\n".join(lines)


@tool
def plan_status(state: Annotated[Dict[str, Any], InjectedState] = None) -> str:
    '''Show this conversation's execution plan and how far it has got.

    Returns every step of the active run with its status, which specialist it is
    assigned to, and which step is next. This is the authoritative copy — the plan
    lives on disk, not in the conversation — so read it whenever you are unsure
    what remains, rather than reconstructing it from the transcript.

    Parameters:
    ---------
    state (dict): graph state, injected by LangGraph, used only to resolve the scope.

    Returns:
    ----------
    ledger (str): every step of the active run with its status and the specialist it is assigned to.
    '''

    scope = _scope(state)
    document = plan_store.load_document(**scope)
    return render_ledger(document.active, path=str(plan_store.plan_file_path(**scope)))


@tool
def plan_update(
    step: int,
    status: str,
    note: str = "",
    state: Annotated[Dict[str, Any], InjectedState] = None,
) -> str:
    '''Record the outcome of one plan step, then show the updated plan.

    Call this once per step, when that step's outcome is actually established by a
    specialist's returned evidence — not when you start it, and not to restate
    progress. The tool validates the step number and status against the file, so a
    step number that does not exist comes back as an error naming the valid ones.

    Parameters:
    ---------
    step (int): Step number in the active run, as shown by `plan_status`.
    status (str): One of `in_progress`, `completed`, `blocked`, `skipped`, `pending`.
    note (str): One line worth carrying forward — a key value, an output path, or why the step was blocked.

    Returns:
    ----------
    ledger (str): The refreshed plan, or an error naming the valid steps or statuses.
    '''

    scope = _scope(state)
    ok, message, run = plan_store.update_step(
        step_number=step, status=status, note=note, **scope
    )
    if not ok:
        return f"Plan not updated: {message}"
    return f"{message}\n\n" + render_ledger(
        run, path=str(plan_store.plan_file_path(**scope))
    )


__all__ = ["plan_status", "plan_update", "render_ledger"]
