'''`plan.md` — progress as a file on disk rather than prose the model reproduces.

One document per conversation, in that conversation's output scope, with each run
appended as a `## Run N` section. So it is three things at once: the contract the
supervisor executes, the live progress the UI panel renders, and a durable work
log a later follow-up can read.

Everything structural lives here in code: parsing the planner's breakdown into
steps, validating a status against the file, timestamping, atomic rewrite, and
the progress arithmetic. The supervisor supplies only *which step* reached *which
status* — which is the whole point. It replaced a prompt-only `📋 BREAKDOWN /
⏳ CURRENT / ✓ COMPLETED` block that the model had to re-emit from memory before
every delegation, and therefore drifted, repeated and contradicted itself with
nothing able to check it.

No LLM is involved in this module.
'''

from __future__ import annotations

import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from backend.utils.output_paths import conversation_output_root

PLAN_FILENAME = "plan.md"
DOCUMENT_TITLE = "# Execution plan"

PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
BLOCKED = "blocked"
SKIPPED = "skipped"

STATUSES: Tuple[str, ...] = (PENDING, IN_PROGRESS, COMPLETED, BLOCKED, SKIPPED)
TERMINAL_STATUSES: Tuple[str, ...] = (COMPLETED, BLOCKED, SKIPPED)

_MARKERS: Dict[str, str] = {
    PENDING: " ",
    IN_PROGRESS: "~",
    COMPLETED: "x",
    BLOCKED: "!",
    SKIPPED: "-",
}
_MARKER_TO_STATUS: Dict[str, str] = {marker: status for status, marker in _MARKERS.items()}

# One lock per plan file, so read-modify-write stays atomic within the process.
# Runs are serialized per conversation anyway, but `plan_update` is a sync tool
# and LangChain runs those in a thread pool.
_locks: Dict[str, threading.RLock] = {}
_locks_guard = threading.Lock()

# --- Reading back the document we wrote ---------------------------------------

_RUN_HEADING = re.compile(r"^##\s+Run\s+(\d+)\s*(?:·\s*(\S+))?\s*(?:·\s*(.*))?$")
_STEP_LINE = re.compile(
    r"^-\s+\[(.)\]\s+\*\*(\d+)\.\*\*\s+(.*?)"          # marker, number, title
    r"(?:\s+·\s+@(\S+))?"                                # optional agent
    r"(?:\s+·\s+`(\w+)`)?"                               # optional status
    r"(?:\s+·\s+(.*))?$"                                 # optional timestamp
)
_CONTINUATION = re.compile(r"^\s{4,}(Details|Depends on|Note):\s*(.*)$")
_GOAL_LINE = re.compile(r"^\*\*Goal:\*\*\s*(.*)$")
_OUTCOME_LINE = re.compile(r"^_Outcome:\s*(.*?)_?$")

# --- Reading the planning agent's breakdown -----------------------------------

# Canonical: `  [1] **Title**`
_PLAN_STEP = re.compile(r"^\s*\[(\d+)\]\s*\*\*(.+?)\*\*\s*$")
_PLAN_AGENT = re.compile(r"^\s*\*\*Agent:\*\*\s*(.*)$", re.I)
_PLAN_DETAIL = re.compile(r"^\s*\*\*Details:\*\*\s*(.*)$", re.I)
_PLAN_DEPENDS = re.compile(r"^\s*\*\*Depends on:\*\*\s*(.*)$", re.I)
# Fallbacks: `1. Title` / `1) Title`
_NUMBERED = re.compile(r"^\s*(\d+)[.)]\s+(.{3,})$")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _clip(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


@dataclass
class PlanStep:
    number: int
    title: str
    agent: str = ""
    details: str = ""
    depends_on: str = ""
    status: str = PENDING
    note: str = ""
    updated_at: str = ""

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES


@dataclass
class PlanRun:
    run_id: int
    kind: str = "complex"
    started_at: str = ""
    goal: str = ""
    constraints: List[str] = field(default_factory=list)
    steps: List[PlanStep] = field(default_factory=list)
    outcome: str = ""

    def step(self, number: int) -> Optional[PlanStep]:
        return next((item for item in self.steps if item.number == number), None)

    def progress(self) -> Tuple[int, int]:
        return sum(1 for item in self.steps if item.is_terminal), len(self.steps)

    @property
    def unresolved(self) -> List[PlanStep]:
        return [item for item in self.steps if not item.is_terminal]


@dataclass
class PlanDocument:
    runs: List[PlanRun] = field(default_factory=list)
    preamble: str = ""

    def run(self, run_id: Optional[int]) -> Optional[PlanRun]:
        if run_id is None:
            return None
        return next((item for item in self.runs if item.run_id == run_id), None)

    @property
    def active(self) -> Optional[PlanRun]:
        return self.runs[-1] if self.runs else None


# --------------------------------------------------------------------------
# Parsing the planning agent's output into steps
# --------------------------------------------------------------------------


def parse_plan_steps(plan_text: str) -> List[PlanStep]:
    '''Turn an approved plan into structured steps.

    Deliberately tolerant, in three tiers: the canonical `[1] **Title**`
    breakdown, then a plain numbered list, then a single catch-all step holding
    the whole plan. Execution must never be blocked because the planner phrased
    its output differently — a mis-parsed plan should degrade the ledger, not the
    science.

    Parameters:
    ---------
    plan_text (str): the approved plan as the planning agent wrote it.

    Returns:
    ----------
    steps (list): the parsed `PlanStep`s, degrading to a single catch-all step rather than failing.
    '''

    lines = (plan_text or "").splitlines()

    steps: List[PlanStep] = []
    current: Optional[PlanStep] = None
    for line in lines:
        match = _PLAN_STEP.match(line)
        if match:
            current = PlanStep(number=int(match.group(1)), title=match.group(2).strip())
            steps.append(current)
            continue
        if current is None:
            continue
        agent = _PLAN_AGENT.match(line)
        if agent:
            current.agent = _normalize_agent(agent.group(1))
            continue
        detail = _PLAN_DETAIL.match(line)
        if detail:
            current.details = detail.group(1).strip()
            continue
        depends = _PLAN_DEPENDS.match(line)
        if depends:
            current.depends_on = depends.group(1).strip()

    if not steps:
        for line in lines:
            match = _NUMBERED.match(line)
            if match:
                title = match.group(2).strip().strip("*").strip()
                if title:
                    steps.append(PlanStep(number=int(match.group(1)), title=title))

    if not steps:
        summary = next((line.strip() for line in lines if line.strip()), "Execute the approved plan")
        steps = [PlanStep(number=1, title=_clip(summary, 120), details=_clip(plan_text or "", 600))]

    # Renumber densely so step ids are always 1..N and therefore addressable.
    for index, step in enumerate(steps, start=1):
        step.number = index
    return steps


def _normalize_agent(value: str) -> str:
    '''Map whatever the planner wrote to a specialist name the UI can label.

    Parameters:
    ---------
    value (str): whatever the planner named as the agent for a step.

    Returns:
    ----------
    agent (str): a specialist name the UI can label, or an empty string when none matches.
    '''

    text = " ".join((value or "").split()).strip().strip("`*").lower()
    if not text or text in {"none", "-", "n/a"}:
        return ""
    for key, canonical in (
        ("research", "research_agent"),
        ("knowledge", "research_agent"),
        ("literature", "research_agent"),
        ("predict", "prediction_agent"),
        ("admet", "prediction_agent"),
        ("data", "data_agent"),
        ("analysis", "data_agent"),
        ("report", "report_agent"),
    ):
        if key in text:
            return canonical
    return _clip(text.replace(" ", "_"), 40)


def parse_plan_goal(plan_text: str) -> str:
    '''The one-line goal from `📋 **PLAN:** ...`, else the first useful line.

    Parameters:
    ---------
    plan_text (str): the approved plan as the planning agent wrote it.

    Returns:
    ----------
    goal (str): the conversation goal, or an empty string when the plan states none.
    '''

    lines = (plan_text or "").splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        marker = re.match(r"^.*?\*\*PLAN:\*\*\s*(.+)$", stripped)
        if marker:
            return _clip(marker.group(1).strip(), 200)
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith(("#", "-", "*", "[", "_")):
            return _clip(stripped, 200)
    return ""


# --------------------------------------------------------------------------
# Rendering and re-reading the document
# --------------------------------------------------------------------------


def render_step(step: PlanStep) -> List[str]:
    marker = _MARKERS.get(step.status, " ")
    head = f"- [{marker}] **{step.number}.** {step.title}"
    if step.agent:
        head += f" · @{step.agent}"
    head += f" · `{step.status}`"
    if step.updated_at:
        head += f" · {step.updated_at}"
    lines = [head]
    if step.details:
        lines.append(f"      Details: {step.details}")
    if step.depends_on:
        lines.append(f"      Depends on: {step.depends_on}")
    if step.note:
        lines.append(f"      Note: {step.note}")
    return lines


def render_run(run: PlanRun) -> str:
    done, total = run.progress()
    lines = [f"## Run {run.run_id} · {run.kind} · {run.started_at}"]
    if run.goal:
        lines.append(f"**Goal:** {run.goal}")
    if run.constraints:
        lines.append("")
        lines.append("**Approval conditions:**")
        lines.extend(f"- {item}" for item in run.constraints)
    lines.append("")
    for step in run.steps:
        lines.extend(render_step(step))
    lines.append("")
    lines.append(f"_Progress: {done}/{total} steps resolved._")
    if run.outcome:
        lines.append(f"_Outcome: {run.outcome}_")
    return "\n".join(lines)


def render_document(document: PlanDocument) -> str:
    parts = [DOCUMENT_TITLE, ""]
    if document.preamble.strip():
        parts.extend([document.preamble.strip(), ""])
    for run in document.runs:
        parts.append(render_run(run))
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def parse_document(text: str) -> PlanDocument:
    document = PlanDocument()
    run: Optional[PlanRun] = None
    step: Optional[PlanStep] = None

    for line in (text or "").splitlines():
        heading = _RUN_HEADING.match(line.strip())
        if heading:
            run = PlanRun(
                run_id=int(heading.group(1)),
                kind=(heading.group(2) or "complex").strip(),
                started_at=(heading.group(3) or "").strip(),
            )
            document.runs.append(run)
            step = None
            continue
        if run is None:
            continue

        goal = _GOAL_LINE.match(line.strip())
        if goal:
            run.goal = goal.group(1).strip()
            continue

        step_match = _STEP_LINE.match(line.rstrip())
        if step_match:
            marker, number, title, agent, status, stamp = step_match.groups()
            resolved = status if status in STATUSES else _MARKER_TO_STATUS.get(marker, PENDING)
            step = PlanStep(
                number=int(number),
                title=(title or "").strip(),
                agent=(agent or "").strip(),
                status=resolved,
                updated_at=(stamp or "").strip(),
            )
            run.steps.append(step)
            continue

        continuation = _CONTINUATION.match(line)
        if continuation and step is not None:
            key, value = continuation.group(1), continuation.group(2).strip()
            if key == "Details":
                step.details = value
            elif key == "Depends on":
                step.depends_on = value
            else:
                step.note = value
            continue

        stripped = line.strip()
        if stripped.startswith("_Outcome:"):
            outcome = _OUTCOME_LINE.match(stripped)
            if outcome:
                run.outcome = outcome.group(1).strip()
        elif stripped.startswith("- ") and step is None and not run.steps:
            # Approval conditions are the only bare bullets before the steps.
            run.constraints.append(stripped[2:].strip())

    return document


# --------------------------------------------------------------------------
# File access
# --------------------------------------------------------------------------


def plan_file_path(
    *,
    conversation_id: Optional[str] = None,
) -> Path:
    return conversation_output_root(conversation_id) / PLAN_FILENAME


def _lock_for(path: Path) -> threading.RLock:
    key = str(path)
    with _locks_guard:
        lock = _locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _locks[key] = lock
        return lock


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(text, encoding="utf-8")
    os.replace(temp, path)


def load_document(
    *,
    conversation_id: Optional[str] = None,
) -> PlanDocument:
    path = plan_file_path(conversation_id=conversation_id)
    if not path.exists():
        return PlanDocument()
    try:
        return parse_document(path.read_text(encoding="utf-8"))
    except OSError:
        return PlanDocument()


def start_run(
    *,
    goal: str,
    steps: List[PlanStep],
    kind: str = "complex",
    constraints: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
) -> PlanRun:
    '''Append a new run section and return it.

    Parameters:
    ---------
    goal (str): the conversation goal, clipped to 200 characters.
    steps (list): the parsed steps this run will execute.
    kind (str): the task category that produced the plan, `complex`, `simple` or `follow_up`.
    constraints (list): conditions the user attached when approving, which override the steps they touch.
    conversation_id (str): the conversation whose `plan.md` to append to, defaulting to the ambient scope.

    Returns:
    ----------
    run (PlanRun): the run just appended, numbered one past the last.
    '''

    path = plan_file_path(conversation_id=conversation_id)
    with _lock_for(path):
        document = load_document(conversation_id=conversation_id)
        run = PlanRun(
            run_id=(document.runs[-1].run_id + 1) if document.runs else 1,
            kind=kind,
            started_at=_now(),
            goal=_clip(goal, 200),
            constraints=[item for item in (constraints or []) if str(item).strip()],
            steps=steps,
        )
        document.runs.append(run)
        _write_atomic(path, render_document(document))
        return run


def update_step(
    *,
    step_number: int,
    status: str,
    note: str = "",
    run_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> Tuple[bool, str, Optional[PlanRun]]:
    '''Set one step's status. Returns `(ok, message, updated run)`.

    Validates the status and the step number against the file rather than
    trusting the caller, so a hallucinated step number is an error the model can
    see and correct instead of a silent no-op.

    Parameters:
    ---------
    step_number (int): step to record, as shown by `plan_status`.
    status (str): one of `pending`, `in_progress`, `completed`, `blocked`, `skipped`.
    note (str): one line worth carrying forward, clipped to 300 characters.
    run_id (int): run to update, defaulting to the active one.
    conversation_id (str): the conversation whose `plan.md` to rewrite, defaulting to the ambient scope.

    Returns:
    ----------
    result (tuple): `(ok, message, run)` — the message names the valid steps or statuses when `ok` is False.
    '''

    normalized = (status or "").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in STATUSES:
        return False, f"Unknown status {status!r}. Use one of: {', '.join(STATUSES)}.", None

    path = plan_file_path(conversation_id=conversation_id)
    with _lock_for(path):
        document = load_document(conversation_id=conversation_id)
        run = document.run(run_id) if run_id is not None else document.active
        if run is None:
            return False, "No execution plan exists for this conversation yet.", None
        step = run.step(step_number)
        if step is None:
            available = ", ".join(str(item.number) for item in run.steps) or "none"
            return (
                False,
                f"Step {step_number} is not in run {run.run_id}. Available steps: {available}.",
                run,
            )

        step.status = normalized
        step.updated_at = _now()
        if note:
            step.note = _clip(note, 300)
        _write_atomic(path, render_document(document))
        return True, f"Step {step_number} set to {normalized}.", run


def set_outcome(
    *,
    outcome: str,
    run_id: Optional[int] = None,
    conversation_id: Optional[str] = None,
) -> Optional[PlanRun]:
    path = plan_file_path(conversation_id=conversation_id)
    with _lock_for(path):
        document = load_document(conversation_id=conversation_id)
        run = document.run(run_id) if run_id is not None else document.active
        if run is None:
            return None
        run.outcome = _clip(outcome, 300)
        _write_atomic(path, render_document(document))
        return run


def progress_line(run: Optional[PlanRun]) -> str:
    '''One line for the pinned context: how far the run has got and what is next.

    Parameters:
    ---------
    run (PlanRun): the run to describe, or None.

    Returns:
    ----------
    line (str): how many steps are resolved and which is next, or an empty string when there is no plan.
    '''

    if run is None or not run.steps:
        return ""
    done, total = run.progress()
    unresolved = run.unresolved
    nxt = f" · next: step {unresolved[0].number} ({unresolved[0].title})" if unresolved else ""
    return f"Run {run.run_id}: {done}/{total} steps resolved{nxt}"


def summarize_outcome(run: PlanRun) -> str:
    '''What actually happened, phrased for the run's `_Outcome:` line.

    `blocked` and `skipped` are terminal, so a run can be fully "resolved" without
    having done what it set out to do. Saying "All N steps resolved" in that case
    reads as success, so each non-completed step is named instead.

    Parameters:
    ---------
    run (PlanRun): the finished run to describe.

    Returns:
    ----------
    outcome (str): a sentence naming each step that did not complete, so a blocked run is not read as a success.
    '''

    total = len(run.steps)
    completed = [step for step in run.steps if step.status == COMPLETED]
    blocked = [step for step in run.steps if step.status == BLOCKED]
    skipped = [step for step in run.steps if step.status == SKIPPED]
    unresolved = run.unresolved

    if len(completed) == total and total:
        return f"All {total} steps completed."

    parts = [f"{len(completed)}/{total} steps completed"]
    if blocked:
        parts.append("blocked: " + ", ".join(str(step.number) for step in blocked))
    if skipped:
        parts.append("skipped: " + ", ".join(str(step.number) for step in skipped))
    if unresolved:
        listed = ", ".join(str(step.number) for step in unresolved[:8])
        parts.append(f"unresolved: {listed}" + ("…" if len(unresolved) > 8 else ""))
    return "; ".join(parts) + "."
