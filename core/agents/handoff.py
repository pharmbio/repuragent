'''Delegation: how the supervisor briefs a specialist.

A handoff is a tool the supervisor calls. It returns a `Command` that jumps to the
specialist's node in the **execution subgraph** (`graph=Command.PARENT`, relative to the
supervisor's own agent graph).

**The specialist sees only what the brief carries** — no conversation, no plan, no other
specialist's work (`core/agents/context.py::build_specialist_input_messages`). So the
brief is the entire input, and the tool's shape is what makes it complete:

* `objective` — the outcome, not the procedure.
* `inputs` — the values to start from.
* `artifacts` — files from earlier steps, **each with what it contains**. The specialist
  does receive a raw list of paths in its context, but that list is a directory listing:
  ten files with byte counts and nothing saying which is the candidate table, which is
  the pathway extract, or how they join. Naming the relevant ones and what they hold is
  the difference between a specialist working and a specialist guessing.
* `constraints` — the non-negotiables of this step.
* `expected_output` — what to hand back.
* `context` — why, and what earlier steps established.

The level to pitch it at is a real design constraint in both directions. Prescribe the
method and the specialist becomes a typist: its domain knowledge goes unused and a
mistake in the supervisor's method goes uncorrected. Leave it too thin and the specialist
guesses, which is where wrong results come from. `validate_brief` catches the mechanical
half of that — dangling references, undescribed files, paths that do not exist — and the
supervisor prompt's calibration section teaches the rest.

The rendered brief is the handoff `ToolMessage`'s content, so it needs no extra state
channel, survives two handoffs in one step, and is visible in the transcript for free.

Why the supervisor node has no outgoing edge: a static edge from `supervisor` would fire
*in parallel* with the handoff, running the next node before the specialist had done
anything. The subgraph instead ends when the supervisor stops delegating.

The same `Command` has a second consequence that `handoff_messages` exists for:
`graph=Command.PARENT` abandons the supervisor's own agent graph, so the parent gets the
`update` and nothing else — the `AIMessage` carrying the `transfer_to_<agent>` `tool_call`
included. Propagating the brief alone leaves the parent holding a `tool` message that
answers a call it cannot see, and every later request is rejected outright.
'''

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command

BRIEF_HEADING = "TASK BRIEF"

# Each specialist, as the supervisor sees it. These are the supervisor's only basis for
# choosing, so they say what the agent can and cannot do.
SPECIALISTS: dict[str, str] = {
    "research_agent": (
        "Delegate literature and knowledge-graph work: resolve a disease identifier, "
        "build or read a disease knowledge graph, extract drug / protein / pathway / "
        "mechanism candidates from it, look up drugs for a target, pathway or mechanism, "
        "search the literature, retrieve an SOP, or annotate compounds against ChEMBL and "
        "PubChem. Cannot run ADMET models, write code, or produce figures."
    ),
    "prediction_agent": (
        "Delegate molecular property prediction: the CPSign ADMET panel (CYP3A4, CYP2C19, "
        "CYP2D6, CYP1A2, CYP2C9, hERG, Ames, P-gp, PAMPA, BBB, solubility, lipophilicity) "
        "and new-indication prediction for a set of drugs. Needs SMILES or a CSV "
        "containing them. Cannot search literature, traverse a knowledge graph, or "
        "analyse results."
    ),
    "data_agent": (
        "Delegate anything that needs code: inspecting an uploaded or generated file, "
        "merging and cleaning tables, scoring and ranking candidates, statistics, and "
        "producing figures. Runs Python with the conversation's files in scope. Cannot "
        "search literature or run the ADMET models."
    ),
}

# --- Validation ---------------------------------------------------------------

# Phrases that only mean something to someone who watched the earlier steps. In
# `inputs` or `artifacts` — the two fields whose whole job is to carry concrete values —
# they are unresolvable, and they are the single most likely way an isolated delegation
# fails. Checked only in those fields: "the previous step established X" is legitimate
# prose in `context`.
_DANGLING_REFERENCES = re.compile(
    r"\b("
    r"the (previous|prior|last|earlier|preceding) (step|run|result|output|analysis|one)"
    r"|(from|of) (the )?(previous|prior|last|earlier) (step|run)"
    # "the shortlist that we produced", "the csv you found earlier", "the graph built before"
    r"|the (file|files|csv|table|list|results?|candidates?|shortlist|graph|output)\b"
    r"[^.\n]{0,24}?\b(previous|prior|earlier|before|found|produced|created|built|extracted|generated)"
    r"|as (before|above|discussed|previously)"
    r"|same as (before|above)"
    r"|you (found|produced|created|built|extracted|generated) (it|them|earlier|before)"
    r")\b",
    re.IGNORECASE,
)

# An artifact entry has to say what the file holds, not just name it.
_DESCRIPTION_SEPARATORS = ("—", "–", " - ", ": ", " | ")

_MIN_OBJECTIVE_CHARS = 15
_MIN_EXPECTED_OUTPUT_CHARS = 8


def _artifact_path(entry: str) -> str:
    '''The path token at the head of an artifact entry.

    Parameters:
    ---------
    entry (str): one artifact entry, `<full path> — <what it contains>`.

    Returns:
    ----------
    path (str): the path token at the head of the entry, stripped of quotes and backticks.
    '''

    head = entry.strip()
    for separator in _DESCRIPTION_SEPARATORS:
        if separator in head:
            head = head.split(separator, 1)[0]
            break
    else:
        head = head.split()[0] if head.split() else ""
    return head.strip().strip("`'\"")


def validate_brief(
    *,
    agent_name: str,
    objective: str,
    inputs: str,
    expected_output: str,
    artifacts: Sequence[str] = (),
    constraints: Sequence[str] = (),
) -> Optional[str]:
    '''The reason this brief cannot be delivered, or None.

    Returned to the supervisor as the tool result so it can rewrite and retry — the same
    mechanism `plan_update` uses for an invalid step number. Every check here is for a
    defect the specialist could not recover from on its own.

    Parameters:
    ---------
    agent_name (str): the specialist this brief is addressed to, named in the refusal so the supervisor knows who cannot see what.
    objective (str): what must be true when the step is done.
    inputs (str): the values needed to start, checked for references that resolve to nothing in isolation.
    expected_output (str): what the specialist should hand back.
    artifacts (list): files from earlier steps, each of which must say what it holds and must exist on disk.
    constraints (list): the non-negotiables of this step.

    Returns:
    ----------
    problem (str): why the brief cannot be delivered, or None when it is complete.
    '''

    problems: List[str] = []

    if len(" ".join(objective.split())) < _MIN_OBJECTIVE_CHARS:
        problems.append(
            "`objective` is too short to act on. State what must be true when the step is "
            "done, e.g. \"Build the AML disease knowledge graph and report the drug and "
            "protein counts\"."
        )
    if len(" ".join(expected_output.split())) < _MIN_EXPECTED_OUTPUT_CHARS:
        problems.append(
            "`expected_output` is missing. Say what to hand back — which values, counts "
            "or file paths, and in what form — or the result will not be usable by the "
            "next step."
        )

    checked = [("inputs", inputs)]
    # Only the path head of an artifact entry is checked. Its description is prose, and
    # "…/drugs.csv — the list extracted from the graph" is a good description of a file
    # whose path is right there.
    checked += [("artifacts", _artifact_path(entry)) for entry in artifacts if entry]
    for field_name, value in checked:
        match = _DANGLING_REFERENCES.search(value or "")
        if match:
            problems.append(
                f"`{field_name}` refers to {match.group(0)!r}. {agent_name} cannot see any "
                "earlier step, so that resolves to nothing — substitute the actual value, "
                "identifier or full file path."
            )

    for entry in artifacts:
        text = (entry or "").strip()
        if not text:
            continue
        path = _artifact_path(text)
        if not any(separator in text for separator in _DESCRIPTION_SEPARATORS):
            problems.append(
                f"artifact {text!r} says only where the file is, not what it holds. Write "
                "it as `<path> — <what it contains>`, e.g. "
                "\"/…/master_candidates.csv — 412 candidate drugs with ChEMBL ids and SMILES\"."
            )
            continue
        if not path:
            problems.append(f"artifact {text!r} has no path in front of its description.")
            continue
        # A hallucinated path costs a whole delegation to discover, and it is free to
        # check here. Relative paths are left alone: the specialist resolves those
        # against its own scope.
        if path.startswith("/") and not Path(path).exists():
            problems.append(
                f"artifact path {path!r} does not exist. Name a file an earlier step "
                "actually produced — the artifact list in your context shows what is there."
            )

    if not problems:
        return None
    return (
        f"Brief not delivered to {agent_name}. Fix and call again:\n"
        + "\n".join(f"- {problem}" for problem in problems)
    )


# --- Rendering ----------------------------------------------------------------


def render_brief(
    *,
    agent_name: str,
    objective: str,
    inputs: str,
    expected_output: str,
    artifacts: Sequence[str] = (),
    constraints: Sequence[str] = (),
    context: str = "",
) -> str:
    '''The brief as the specialist will read it.

    Parameters:
    ---------
    agent_name (str): the specialist the brief is addressed to.
    objective (str): what must be true when the step is done.
    inputs (str): the values needed to start.
    expected_output (str): what to hand back.
    artifacts (list): files from earlier steps, each with what it contains.
    constraints (list): the non-negotiables of this step.
    context (str): background the specialist cannot see and would otherwise guess.

    Returns:
    ----------
    brief (str): the handoff `ToolMessage` content, which is the specialist's entire input.
    '''

    lines = [f"{BRIEF_HEADING} for {agent_name}", "", f"Objective: {objective.strip()}"]

    if inputs and inputs.strip():
        lines.append(f"Inputs: {inputs.strip()}")

    kept_artifacts = [item.strip() for item in artifacts if item and item.strip()]
    if kept_artifacts:
        lines.append("Files produced by earlier steps, and what they hold:")
        lines.extend(f"  - {item}" for item in kept_artifacts)

    kept_constraints = [item.strip() for item in constraints if item and item.strip()]
    if kept_constraints:
        lines.append("Constraints for this step:")
        lines.extend(f"  - {item}" for item in kept_constraints)

    lines.append(f"Expected output: {expected_output.strip()}")

    if context and context.strip():
        lines.append(f"Background: {context.strip()}")
    return "\n".join(lines)


# --- Carrying the delegation up to the parent graph ---------------------------


def handoff_messages(
    messages: Sequence[BaseMessage],
    tool_message: ToolMessage,
    tool_call_id: str,
) -> List[BaseMessage]:
    '''The parent's message list, with the delegating `AIMessage` carried up beside the brief.

    **This is what stops the run 400-ing on the supervisor's next model call.** The
    handoff returns `Command(graph=Command.PARENT)`, which abandons the supervisor's
    own agent graph mid-run: the parent receives the `Command`'s `update` and nothing
    else. So an update of only the brief leaves the parent holding a `ToolMessage`
    whose `AIMessage` — the one carrying the `transfer_to_<agent>` `tool_call` — was
    never written anywhere the parent can see. Every later model call then ships a
    `tool` message answering a call that is not in the list, and OpenAI rejects the
    request outright:

        Invalid parameter: messages with role 'tool' must be a response to a
        preceeding message with 'tool_calls'.

    The whole inner list is returned rather than just the tail. `add_messages` keys on
    id, so the messages the parent already holds are replaced by themselves, and only
    what the supervisor actually added this turn is appended — in order. Sending the
    tail instead would mean guessing where the parent's copy ended, and guessing short
    drops a tool pair.

    Parameters:
    ---------
    messages (list): the supervisor's inner message list, which is the parent's copy plus whatever this turn added.
    tool_message (ToolMessage): the rendered brief, answering this handoff's `tool_call`.
    tool_call_id (str): the id of the `tool_call` this handoff answers, which is the only one the propagated `AIMessage` may keep.

    Returns:
    ----------
    messages (list): the inner list plus the brief, with the delegating `AIMessage` reduced to the one `tool_call` the brief answers.
    '''

    history = list(messages or [])
    if not history:
        return [tool_message]

    last = history[-1]
    calls = list(getattr(last, "tool_calls", None) or [])
    # A parallel call — `plan_update` alongside the handoff, or two handoffs at once —
    # is the mirror-image failure: the sibling `ToolMessage`s are written into the
    # abandoned inner graph, so propagating the `AIMessage` whole would leave *their*
    # `tool_call`s unanswered and OpenAI would reject that instead. Only the call this
    # brief answers survives; the supervisor can make the others again.
    if isinstance(last, AIMessage) and len(calls) > 1:
        kept = [call for call in calls if call.get("id") == tool_call_id]
        if kept:
            update: Dict[str, Any] = {"tool_calls": kept}
            raw = last.additional_kwargs.get("tool_calls")
            if isinstance(raw, list):
                update["additional_kwargs"] = {
                    **last.additional_kwargs,
                    "tool_calls": [
                        item
                        for item in raw
                        if not isinstance(item, dict) or item.get("id") == tool_call_id
                    ],
                }
            history[-1] = last.model_copy(update=update)

    return history + [tool_message]


def superseded_handoff(
    messages: Sequence[BaseMessage], tool_call_id: str
) -> Optional[str]:
    '''Why this handoff must stand down, when the same turn asked for two of them.

    Two `transfer_to_*` calls in one `AIMessage` cannot both be delivered. Each returns
    its own `Command(graph=Command.PARENT)` carrying its own copy of that `AIMessage`,
    and both copies share its id — so `add_messages` keeps whichever lands last and the
    other specialist's brief is left answering a call that is no longer there. Two
    specialists also cannot execute at once here: they share the message list and the
    plan file, and concurrent writes to `plan_progress` are what LangGraph rejects with
    `InvalidUpdateError`.

    So the first handoff in the turn proceeds and the rest stand down. "First" is the
    order the model wrote the calls in, not the order the tool node happens to run them,
    which keeps the choice deterministic. The step the supervisor stood down on is still
    unresolved in `plan.md`, so it delegates it again on its next turn.

    Parameters:
    ---------
    messages (list): the supervisor's inner message list, whose last entry is the delegating turn.
    tool_call_id (str): the `tool_call` this handoff is answering.

    Returns:
    ----------
    reason (str): why this handoff stood down, to be returned to the supervisor as the tool result, or None when it is the one to proceed.
    '''

    history = list(messages or [])
    if not history:
        return None
    last = history[-1]
    if not isinstance(last, AIMessage):
        return None
    handoffs = [
        call
        for call in (last.tool_calls or [])
        if is_handoff_tool_name(call.get("name")) and call.get("id")
    ]
    if len(handoffs) < 2 or handoffs[0].get("id") == tool_call_id:
        return None
    winner = agent_from_handoff_tool_name(str(handoffs[0].get("name")))
    return (
        "Not delivered: you delegated to more than one specialist in a single turn, and "
        f"only the first ({winner}) can be handed the work — they share one message "
        "list and one plan file, so they cannot run at the same time. Wait for "
        f"{winner} to report, then delegate this step on your next turn."
    )


# --- The tool -----------------------------------------------------------------


def make_handoff_tool(agent_name: str, description: str) -> BaseTool:
    '''A `transfer_to_<agent>` tool that briefs and routes to that specialist.

    Parameters:
    ---------
    agent_name (str): the specialist to route to, which becomes the `transfer_to_<agent>` tool name.
    description (str): how this specialist is presented to the supervisor when it chooses.

    Returns:
    ----------
    handoff (BaseTool): the tool, returning a `Command` that jumps to the specialist's node.
    '''

    tool_name = f"transfer_to_{agent_name}"

    # `parse_docstring=True` is load-bearing: with `description=` alone LangChain
    # leaves every argument description empty, so the calibration guidance below —
    # the whole reason the brief has separate fields — never reaches the model.
    # It also fixes the section header: `_parse_google_docstring` looks for a block
    # starting with the literal `Args:` and raises on anything else, so this one
    # docstring says `Args:` where the rest of the codebase says `Parameters:`.
    # The dashed underline and the `name (type):` entries are both tolerated.
    @tool(tool_name, description=description, parse_docstring=True)
    def handoff(
        objective: str,
        inputs: str,
        expected_output: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        state: Annotated[Dict[str, Any], InjectedState],
        artifacts: Optional[List[str]] = None,
        constraints: Optional[List[str]] = None,
        context: str = "",
    ):
        '''Brief the specialist and hand the step to it.

        This agent **cannot see the conversation, the plan, or any other agent's work** —
        the brief is everything it gets. Write it so someone who just walked in could
        execute the step, and leave the method to them: state the outcome you need, not
        the tool calls to make.

        Args:
        ---------
        objective (str): What must be true when the step is done, as an instruction. "Build the disease knowledge graph for MONDO:0018874 and report the drug, protein and pathway counts" — not "do the research step" (too thin), and not "call create_knowledge_graph then extract_drugs_from_kg" (that is the specialist's decision, and it knows those tools better than you do).
        inputs (str): The values needed to start, spelled out: identifiers, SMILES, thresholds, column names, gene symbols. Not files — those go in `artifacts`. Write "none" when the step needs no starting value.
        expected_output (str): What to hand back so the next step can use it: which values, which counts, which paths, in what form.
        artifacts (list): Files from earlier steps this step should use, one per entry, as `<full path> — <what it contains>`. The specialist can see a bare list of paths but nothing about what any of them holds, so this is what tells it which file is the candidate table and which is the pathway extract. Take the paths from the artifact list in your context.
        constraints (list): The non-negotiables of this step, one per entry: which endpoint matters, a threshold to apply, a category to exclude. Conditions the user attached when approving the plan already reach every agent, so put only step-specific requirements here.
        context (str): Optional background it cannot see and would otherwise guess: why this step matters, what an earlier step established, an ambiguity to watch for.

        Returns:
        ----------
        outcome (Command or str): a `Command` routing to the specialist, or the reason the brief was refused — returned as a plain string so the supervisor can rewrite it and call again.
        '''

        artifacts = list(artifacts or [])
        constraints = list(constraints or [])

        history = (state or {}).get("messages") or []
        superseded = superseded_handoff(history, tool_call_id)
        if superseded:
            return superseded

        problem = validate_brief(
            agent_name=agent_name,
            objective=objective,
            inputs=inputs,
            expected_output=expected_output,
            artifacts=artifacts,
            constraints=constraints,
        )
        if problem:
            # Returning a plain string rather than a Command keeps the supervisor in
            # control: it gets the reason as a tool result and can rewrite the brief,
            # instead of a specialist being handed something it cannot execute.
            return problem

        brief = ToolMessage(
            content=render_brief(
                agent_name=agent_name,
                objective=objective,
                inputs=inputs,
                expected_output=expected_output,
                artifacts=artifacts,
                constraints=constraints,
                context=context,
            ),
            name=tool_name,
            tool_call_id=tool_call_id,
        )
        return Command(
            goto=agent_name,
            # The brief alone is not enough: `graph=Command.PARENT` abandons the
            # supervisor's own graph, so the `AIMessage` holding this `tool_call` has to
            # travel up with it or the parent is left with an unanswerable `tool` message.
            update={"messages": handoff_messages(history, brief, tool_call_id)},
            graph=Command.PARENT,
        )

    return handoff


def build_handoff_tools(agent_names: Optional[List[str]] = None) -> List[BaseTool]:
    names = agent_names if agent_names is not None else list(SPECIALISTS)
    return [make_handoff_tool(name, SPECIALISTS[name]) for name in names]


def handoff_tool_name(agent_name: str) -> str:
    return f"transfer_to_{agent_name}"


def is_handoff_tool_name(name: Optional[str]) -> bool:
    return bool(name) and str(name).startswith("transfer_to_")


def agent_from_handoff_tool_name(name: str) -> str:
    return str(name).removeprefix("transfer_to_")


__all__ = [
    "BRIEF_HEADING",
    "SPECIALISTS",
    "agent_from_handoff_tool_name",
    "build_handoff_tools",
    "handoff_messages",
    "handoff_tool_name",
    "is_handoff_tool_name",
    "make_handoff_tool",
    "render_brief",
    "superseded_handoff",
    "validate_brief",
]
