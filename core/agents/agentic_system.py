from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from app.config import (
    APPROVAL_JUDGE_MODEL,
    CONTEXT_COMPRESSION,
    CONTEXT_SUMMARY_MODEL,
    DATA_MODEL,
    OPENAI_API_KEY,
    PLANNING_MODEL,
    PREDICTION_MODEL,
    REPORT_MODEL,
    RESEARCH_MODEL,
    SUPERVISOR_MODEL,
    TASK_CLASSIFIER_MODEL,
    logger,
)
from backend.utils import plan_store
from core.agents.agents import (
    build_data_agent,
    build_planning_agent,
    build_prediction_agent,
    build_report_agent,
    build_research_agent,
    build_supervisor_agent,
)
from core.agents.context import (
    AgentGraphState,
    clipped_messages_for_summary,
    coerce_text,
    describe_prior_context,
    has_completed_turn,
    latest_summary_record,
    latest_user_request_text,
    make_summary_message,
    normalize_memory,
    render_transcript,
    should_summarize,
)
from core.prompts.prompts import (
    CONTEXT_SUMMARY_PROMPT,
    REPORT_META_SYSTEM_PROMPT,
    REPORT_SIMPLE_SYSTEM_PROMPT,
    REPORT_SYSTEM_PROMPT,
    TASK_CLASSIFIER_SYSTEM_PROMPT,
)
from core.tools.plan_tools import render_ledger

TaskCategory = Literal["simple", "complex", "meta_query", "follow_up"]


# Structured outputs for the two decision-only model calls 
class TaskClassification(BaseModel):
    category: TaskCategory = Field(description="Classification of the user's latest request.")


class PlanFeedbackVerdict(BaseModel):
    decision: Literal["approve", "revise"] = Field(
        description=("approve = the user authorizes execution now, even if they attach conditions. revise = they want the plan changed first, are asking a question, or are uncertain.")
    )
    constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Conditions the user attached to an approval, each as one imperative instruction for the executor. Empty for an unconditional approval."
        ),
    )

# For the context compress node
class ContextOutputRecord(BaseModel):
    path: str = Field(default="", description="Full path of a produced artifact.")
    description: str = Field(default="", description="What the artifact contains.")

class ContextMemory(BaseModel):
    facts: List[str] = Field(
        default_factory=list,
        description="Established findings with their values, units, identifiers and sources.",
    )
    outputs: List[ContextOutputRecord] = Field(default_factory=list)
    decisions: List[str] = Field(default_factory=list, description="Choices made, each with its reason.")
    open_questions: List[str] = Field(
        default_factory=list, description="Unverified, blocked, assumed or unresolved items."
    )

class ContextDigest(BaseModel):
    summary: str = Field(description="Evidence-first record of the work so far.")
    memory: ContextMemory = Field(default_factory=ContextMemory)


# Helper models

_approval_judge = None
_context_summarizer = None
_task_classifier = None


# If we would like to change models, or model's providers, chaneg it from here only
def _chat(model_name: str):
    return init_chat_model(model_name, model_provider="openai", api_key=OPENAI_API_KEY)

def _get_approval_judge():
    global _approval_judge
    if _approval_judge is None:
        _approval_judge = _chat(APPROVAL_JUDGE_MODEL).with_structured_output(PlanFeedbackVerdict)
    return _approval_judge

def _get_context_summarizer():
    global _context_summarizer
    if _context_summarizer is None:
        _context_summarizer = _chat(CONTEXT_SUMMARY_MODEL).with_structured_output(ContextDigest)
    return _context_summarizer

def _get_task_classifier():
    global _task_classifier
    if _task_classifier is None:
        _task_classifier = _chat(TASK_CLASSIFIER_MODEL).with_structured_output(TaskClassification)
    return _task_classifier

def reset_model_cache() -> None:
    '''Drop the cached helper models.'''

    global _approval_judge, _context_summarizer, _task_classifier
    _approval_judge = _context_summarizer = _task_classifier = None

def install_helper_models(
    *,
    task_classifier=None,
    approval_judge=None,
    context_summarizer=None,
) -> None:
    '''Substitute the three decision-only models.

    Parameters:
    ---------
    task_classifier (chat model): replacement for the routing model.
    approval_judge (chat model): replacement for the model that reads plan feedback.
    context_summarizer (chat model): replacement for the model that folds a finished turn into the carried summary.
    '''

    global _approval_judge, _context_summarizer, _task_classifier
    if task_classifier is not None:
        _task_classifier = task_classifier
    if approval_judge is not None:
        _approval_judge = approval_judge
    if context_summarizer is not None:
        _context_summarizer = context_summarizer


# Nodes 
def _latest_human_text(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return coerce_text(getattr(message, "content", ""))
    return ""


async def task_classifier_node(state: AgentGraphState) -> Dict[str, Any]:
    '''Route the request: simple / complex / meta_query / follow_up.

    Parameters:
    ---------
    state (AgentGraphState): graph state, read for the message plus the conversation goal and previous exchange.

    Returns:
    ----------
    update (dict): the chosen `task_category` — `simple`, `complex`, `meta_query` or `follow_up`. Routing is on dependence, not size: 'now rank those by hERG risk' cannot be classified from the bare message.
    '''

    messages = list(state.get("messages") or [])
    user_text = latest_user_request_text(messages) or _latest_human_text(messages)
    if not user_text:
        return {"task_category": "complex"}

    # A follow-up cannot be recognised from the bare message ("now rank those by hERG risk"), so the classifier is given the goal and the last exchange.
    prior = describe_prior_context(messages)
    can_follow_up = has_completed_turn(messages)
    if prior:
        payload = f"{prior}\n\n--- NEW USER MESSAGE ---\n{user_text}"
    else:
        payload = (
            "This is the first request in the conversation; there is no prior "
            f"exchange.\n\n--- NEW USER MESSAGE ---\n{user_text}"
        )

    try:
        result: TaskClassification = await _get_task_classifier().ainvoke(
            [
                SystemMessage(content=TASK_CLASSIFIER_SYSTEM_PROMPT),
                HumanMessage(content=payload),
            ]
        )
        category: TaskCategory = result.category
    except Exception as exc:  # noqa: BLE001 - a routing failure must not end the run
        logger.warning("Task classifier failed; defaulting to complex: %s", exc)
        category = "complex"

    if category == "follow_up" and not can_follow_up:
        category = "complex"

    updates: Dict[str, Any] = {"task_category": category}
    if category in ("complex", "simple"):
        # An approval belongs to the plan it was given for. A new task is not
        # governed by it, whether or not it gets its own plan. Follow-ups continue
        # under the standing approval; a meta query leaves it alone so an aside
        # does not discard a plan the next follow-up needs.
        updates["approved_plan"] = ""
        updates["approval_constraints"] = []
    return updates


def _latest_plan(messages: List[BaseMessage]) -> str:
    for message in reversed(messages):
        if getattr(message, "name", None) == "planning_agent":
            return coerce_text(getattr(message, "content", "")) or ""
    return ""


def _judge_plan_feedback(feedback: str, plan: str = "") -> tuple[str, List[str]]:
    '''Map free-text plan feedback to a decision plus any attached conditions.

    Parameters:
    ---------
    feedback (str): what the user replied to the proposed plan.
    plan (str): the plan they were replying to.

    Returns:
    ----------
    decision (tuple): `(verdict, constraints)` — `approve` or `revise`, plus any conditions the approval carried, such as 'go ahead, but only phase 3 drugs'. Defaults to `revise` on any failure.
    '''

    if not feedback:
        return "revise", []
    prompt = (
        "You evaluate a user's feedback on a proposed execution plan.\n\n"
        "Decide:\n"
        "- approve -> the user authorizes execution now. This still counts as an "
        'approval when they attach conditions or corrections ("go ahead, but ...", '
        '"yes, just use X instead of Y").\n'
        "- revise -> the user wants the plan itself reworked first, asks a question, "
        "or is uncertain.\n\n"
        "If the decision is approve, list every condition they attached as a separate "
        "imperative instruction for the executor, preserving their exact numbers, "
        "units and identifiers. Return an empty list for an unconditional approval, "
        "and never invent a constraint they did not state.\n\n"
        f"--- PLAN UNDER REVIEW ---\n{plan or '(plan text unavailable)'}\n\n"
        f"--- USER FEEDBACK ---\n{feedback}\n"
    )
    try:
        verdict: PlanFeedbackVerdict = _get_approval_judge().invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Approval judge failed; defaulting to revise: %s", exc)
        return "revise", []
    if verdict.decision == "approve":
        return "approved", [item.strip() for item in (verdict.constraints or []) if str(item).strip()]
    return "revise", []


def human_chat_node(state: AgentGraphState) -> Dict[str, Any]:
    '''Pause for plan review.

    Parameters:
    ---------
    state (AgentGraphState): graph state, holding the plan awaiting review.

    Returns:
    ----------
    update (dict): the interrupt that pauses for plan review, or the routing decision once the user has replied.
    '''

    plan = _latest_plan(list(state.get("messages") or []))
    feedback = (
        interrupt(
            {
                "type": "plan_review",
                "plan": plan,
                "message": (
                    "Review the plan. Approve it to start execution, or describe what "
                    "should change and the planner will revise it."
                ),
            }
        )
        or ""
    ).strip()

    decision, constraints = _judge_plan_feedback(feedback, plan)

    # The user's words go on the record either way. A conditional approval used to be reduced to a status flag, and the condition never reached the executor.
    messages: List[BaseMessage] = [HumanMessage(content=feedback)] if feedback else []

    if decision == "approved":
        return {
            "messages": messages,
            "plan_status": "approved",
            "approved_plan": plan,
            "approval_constraints": constraints,
        }
    if not messages:
        messages.append(HumanMessage(content="Please revise the plan."))
    return {"messages": messages, "plan_status": "revise"}


def approval_ack_node(state: AgentGraphState) -> Dict[str, Any]:
    constraints = state.get("approval_constraints") or []
    content = "Thank you for approving the plan. The task is now started."
    if constraints:
        listed = "\n".join(f"- {item}" for item in constraints)
        content += (
            "\n\nYour approval carried these conditions, which override the "
            f"corresponding plan steps:\n{listed}"
        )
    return {"messages": [AIMessage(content=content, name="approval_ack")]}


def _scope(state: AgentGraphState) -> Dict[str, Any]:
    return {"conversation_id": state.get("conversation_id")}


def _plan_title(request: str) -> str:
    text = " ".join((request or "Handle the request").split())
    return text if len(text) <= 120 else text[:119].rstrip() + "…"


def plan_init_node(state: AgentGraphState) -> Dict[str, Any]:
    '''Write this run's section of `plan.md`, then display it.

    Parameters:
    ---------
    state (AgentGraphState): graph state, holding the approved plan and any approval constraints.

    Returns:
    ----------
    update (dict): the plan pointer and the rendered ledger. Runs after approval, so revising a plan several times leaves no spurious runs, and contains no LLM call.
    '''

    category = str(state.get("task_category") or "complex")
    messages = list(state.get("messages") or [])
    request = latest_user_request_text(messages) or _latest_human_text(messages)

    if category == "complex":
        plan_text = coerce_text(state.get("approved_plan")) or _latest_plan(messages)
        steps = plan_store.parse_plan_steps(plan_text)
        goal = plan_store.parse_plan_goal(plan_text) or request
    else:
        # Routes without a plan still get a section, so the document stays a
        # complete record of the conversation and a follow-up can read it.
        steps = [plan_store.PlanStep(number=1, title=_plan_title(request))]
        goal = request

    try:
        run = plan_store.start_run(
            goal=goal,
            steps=steps,
            kind=category,
            constraints=list(state.get("approval_constraints") or []),
            **_scope(state),
        )
    except OSError as exc:
        # Bookkeeping must never stop the actual work.
        logger.warning("Could not write the plan file: %s", exc)
        return {}

    path = str(plan_store.plan_file_path(**_scope(state)))
    return {
        "messages": [AIMessage(content=render_ledger(run, path=path), name="plan_init")],
        "plan_path": path,
        "plan_run_id": run.run_id,
        "plan_progress": plan_store.progress_line(run),
    }


def plan_finalize_node(state: AgentGraphState) -> Dict[str, Any]:
    '''Reconcile the plan file after execution and record the run's outcome.

    Parameters:
    ---------
    state (AgentGraphState): graph state, holding the plan pointer.

    Returns:
    ----------
    update (dict): the run's recorded outcome, read back from the file rather than from the transcript. Contains no LLM call.
    '''

    scope = _scope(state)
    try:
        document = plan_store.load_document(**scope)
    except OSError as exc:
        logger.warning("Could not read the plan file: %s", exc)
        return {}

    run_id = state.get("plan_run_id")
    run = document.run(run_id) if run_id else document.active
    if run is None:
        return {}

    outcome = plan_store.summarize_outcome(run)
    plan_store.set_outcome(outcome=outcome, run_id=run.run_id, **scope)
    return {
        "messages": [
            AIMessage(
                content=f"Plan file updated — {outcome}\n{state.get('plan_path', '')}".strip(),
                name="plan_finalize",
            )
        ],
        "plan_progress": plan_store.progress_line(run),
    }


async def compress_context_node(state: AgentGraphState) -> Dict[str, Any]:
    '''Fold the finished turn into the carry-forward record.

    Parameters:
    ---------
    state (AgentGraphState): graph state, holding the turn just finished.

    Returns:
    ----------
    update (dict): the refreshed carry-forward summary.
    '''

    messages = list(state.get("messages") or [])
    if not messages or not should_summarize(messages):
        return {}

    source = clipped_messages_for_summary(messages)
    transcript = render_transcript(source)
    if not transcript.strip():
        return {}

    _, previous_summary, previous_memory = latest_summary_record(messages)
    import json

    payload = HumanMessage(
        content=(
            "Existing summary:\n"
            f"{previous_summary or '(none)'}\n\n"
            "Existing structured memory JSON:\n"
            f"{json.dumps(previous_memory or {}, ensure_ascii=True)}\n\n"
            "New messages since that summary:\n"
            f"{transcript}\n"
        )
    )

    try:
        digest: ContextDigest = await _get_context_summarizer().ainvoke(
            [SystemMessage(content=CONTEXT_SUMMARY_PROMPT), payload]
        )
    except Exception as exc:  # noqa: BLE001 - losing a summary is survivable
        logger.warning("Context compression failed: %s", exc)
        return {}

    summary_text = (digest.summary or "").strip() or (previous_summary or "")
    if not summary_text:
        return {}
    memory = normalize_memory(digest.memory.model_dump(), previous_memory)
    return {"messages": [make_summary_message(summary_text, memory)]}


# Routers 
def _route_after_classifier(state: AgentGraphState) -> str:
    category = str(state.get("task_category") or "complex")
    if category == "meta_query":
        return "report_agent_meta"
    if category in ("simple", "follow_up"):
        return "plan_init"
    return "planning_agent"

def _route_after_human(state: AgentGraphState) -> str:
    return "approval_ack" if state.get("plan_status") == "approved" else "planning_agent"


def _route_after_plan_finalize(state: AgentGraphState) -> str:
    category = str(state.get("task_category") or "complex")
    return "report_agent_simple" if category in ("simple", "follow_up") else "report_agent_complex"


# Build graph
def build_execution_subgraph(
    *,
    supervisor,
    research_agent,
    prediction_agent,
    data_agent,
):
    '''Execution graph has supervisor and its sub-agents, as one node from the outside.

    Parameters:
    ---------
    supervisor (CompiledGraph): the orchestrator.
    research_agent (CompiledGraph): the literature and knowledge-graph specialist.
    prediction_agent (CompiledGraph): the ADMET specialist.
    data_agent (CompiledGraph): the code-running specialist.

    Returns:
    ----------
    subgraph (CompiledGraph): supervisor plus specialists as one node. The supervisor deliberately has no outgoing edge: a static edge would fire in parallel with the handoff and run `plan_finalize` before any specialist had done anything.
    '''

    graph = StateGraph(AgentGraphState)

    graph.add_node("supervisor", supervisor)
    graph.add_node("research_agent", research_agent)
    graph.add_node("prediction_agent", prediction_agent)
    graph.add_node("data_agent", data_agent)

    graph.add_edge(START, "supervisor")
    graph.add_edge("research_agent", "supervisor")
    graph.add_edge("prediction_agent", "supervisor")
    graph.add_edge("data_agent", "supervisor")

    return graph.compile()


def build_graph(
    checkpointer=None,
    *,
    models: Optional[Dict[str, Any]] = None,
    use_context_compression: bool = CONTEXT_COMPRESSION,
):
    '''Compile the whole system.

    Parameters:
    ---------
    checkpointer (AsyncSqliteSaver): the checkpointer to persist conversation state with, or None.
    models (dict): per-role model overrides, keyed by role name.
    use_context_compression (boolean): whether older turns are folded into the carried summary.

    Returns:
    ----------
    graph (CompiledGraph): the whole system, classifier through report.
    '''

    models = models or {}

    def model_for(role: str, default_name: str):
        if role in models:
            return models[role]
        return _chat(default_name)

    compress = bool(use_context_compression)

    supervisor = build_supervisor_agent(model_for("supervisor", SUPERVISOR_MODEL), compress=compress)
    execution = build_execution_subgraph(
        supervisor=supervisor,
        research_agent=build_research_agent(model_for("research", RESEARCH_MODEL), compress=compress),
        prediction_agent=build_prediction_agent(model_for("prediction", PREDICTION_MODEL), compress=compress),
        data_agent=build_data_agent(model_for("data", DATA_MODEL), compress=compress),
    )

    planning_agent = build_planning_agent(model_for("planning", PLANNING_MODEL), compress=compress)
    report_model = model_for("report", REPORT_MODEL)
    report_variants = {
        "report_agent_complex": REPORT_SYSTEM_PROMPT,
        "report_agent_simple": REPORT_SIMPLE_SYSTEM_PROMPT,
        "report_agent_meta": REPORT_META_SYSTEM_PROMPT,
    }

    graph = StateGraph(AgentGraphState)
    graph.add_node("task_classifier", task_classifier_node)
    graph.add_node("planning_agent", planning_agent)
    graph.add_node("human_chat", human_chat_node)
    graph.add_node("approval_ack", approval_ack_node)
    graph.add_node("plan_init", plan_init_node)
    graph.add_node("execution", execution)
    graph.add_node("plan_finalize", plan_finalize_node)
    for name, prompt in report_variants.items():
        graph.add_node(name, build_report_agent(report_model, name=name, prompt=prompt, compress=compress))

    if compress:
        # One compression node per terminal branch, so the graph stays acyclic and the streamed label stays meaningful.
        for suffix in ("complex", "simple", "meta"):
            graph.add_node(f"context_summary_{suffix}", compress_context_node)

    graph.add_edge(START, "task_classifier")
    graph.add_conditional_edges(
        "task_classifier",
        _route_after_classifier,
        {
            "planning_agent": "planning_agent",
            "plan_init": "plan_init",
            "report_agent_meta": "report_agent_meta",
        },
    )

    graph.add_edge("planning_agent", "human_chat")
    graph.add_conditional_edges(
        "human_chat",
        _route_after_human,
        {"planning_agent": "planning_agent", "approval_ack": "approval_ack"},
    )
    graph.add_edge("approval_ack", "plan_init")

    graph.add_edge("plan_init", "execution")
    graph.add_edge("execution", "plan_finalize")
    graph.add_conditional_edges(
        "plan_finalize",
        _route_after_plan_finalize,
        {
            "report_agent_complex": "report_agent_complex",
            "report_agent_simple": "report_agent_simple",
        },
    )

    if compress:
        graph.add_edge("report_agent_complex", "context_summary_complex")
        graph.add_edge("report_agent_simple", "context_summary_simple")
        graph.add_edge("report_agent_meta", "context_summary_meta")
        for suffix in ("complex", "simple", "meta"):
            graph.add_edge(f"context_summary_{suffix}", END)
    else:
        for name in report_variants:
            graph.add_edge(name, END)

    return graph.compile(checkpointer=checkpointer)


async def create_app(checkpointer, *, use_context_compression: bool = CONTEXT_COMPRESSION):
    return build_graph(checkpointer, use_context_compression=use_context_compression)


__all__ = [
    "AgentGraphState",
    "build_execution_subgraph",
    "build_graph",
    "create_app",
    "human_chat_node",
    "install_helper_models",
    "plan_finalize_node",
    "plan_init_node",
    "reset_model_cache",
    "task_classifier_node",
]
