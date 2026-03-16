from typing import Optional, Literal, Tuple, Dict, Any, List
import json
import aiosqlite
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph_supervisor import create_supervisor
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command

from app.config import OPENAI_API_KEY, SQLITE_DB_PATH, logger, SUPERVISOR_OUTPUT_MODE
from core.agents.prediction_agent import build_prediction_agent
from core.agents.research_agent import build_research_agent
from core.agents.data_agent import build_data_agent
from core.agents.planning_agent import build_planning_agent
from core.agents.report_agent import build_report_agent
from core.prompts.prompts import SUPERVISOR_SYSTEM_PROMPT_ver3
from core.supervisor.context import (
    build_llm_input_messages,
    is_summary_message,
    messages_since_last_summary,
    SUMMARY_AGENT_NAME,
    SUMMARY_MEMORY_KEY,
    latest_summary_record,
)


def initialize_agents(
    user_request: Optional[str] = None,
    use_episodic_learning: bool = True,
    pre_model_hook=None,
):
    """Initialize all agents with optional episodic learning for planning agent."""
    planning_llm = init_chat_model('gpt-4o', model_provider='openai', streaming=True)
    data_llm = init_chat_model('gpt-5.2', model_provider='openai', streaming=True)
    research_llm = init_chat_model('gpt-5-mini', model_provider='openai', streaming=True)
    prediction_llm = init_chat_model('gpt-5-mini', model_provider='openai', streaming=True)
    report_llm = init_chat_model('gpt-5.2', model_provider='openai', streaming=True)

    research_agent = build_research_agent(research_llm, pre_model_hook=pre_model_hook)
    data_agent = build_data_agent(data_llm, message_trimmer=build_llm_input_messages)
    prediction_agent = build_prediction_agent(prediction_llm, pre_model_hook=pre_model_hook)
    planning_agent = build_planning_agent(
        planning_llm,
        user_request,
        use_episodic_learning,
        pre_model_hook=pre_model_hook,
    )
    report_agent = build_report_agent(report_llm, pre_model_hook=pre_model_hook)

    return research_agent, data_agent, prediction_agent, planning_agent, report_agent


_approval_judge_llm = None
_summary_llm = None

SUMMARY_MAX_MESSAGES = 200
SUMMARY_TRIGGER_MIN_MESSAGES_FIRST = 8
SUMMARY_TRIGGER_MIN_MESSAGES = 20
# Approximate token threshold (chars ~= tokens * 4). Keep conservative.
SUMMARY_TRIGGER_CHAR_LIMIT = 12000
MEMORY_MAX_ITEMS = 20
MEMORY_OUTPUTS_MAX_ITEMS = 20

SUMMARY_PROMPT = """You update two artifacts for ongoing context.
Inputs: existing_summary, existing_memory_json, and new_messages.
1) summary: concise narrative of the full workflow so far.
   Include user goal, key steps, decisions, outputs with file paths, errors/limitations, and open questions.
2) memory: structured facts for future steps.
   Preserve stable facts and prior outputs unless they are superseded.
Be concise and structured; use short bullets inside the summary text.
Return ONLY valid JSON with keys:
- summary (string)
- memory (object)
memory schema:
- facts: list of strings
- outputs: list of objects with keys {path, description}
- decisions: list of strings
- open_questions: list of strings
No markdown fences. No extra commentary."""


def _get_approval_judge_llm():
    global _approval_judge_llm
    if _approval_judge_llm is None:
        _approval_judge_llm = init_chat_model("gpt-5-nano", model_provider="openai", api_key=OPENAI_API_KEY)
    return _approval_judge_llm


def _get_summary_llm():
    global _summary_llm
    if _summary_llm is None:
        _summary_llm = init_chat_model("gpt-5.2", model_provider="openai", api_key=OPENAI_API_KEY)
    return _summary_llm


def _build_llm_input(state):
    messages = state.get("messages") or []
    return {"llm_input_messages": build_llm_input_messages(messages)}


def _estimate_message_chars(message) -> int:
    content = getattr(message, "content", None)
    if content is None:
        return 0
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return sum(len(part) for part in parts)
    return len(str(content))


def _should_summarize(source_messages, has_existing_summary: bool) -> bool:
    if not source_messages:
        return False
    min_messages = SUMMARY_TRIGGER_MIN_MESSAGES_FIRST if not has_existing_summary else SUMMARY_TRIGGER_MIN_MESSAGES
    if len(source_messages) >= min_messages:
        return True
    total_chars = sum(_estimate_message_chars(msg) for msg in source_messages)
    return total_chars >= SUMMARY_TRIGGER_CHAR_LIMIT


def _extract_json_payload(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _coerce_str_list(value, fallback) -> List[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, list):
        items = []
        for item in value:
            text = str(item).strip()
            if text:
                items.append(text)
        return items
    text = str(value).strip()
    return [text] if text else list(fallback or [])


def _normalize_outputs(value, fallback) -> List[Dict[str, str]]:
    outputs: List[Dict[str, str]] = []
    if value is None:
        return list(fallback or [])
    if not isinstance(value, list):
        value = [value]
    for item in value:
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            desc = str(item.get("description", "") or item.get("detail", "")).strip()
            if path or desc:
                outputs.append({"path": path, "description": desc})
            continue
        text = str(item).strip()
        if text:
            outputs.append({"path": "", "description": text})
    return outputs or list(fallback or [])


def _merge_str_lists(new_items: List[str], prior_items: List[str], max_items: int) -> List[str]:
    seen = set()
    merged: List[str] = []
    for item in new_items + prior_items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def _merge_output_lists(
    new_items: List[Dict[str, str]],
    prior_items: List[Dict[str, str]],
    max_items: int,
) -> List[Dict[str, str]]:
    seen = set()
    merged: List[Dict[str, str]] = []
    for item in new_items + prior_items:
        path = str(item.get("path", "")).strip().lower()
        desc = str(item.get("description", "")).strip().lower()
        key = f"{path}|{desc}"
        if not key.strip("|") or key in seen:
            continue
        seen.add(key)
        merged.append({"path": item.get("path", ""), "description": item.get("description", "")})
        if len(merged) >= max_items:
            break
    return merged


def _normalize_memory(candidate, prior) -> Dict[str, Any]:
    prior = prior if isinstance(prior, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}

    facts = _coerce_str_list(candidate.get("facts"), prior.get("facts"))
    outputs = _normalize_outputs(candidate.get("outputs"), prior.get("outputs"))
    decisions = _coerce_str_list(candidate.get("decisions"), prior.get("decisions"))
    open_questions = _coerce_str_list(candidate.get("open_questions"), prior.get("open_questions"))

    return {
        "facts": _merge_str_lists(facts, prior.get("facts", []), MEMORY_MAX_ITEMS),
        "outputs": _merge_output_lists(outputs, prior.get("outputs", []), MEMORY_OUTPUTS_MAX_ITEMS),
        "decisions": _merge_str_lists(decisions, prior.get("decisions", []), MEMORY_MAX_ITEMS),
        "open_questions": _merge_str_lists(open_questions, prior.get("open_questions", []), MEMORY_MAX_ITEMS),
    }


def _summarize_workflow(state):
    messages = state.get("messages") or []
    if not messages:
        return {}
    if is_summary_message(messages[-1]):
        return {}

    source_messages = messages_since_last_summary(messages)
    if not source_messages:
        return {}
    idx, prev_summary, prev_memory = latest_summary_record(messages)
    has_existing_summary = idx >= 0 and bool(prev_summary)
    if not _should_summarize(source_messages, has_existing_summary):
        return {}
    if len(source_messages) > SUMMARY_MAX_MESSAGES:
        source_messages = source_messages[-SUMMARY_MAX_MESSAGES:]

    llm = _get_summary_llm()
    try:
        memory_json = json.dumps(prev_memory or {}, ensure_ascii=True)
        context_msg = SystemMessage(
            content=(
                "Existing summary:\\n"
                f"{prev_summary or ''}\\n\\n"
                "Existing structured memory JSON:\\n"
                f"{memory_json}\\n"
            )
        )
        response = llm.invoke([SystemMessage(content=SUMMARY_PROMPT), context_msg] + list(source_messages))
    except Exception as exc:
        logger.warning("Summary generation failed: %s", exc)
        return {}

    raw_text = getattr(response, "content", str(response)).strip()
    if not raw_text:
        return {}

    payload = _extract_json_payload(raw_text)
    if payload is None:
        summary_text = raw_text
        memory = _normalize_memory(prev_memory, prev_memory)
    else:
        summary_text = str(payload.get("summary", "")).strip()
        memory = _normalize_memory(payload.get("memory"), prev_memory)
        if not summary_text:
            summary_text = prev_summary or ""
    if not summary_text:
        return {}

    summary_msg = AIMessage(
        content=summary_text,
        name=SUMMARY_AGENT_NAME,
        response_metadata={"is_summary": True, SUMMARY_MEMORY_KEY: memory},
    )
    return {"messages": [summary_msg]}


def _judge_plan_feedback(feedback: str) -> Literal["approve", "revise"]:
    """Use a lightweight LLM to classify whether the user approved or requested revisions."""
    if not feedback:
        return "revise"
    llm = _get_approval_judge_llm()
    prompt = (
        "You evaluate a human's feedback on an execution plan.\\n"
        "Reply with EXACTLY one word:\\n"
        "- APPROVE → the human explicitly authorizes execution immediately.\\n"
        "- REVISE → the human asks for changes, more info, or expresses uncertainty.\\n"
        "Do not add punctuation or commentary.\\n"
        f"Feedback: {feedback}\\n"
    )
    try:
        response = llm.invoke(prompt)
        content = getattr(response, "content", str(response)).strip().lower()
    except Exception as exc:  # pragma: no cover - defensive logging
        logger.warning("Approval judge failed, defaulting to revise: %s", exc)
        return "revise"
    if content.startswith("approve"):
        return "approve"
    if content.startswith("revise"):
        return "revise"
    logger.info("Approval judge returned unrecognized answer '%s'; defaulting to revise", content)
    return "revise"


async def initialize_memory() -> Tuple[AsyncSqliteSaver | MemorySaver, Optional[aiosqlite.Connection]]:
    """Initialize memory/checkpointer (async-aware)."""
    try:
        connection = await aiosqlite.connect(str(SQLITE_DB_PATH))
        memory = AsyncSqliteSaver(connection)
        logger.info(f"AsyncSqliteSaver initialized with database: {SQLITE_DB_PATH}")
        return memory, connection
    except Exception as e:
        logger.error(f"Error initializing AsyncSqliteSaver: {e}")
        # Fallback to in-memory for development
        memory = MemorySaver()
        logger.warning("Falling back to MemorySaver")
        return memory, None


def _latest_user_text(state) -> str:
    """Extract the latest user message from the state."""
    msgs = state.get("messages") or []
    # Walk from the end to get the most recent user message
    for m in reversed(msgs):
        # LangChain message objects
        if isinstance(m, HumanMessage):
            return m.content or ""
        # Fallbacks for other LC message types that expose `.type`
        if getattr(m, "type", None) == "human":
            return getattr(m, "content", "") or ""
        # Dict-style messages
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def route_from_start(state) -> Literal["plan", "skip"]:
    """Route user requests either to planning agent first or directly to supervisor."""
    user_text = _latest_user_text(state)

    # Default to planning if we couldn't find any user text
    if not user_text:
        return "plan"

    # Initialize LLM for routing decision
    llm = init_chat_model('gpt-4o', model_provider='openai', streaming=True)

    prompt = (
        "You are a router for an agent workflow.\\n"
        "If the request is concrete and ready to execute, answer 'skip'.\\n"
        "If the request is vague/complex and needs decomposition, answer 'plan'.\\n"
        f"Request: {user_text}\\n"
        "Answer with exactly one word: skip or plan."
    )

    out = llm.invoke(prompt)
    # llm.invoke returns an AIMessage (LangChain) or a string depending on your wrapper
    out_text = getattr(out, "content", str(out)).strip().lower()

    return "skip" if out_text.startswith("skip") else "plan"


def route_from_planning(state) -> Literal["human_chat", "supervisor"]:
    """Route from planning agent - check if human message contains approval terms."""
    messages = state.get("messages", [])

    # Get all human messages in chronological order
    human_messages = []
    for msg in messages:
        human_content = None

        # Check different message formats for human messages
        if hasattr(msg, 'type') and msg.type == 'human':
            human_content = msg.content
        elif isinstance(msg, dict) and msg.get('role') == 'user':
            human_content = msg.get('content', '')
        elif hasattr(msg, 'role') and msg.role == 'user':
            human_content = msg.content

        if human_content:
            human_messages.append(human_content)

    # CRITICAL FIX: Only check messages AFTER the first one (exclude original user request)
    if len(human_messages) <= 1:
        # Only the original request exists, no approval possible yet
        logger.info("Only original request exists, routing to human_chat for plan review")
        return "human_chat"

    # Treat a follow-up user message right after supervisor/report output as a fresh request
    last_human_idx = None
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        is_human = (
            (hasattr(msg, "type") and msg.type == "human")
            or (isinstance(msg, dict) and msg.get("role") == "user")
            or (hasattr(msg, "role") and msg.role == "user")
        )
        if is_human:
            last_human_idx = idx
            break

    if last_human_idx is not None:
        prev_idx = last_human_idx - 1
        if prev_idx >= 0:
            prev_msg = messages[prev_idx]
            prev_name = None
            if hasattr(prev_msg, "name"):
                prev_name = prev_msg.name
            elif isinstance(prev_msg, dict):
                prev_name = prev_msg.get("name")
            if prev_name in {"supervisor", "report_agent", SUMMARY_AGENT_NAME}:
                logger.info("Detected follow-up after %s, routing to human_chat for plan review", prev_name)
                return "human_chat"

    # Check the most recent human message (excluding the first) for approval terms
    most_recent_feedback = human_messages[-1]
    decision = _judge_plan_feedback(most_recent_feedback)
    if decision == "approve":
        logger.info("Approval judge confirmed plan approval")
        return "supervisor"
    logger.info("Approval judge requested more revisions")
    return "human_chat"


def route_from_supervisor(state) -> Literal["summary", "skip"]:
    """Send to summary only when supervisor terminates (no handoff tool calls)."""
    messages = state.get("messages") or []
    if not messages:
        return "skip"

    last_supervisor = None
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (getattr(msg, "name", None) == "supervisor"):
            last_supervisor = msg
            break

    if last_supervisor is None:
        return "skip"

    tool_calls = getattr(last_supervisor, "tool_calls", None) or []
    if tool_calls:
        return "skip"

    # If the last message is a handoff tool confirmation, do not summarize.
    last_meta = getattr(messages[-1], "response_metadata", None) or {}
    if last_meta.get("__handoff_destination"):
        return "skip"

    return "summary"


def human_chat_node(state):
    """Handle human-in-the-loop conversation for plan approval."""
    from langgraph.types import interrupt

    # Get the plan from state
    messages = state.get("messages", [])
    planning_output = ""

    # Extract the latest planning agent output
    for msg in reversed(messages):
        if hasattr(msg, 'name') and msg.name == 'planning_agent':
            planning_output = msg.content
            break
        elif isinstance(msg, dict) and msg.get('name') == 'planning_agent':
            planning_output = msg.get('content', '')
            break

    # Interrupt for human input with the current plan
    human_input = interrupt({
        "type": "plan_review",
        "plan": planning_output,
        "message": "Please review the plan above. You can:\\n1. Ask for changes or refinements\\n2. Click 'Approve Plan' to proceed with execution"
    })

    # Check if human approved the plan or wants to refine it
    if human_input and human_input.lower().strip() == "approved":
        return {"plan_approved": True}
    else:
        # Continue conversation - add human feedback to messages
        if human_input:
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=human_input))
        return {"messages": messages, "plan_approved": False}


async def create_app(user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """
    Initialize the LangGraph application with episodic learning for planning agent.

    Args:
        user_request: Current user request for context-aware planning agent enhancement
        use_episodic_learning: Whether to use episodic learning for planning agent
    """
    llm = init_chat_model('gpt-5-mini', model_provider='openai', streaming=True)
    pre_model_hook = _build_llm_input

    # Build agents with episodic learning for planning agent
    research_agent, data_agent, prediction_agent, planning_agent, report_agent = initialize_agents(
        user_request, use_episodic_learning, pre_model_hook
    )

    # Initialize memory
    memory, connection = await initialize_memory()

    # Create supervisor with execution agents (planning agent added separately, report agent included but routed to END)
    supervisor_agent = create_supervisor(
        [research_agent, prediction_agent, data_agent, report_agent],
        model=llm,
        output_mode=SUPERVISOR_OUTPUT_MODE,
        prompt=SUPERVISOR_SYSTEM_PROMPT_ver3,
        add_handoff_message=True,
        supervisor_name='supervisor',
        pre_model_hook=pre_model_hook,
    )

    # Modify the graph structure to add routing and human-in-the-loop
    # Remove the default START -> supervisor edge
    supervisor_agent.edges.remove(('__start__', 'supervisor'))

    # Add planning agent as a separate node
    supervisor_agent.add_node('planning_agent', planning_agent)

    # Add human chat node for plan approval
    supervisor_agent.add_node('human_chat', human_chat_node)

    # Add summary node
    supervisor_agent.add_node('summary', _summarize_workflow)

    # Remove default edge from report_agent back to supervisor (similar to planning_agent)
    supervisor_agent.edges.remove(('report_agent', 'supervisor'))

    # Add conditional routing from START
    supervisor_agent.add_conditional_edges(
        START,
        route_from_start,
        {"plan": "planning_agent", "skip": "supervisor"},
    )

    # Add conditional routing from planning agent
    supervisor_agent.add_conditional_edges(
        'planning_agent',
        route_from_planning,
        {"human_chat": "human_chat", "supervisor": "supervisor"},
    )

    # Add edge from human_chat back to planning_agent for refinements
    supervisor_agent.add_edge('human_chat', 'planning_agent')

    # Add edge from report_agent to summary, then END
    supervisor_agent.add_edge('report_agent', 'summary')
    supervisor_agent.add_edge('summary', END)

    # When supervisor finishes without a handoff, route to summary
    supervisor_agent.add_conditional_edges(
        'supervisor',
        route_from_supervisor,
        {"summary": "summary", "skip": END},
    )

    app = supervisor_agent.compile(checkpointer=memory)

    if use_episodic_learning and user_request:
        logger.info("Created app with episodic learning enhancement and separate planning node")
    else:
        logger.info("Created app with standard agents and separate planning node")

    return app, memory, connection
