'''Shared graph state, and what each agent actually sees before a model call.

Two things live here.

**`AgentGraphState`** is the state every node and every agent shares. It must be
passed to `create_agent(..., state_schema=AgentGraphState)`: without it LangGraph
filters the parent state down to the plain `AgentState`, and
`conversation_id`, `approved_plan` and the plan pointer never reach the middleware
that needs them.

**The context middleware** rewrites each model call. It is turn-anchored rather
than summary-anchored, and assembles four things:

1. a pinned block compression can never remove — the output scope, the
   conversation goal, the plan pointer and progress, the approval conditions, and
   an artifact ledger read from the filesystem rather than from a list the model
   maintains;
2. the compressed summary of older turns, explicitly marked as less reliable than
   what follows;
3. the last few completed exchanges **verbatim** — a follow-up like "now rank
   those by hERG risk" refers to exactly these, and they are small;
4. the live turn, with tool traffic bounded in place.

Tool results are shortened, never dropped: removing a `ToolMessage` would orphan
its `tool_call` and fail the agent's own history validation.
'''

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain.agents.middleware import AgentState, wrap_model_call, wrap_tool_call
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import (
    CONTEXT_ANCHOR_REPORT_MAX_CHARS,
    CONTEXT_ANCHOR_REQUEST_MAX_CHARS,
    CONTEXT_ARTIFACT_MAX_ITEMS,
    CONTEXT_GOAL_MAX_CHARS,
    CONTEXT_KEEP_TURNS,
    EPISODIC_CACHE_TTL_SECONDS,
    MEMORY_MAX_ITEMS,
    MEMORY_OUTPUTS_MAX_ITEMS,
    SUMMARY_MAX_MESSAGES,
    SUMMARY_SOURCE_MAX_CHARS,
    SUMMARY_SOURCE_MESSAGE_MAX_CHARS,
    SUMMARY_TRIGGER_CHAR_LIMIT,
    SUMMARY_TRIGGER_MIN_MESSAGES,
    SUMMARY_TRIGGER_MIN_MESSAGES_FIRST,
    TOOL_RESULT_ELIDED_CHARS,
    TOOL_RESULT_MAX_CHARS,
    TOOL_RESULT_PERSIST_MAX_CHARS,
    TOOL_RESULT_RECENT_FULL,
    logger,
)
from backend.utils.output_paths import describe_output_artifacts, describe_output_scope

SUMMARY_AGENT_NAME = "context_summary"
SUPERVISOR_AGENT_NAME = "supervisor"
SUMMARY_MEMORY_KEY = "summary_memory"
PLANNING_AGENT_NAME = "planning_agent"
REPORT_AGENT_PREFIX = "report_agent"
# Stamped on the HumanMessage that opens a user turn, so turn boundaries stay
# unambiguous next to the HumanMessages plan review writes into the same transcript.
TURN_ROLE_KEY = "repuragent_turn_role"
TURN_ROLE_REQUEST = "user_request"


class AgentGraphState(AgentState, total=False):
    '''State shared by the top-level graph and every agent inside it.'''

    conversation_id: str
    # Routing: simple | complex | follow_up | meta_query
    task_category: str
    plan_status: str
    # The approved plan text, pinned so the executor sees one authoritative plan
    # instead of every draft the planner produced.
    approved_plan: str
    approval_constraints: List[str]
    # The plan itself lives on disk; state carries a pointer and a progress line.
    plan_path: str
    plan_run_id: int
    plan_progress: str
    # Per-conversation UI toggle, read by the planning agent's middleware.
    use_episodic_learning: bool


# --- Text helpers -------------------------------------------------------------


def coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", "")).strip()
    return str(content).strip()


def _shorten(text: str, limit: int) -> str:
    '''Trim to `limit`, keeping both ends when there is room for both.

    Parameters:
    ---------
    text (str): the text to trim.
    limit (int): how many characters may survive.

    Returns:
    ----------
    shortened (str): the text, keeping the head and the tail when there is room for both.
    '''

    if limit <= 0 or len(text) <= limit:
        return text
    if limit < 400:
        return text[:limit] + f"\n… [{len(text) - limit:,} more characters omitted]"
    head = int(limit * 0.7)
    tail = limit - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n… [{omitted:,} characters omitted] …\n{text[-tail:]}"


# --- Summary records ----------------------------------------------------------


def is_summary_message(message: BaseMessage) -> bool:
    name = getattr(message, "name", None)
    if name and str(name).lower() == SUMMARY_AGENT_NAME:
        return True
    metadata = getattr(message, "response_metadata", None) or {}
    return bool(metadata.get("is_summary"))


def latest_summary_record(
    messages: Sequence[BaseMessage],
) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if is_summary_message(message):
            metadata = getattr(message, "response_metadata", None) or {}
            return index, coerce_text(getattr(message, "content", None)), metadata.get(SUMMARY_MEMORY_KEY)
    return -1, None, None


def messages_since_last_summary(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    if not messages:
        return []
    index, _, _ = latest_summary_record(messages)
    if index < 0:
        return list(messages)
    return list(messages[index + 1 :])


def make_summary_message(summary_text: str, memory: Dict[str, Any]) -> AIMessage:
    return AIMessage(
        content=summary_text,
        name=SUMMARY_AGENT_NAME,
        response_metadata={"is_summary": True, SUMMARY_MEMORY_KEY: memory},
    )


# --- Turn boundaries ----------------------------------------------------------


def mark_user_request(message: BaseMessage) -> BaseMessage:
    '''Tag a HumanMessage as opening a user turn.

    Parameters:
    ---------
    message (BaseMessage): the HumanMessage that opens a user turn.

    Returns:
    ----------
    message (BaseMessage): the same message carrying the turn marker, since plan review writes its own HumanMessages into the transcript and turn boundaries cannot be inferred from type alone.
    '''

    kwargs = dict(getattr(message, "additional_kwargs", None) or {})
    kwargs[TURN_ROLE_KEY] = TURN_ROLE_REQUEST
    message.additional_kwargs = kwargs
    return message


def _is_user_request(messages: Sequence[BaseMessage], index: int) -> bool:
    '''True when `messages[index]` opens a new user turn.

    Explicit marker first, then structure, so conversations checkpointed before
    the marker existed still split correctly: plan feedback is a HumanMessage that
    directly follows the plan it answers.

    Parameters:
    ---------
    messages (list): the whole transcript.
    index (int): position to test.

    Returns:
    ----------
    opens_turn (boolean): True when that message carries the turn marker.
    '''

    message = messages[index]
    if not isinstance(message, HumanMessage):
        return False
    kwargs = getattr(message, "additional_kwargs", None) or {}
    if kwargs.get(TURN_ROLE_KEY) == TURN_ROLE_REQUEST:
        return True
    if kwargs.get(TURN_ROLE_KEY):
        return False
    if index == 0:
        return True
    previous = messages[index - 1]
    return str(getattr(previous, "name", "") or "") != PLANNING_AGENT_NAME


def _turn_start_indices(messages: Sequence[BaseMessage]) -> List[int]:
    return [index for index in range(len(messages)) if _is_user_request(messages, index)]


def latest_user_request_text(messages: Sequence[BaseMessage]) -> str:
    '''The newest user turn, ignoring plan-review replies.

    Parameters:
    ---------
    messages (list): the whole transcript.

    Returns:
    ----------
    request (str): the newest user turn's text, ignoring plan-review replies.
    '''

    starts = _turn_start_indices(messages)
    if not starts:
        return ""
    return coerce_text(getattr(messages[starts[-1]], "content", None))


def has_completed_turn(messages: Sequence[BaseMessage]) -> bool:
    return len(_turn_start_indices(messages)) >= 2


def _turn_final_report(body: Sequence[BaseMessage]) -> str:
    '''The polished answer a turn ended on, if it produced one.

    Parameters:
    ---------
    body (list): the messages belonging to one completed turn.

    Returns:
    ----------
    report (str): the polished answer that turn ended on, or an empty string when it produced none.
    '''

    for message in reversed(body):
        if not isinstance(message, AIMessage) or getattr(message, "tool_calls", None):
            continue
        if str(getattr(message, "name", "") or "").startswith(REPORT_AGENT_PREFIX):
            text = coerce_text(getattr(message, "content", None))
            if text:
                return text
    return ""


def describe_prior_context(messages: Sequence[BaseMessage], *, max_chars: int = 1500) -> str:
    '''Goal plus the most recent completed exchange, for the routing decision.

    A request like "now rank them by hERG risk" cannot be classified from the bare
    text; the classifier needs to know what "them" refers to.

    Parameters:
    ---------
    messages (list): the whole transcript.
    max_chars (int): how much of it the classifier may be given.

    Returns:
    ----------
    summary (str): the goal plus the previous exchange, which is what lets `task_classifier` route on dependence rather than size.
    '''

    starts = _turn_start_indices(messages)
    if len(starts) < 2:
        return ""

    goal = coerce_text(getattr(messages[starts[0]], "content", None))
    previous_start, current_start = starts[-2], starts[-1]
    previous_request = coerce_text(getattr(messages[previous_start], "content", None))
    previous_answer = _turn_final_report(messages[previous_start + 1 : current_start])

    parts: List[str] = []
    if goal:
        parts.append("Conversation goal (first request):\n" + _shorten(goal, max_chars))
    if previous_request and previous_start != starts[0]:
        parts.append("Most recent previous request:\n" + _shorten(previous_request, max_chars))
    if previous_answer:
        parts.append("Answer already delivered for it:\n" + _shorten(previous_answer, max_chars))
    return "\n\n".join(parts)


def build_turn_anchors(
    messages: Sequence[BaseMessage],
    *,
    upto_index: int,
    keep: int = CONTEXT_KEEP_TURNS,
) -> List[BaseMessage]:
    '''Verbatim (request, answer) pairs for completed turns before `upto_index`.

    This is what stops a follow-up from starting cold. Everything else in the
    compressed region really is disposable — tool traffic, superseded plan drafts,
    delegation chatter — but the user's own words and the answer they were given
    are exactly what the next request points at.

    Parameters:
    ---------
    messages (list): the whole transcript.
    upto_index (int): where the live turn begins; only completed turns before it are anchored.
    keep (int): how many completed turns to keep verbatim.

    Returns:
    ----------
    anchors (list): the kept (request, answer) pairs, unsummarized.
    '''

    if keep <= 0 or upto_index <= 0:
        return []
    region = list(messages[:upto_index])
    starts = _turn_start_indices(region)
    if not starts:
        return []

    selected = starts[-keep:]
    anchors: List[BaseMessage] = []
    for position, start in enumerate(selected):
        following = [value for value in starts if value > start]
        end = following[0] if following else len(region)
        request_text = coerce_text(getattr(region[start], "content", None))
        if request_text:
            anchors.append(
                HumanMessage(content=_shorten(request_text, CONTEXT_ANCHOR_REQUEST_MAX_CHARS))
            )
        report = _turn_final_report(region[start + 1 : end])
        if report:
            anchors.append(AIMessage(content=_shorten(report, CONTEXT_ANCHOR_REPORT_MAX_CHARS)))
        elif position == len(selected) - 1 and request_text:
            anchors.append(AIMessage(content="(That request did not reach a final answer.)"))
    return anchors


# --- Bounding the live turn ---------------------------------------------------


def prune_tool_traffic(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    '''Bound tool-result size without breaking tool_call pairing.

    Contents are shortened in place; dropping a ToolMessage would orphan its
    AIMessage tool_call and the agent's history validation would reject it.

    Parameters:
    ---------
    messages (list): the live turn's messages, including tool calls and results.

    Returns:
    ----------
    pruned (list): the same messages with oversized results bounded in place — a ToolMessage is never dropped, because that would orphan its `tool_call` and fail history validation.
    '''

    tool_positions = [
        index for index, message in enumerate(messages) if isinstance(message, ToolMessage)
    ]
    if not tool_positions:
        return list(messages)
    keep_full = set(tool_positions[-TOOL_RESULT_RECENT_FULL:]) if TOOL_RESULT_RECENT_FULL > 0 else set()

    pruned: List[BaseMessage] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            pruned.append(message)
            continue
        content = getattr(message, "content", None)
        if not isinstance(content, str):
            pruned.append(message)
            continue
        recent = index in keep_full
        limit = TOOL_RESULT_MAX_CHARS if recent else TOOL_RESULT_ELIDED_CHARS
        if len(content) <= limit:
            pruned.append(message)
            continue
        shortened = _shorten(content, limit)
        if not recent:
            shortened += (
                "\n[Older tool result, trimmed to keep the working context small. "
                "Re-run the call if you need its full output again.]"
            )
        pruned.append(message.model_copy(update={"content": shortened}))
    return pruned


def repair_tool_pairing(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    '''Drop tool traffic that no longer pairs, so no view can be rejected by the API.

    Every view here is a *slice* of the transcript — spans collapsed, turns anchored,
    a specialist's window opened after its brief — and a slice can cut a tool pair in
    half. Both halves are fatal, and the message names the index rather than the
    cause, so it is worth making structurally impossible:

    * a `ToolMessage` whose `AIMessage` is not in the view — *"messages with role
      'tool' must be a response to a preceeding message with 'tool_calls'"*;
    * an `AIMessage` whose `tool_call` is never answered — the same rejection from
      the other side.

    `prune_tool_traffic` keeps a pair intact once it is in the list; this is the pass
    that decides what is a pair at all. It is deliberately the last thing every
    builder does, and it repairs rather than raises: a trimmed-away tool call is
    worth losing, a failed run is not.

    Parameters:
    ---------
    messages (list): an assembled view, possibly holding half a tool pair.

    Returns:
    ----------
    repaired (list): the same view with orphaned `ToolMessage`s dropped and unanswered `tool_call`s stripped from the `AIMessage`s that declared them.
    '''

    # Which call ids were declared, and where. A ToolMessage answers a call only if
    # the AIMessage declaring it comes earlier, and only once.
    declared: Dict[str, int] = {}
    for index, message in enumerate(messages):
        for call in getattr(message, "tool_calls", None) or []:
            call_id = call.get("id")
            if call_id and call_id not in declared:
                declared[call_id] = index

    answered: set = set()
    keep_tool: List[bool] = []
    for index, message in enumerate(messages):
        if not isinstance(message, ToolMessage):
            keep_tool.append(True)
            continue
        call_id = getattr(message, "tool_call_id", None)
        position = declared.get(call_id) if call_id else None
        usable = position is not None and position < index and call_id not in answered
        if usable:
            answered.add(call_id)
        keep_tool.append(bool(usable))

    repaired: List[BaseMessage] = []
    for index, message in enumerate(messages):
        if not keep_tool[index]:
            logger.debug(
                "Dropping orphaned ToolMessage (%s) from the assembled view",
                getattr(message, "name", "?"),
            )
            continue
        calls = list(getattr(message, "tool_calls", None) or [])
        if not calls:
            repaired.append(message)
            continue
        kept = [call for call in calls if call.get("id") in answered]
        if len(kept) == len(calls):
            repaired.append(message)
            continue
        logger.debug(
            "Stripping %d unanswered tool_call(s) from an AIMessage in the assembled view",
            len(calls) - len(kept),
        )
        if not kept and not coerce_text(getattr(message, "content", None)).strip():
            # Nothing left to say and nothing left to call.
            continue
        update: Dict[str, Any] = {"tool_calls": kept}
        raw = getattr(message, "additional_kwargs", None) or {}
        if isinstance(raw.get("tool_calls"), list):
            update["additional_kwargs"] = {
                **raw,
                "tool_calls": [
                    item
                    for item in raw["tool_calls"]
                    if not isinstance(item, dict) or item.get("id") in answered
                ],
            }
        repaired.append(message.model_copy(update=update))
    return repaired


def _estimate_message_chars(message: BaseMessage) -> int:
    return len(coerce_text(getattr(message, "content", None)))


def should_summarize(messages: Sequence[BaseMessage]) -> bool:
    index, previous_summary, _ = latest_summary_record(messages)
    source = messages_since_last_summary(messages)
    if not source:
        return False
    minimum = (
        SUMMARY_TRIGGER_MIN_MESSAGES_FIRST
        if index < 0 or not previous_summary
        else SUMMARY_TRIGGER_MIN_MESSAGES
    )
    if len(source) >= minimum:
        return True
    return sum(_estimate_message_chars(message) for message in source) >= SUMMARY_TRIGGER_CHAR_LIMIT


def clipped_messages_for_summary(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    '''Bound what the compressor reads: newest first, by message count and chars.

    One repurposing run can hold megabytes of knowledge-graph output; handing all
    of it to the summarizer was slow and could exceed its own context.

    Parameters:
    ---------
    messages (list): the messages available to summarize.

    Returns:
    ----------
    clipped (list): the newest of them, bounded by both message count and character budget.
    '''

    source = messages_since_last_summary(messages)
    if len(source) > SUMMARY_MAX_MESSAGES:
        source = source[-SUMMARY_MAX_MESSAGES:]

    budget = SUMMARY_SOURCE_MAX_CHARS
    selected: List[BaseMessage] = []
    for message in reversed(source):
        content = getattr(message, "content", None)
        if isinstance(content, str) and len(content) > SUMMARY_SOURCE_MESSAGE_MAX_CHARS:
            message = message.model_copy(
                update={"content": _shorten(content, SUMMARY_SOURCE_MESSAGE_MAX_CHARS)}
            )
        cost = _estimate_message_chars(message)
        if selected and cost > budget:
            break
        budget -= cost
        selected.append(message)
    selected.reverse()
    return selected


def render_transcript(messages: Sequence[BaseMessage]) -> str:
    '''Flatten messages into a labelled transcript for the compressor.

    Passing raw message objects risks sending an orphaned ToolMessage once the
    window is clipped, which the API rejects. The compressor only has to read the
    exchange, not participate in it.

    Parameters:
    ---------
    messages (list): the messages to flatten.

    Returns:
    ----------
    transcript (str): one labelled line per message, for the compressor to read.
    '''

    lines: List[str] = []
    for message in messages:
        name = str(getattr(message, "name", "") or "")
        if isinstance(message, HumanMessage):
            label = "USER"
        elif isinstance(message, ToolMessage):
            label = f"TOOL RESULT ({name or 'tool'})"
        elif isinstance(message, AIMessage):
            label = f"ASSISTANT ({name})" if name else "ASSISTANT"
        elif isinstance(message, SystemMessage):
            label = "SYSTEM"
        else:
            label = type(message).__name__.upper()

        text = coerce_text(getattr(message, "content", None))
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            requested = ", ".join(str(call.get("name", "?")) for call in tool_calls)
            text = (text + "\n" if text else "") + f"[called tools: {requested}]"
        if not text:
            continue
        lines.append(f"### {label}\n{text}")
    return "\n\n".join(lines)


# --- Structured memory --------------------------------------------------------


def _coerce_str_list(value, fallback) -> List[str]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    return [text] if text else list(fallback or [])


def _normalize_outputs(value, fallback) -> List[Dict[str, str]]:
    if value is None:
        return list(fallback or [])
    if not isinstance(value, list):
        value = [value]
    outputs: List[Dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            path = str(item.get("path", "")).strip()
            description = str(item.get("description", "") or item.get("detail", "")).strip()
            if path or description:
                outputs.append({"path": path, "description": description})
            continue
        text = str(item).strip()
        if text:
            outputs.append({"path": "", "description": text})
    return outputs or list(fallback or [])


def _merge_str_lists(new_items: List[str], prior_items: List[str], max_items: int) -> List[str]:
    seen: set[str] = set()
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
    seen: set[str] = set()
    merged: List[Dict[str, str]] = []
    for item in new_items + prior_items:
        key = f"{str(item.get('path','')).strip().lower()}|{str(item.get('description','')).strip().lower()}"
        if not key.strip("|") or key in seen:
            continue
        seen.add(key)
        merged.append({"path": item.get("path", ""), "description": item.get("description", "")})
        if len(merged) >= max_items:
            break
    return merged


def normalize_memory(candidate, prior) -> Dict[str, Any]:
    prior = prior if isinstance(prior, dict) else {}
    candidate = candidate if isinstance(candidate, dict) else {}
    return {
        "facts": _merge_str_lists(
            _coerce_str_list(candidate.get("facts"), prior.get("facts")),
            prior.get("facts", []),
            MEMORY_MAX_ITEMS,
        ),
        "outputs": _merge_output_lists(
            _normalize_outputs(candidate.get("outputs"), prior.get("outputs")),
            prior.get("outputs", []),
            MEMORY_OUTPUTS_MAX_ITEMS,
        ),
        "decisions": _merge_str_lists(
            _coerce_str_list(candidate.get("decisions"), prior.get("decisions")),
            prior.get("decisions", []),
            MEMORY_MAX_ITEMS,
        ),
        "open_questions": _merge_str_lists(
            _coerce_str_list(candidate.get("open_questions"), prior.get("open_questions")),
            prior.get("open_questions", []),
            MEMORY_MAX_ITEMS,
        ),
    }


def _format_memory_for_prompt(memory: Optional[Dict[str, Any]]) -> str:
    if not isinstance(memory, dict):
        return ""
    lines: List[str] = []
    if memory.get("facts"):
        lines.append("Facts:")
        lines.extend(f"- {item}" for item in memory["facts"] if str(item).strip())
    if memory.get("outputs"):
        lines.append("Outputs:")
        for item in memory["outputs"]:
            path = str(item.get("path", "")).strip()
            description = str(item.get("description", "")).strip()
            if path and description:
                lines.append(f"- {path} | {description}")
            elif path or description:
                lines.append(f"- {path or description}")
    if memory.get("decisions"):
        lines.append("Decisions:")
        lines.extend(f"- {item}" for item in memory["decisions"] if str(item).strip())
    if memory.get("open_questions"):
        lines.append("Open questions:")
        lines.extend(f"- {item}" for item in memory["open_questions"] if str(item).strip())
    return "\n".join(lines).strip()


# --- Episodic memory (planning agent only) ------------------------------------

_episodic_cache: Dict[str, Tuple[float, str]] = {}
# Set once the store proves unavailable (no API key, no vector store). Constructing
# the orchestrator opens ChromaDB and a chat model, so retrying it before every
# planning call would cost more than the examples are worth.
_episodic_unavailable = False


def episodic_examples_block(user_request: str) -> str:
    '''Similar past tasks and how they were decomposed.

    Injected here rather than baked into the planner's static prompt. That is what
    lets the compiled graph be cached: the previous build rebuilt the entire app —
    five chat models and five agents — on every user message purely so this text
    could change.

    Parameters:
    ---------
    user_request (str): the request to find precedent for.

    Returns:
    ----------
    block (str): similar past tasks and how they were decomposed, injected here so the compiled graph can stay cached.
    '''

    request = (user_request or "").strip()
    if not request:
        return ""

    key = request[:400]
    now = time.monotonic()
    cached = _episodic_cache.get(key)
    if cached is not None and now - cached[0] < EPISODIC_CACHE_TTL_SECONDS:
        return cached[1]

    global _episodic_unavailable
    if _episodic_unavailable:
        return ""

    try:
        from persistence.memory.episodic_memory.episodic_learning import get_orchestrator

        examples = get_orchestrator().episodic_system.get_relevant_examples(request)
    except Exception as exc:  # noqa: BLE001 - planning must survive a cold store
        logger.warning("Episodic examples unavailable, disabling for this process: %s", exc)
        _episodic_unavailable = True
        _episodic_cache[key] = (now, "")
        return ""

    if not examples:
        _episodic_cache[key] = (now, "")
        return ""

    blocks: List[str] = [
        "Similar tasks this system has planned before, and what was learned. Treat "
        "them as precedent for shaping the breakdown, not as facts about the current "
        "request:"
    ]
    for index, example in enumerate(examples, start=1):
        blocks.append(f"\nExample {index}")
        blocks.append(f"  Task: {_shorten(str(example.get('task', '')), 600)}")
        decomposition = example.get("final_decomposition") or example.get("initial_decomposition") or ""
        blocks.append(f"  Decomposition: {_shorten(str(decomposition), 900)}")
        notes = example.get("notes")
        if notes:
            blocks.append(f"  Lesson learned: {_shorten(str(notes), 600)}")

    rendered = "\n".join(blocks)
    _episodic_cache[key] = (now, rendered)
    return rendered


def clear_episodic_cache() -> None:
    '''Forget cached examples, and give a previously unavailable store another try.'''

    global _episodic_unavailable
    _episodic_cache.clear()
    _episodic_unavailable = False


# --- The pinned block ---------------------------------------------------------


def _conversation_goal(messages: Sequence[BaseMessage]) -> str:
    starts = _turn_start_indices(messages)
    if not starts:
        return ""
    return coerce_text(getattr(messages[starts[0]], "content", None))


def build_pinned_context_block(
    state: Dict[str, Any],
    messages: Sequence[BaseMessage],
    *,
    include_goal: bool = True,
    include_plan: bool = True,
    include_artifacts: bool = True,
) -> str:
    '''The part of context that must never be summarized away.

    Parameters:
    ---------
    state (dict): graph state, for the goal, the plan pointer and the approval constraints.
    messages (list): the transcript, for the current request.
    include_goal (boolean): whether the conversation goal belongs in this role's block.
    include_plan (boolean): whether the plan path and progress line belong in it.
    include_artifacts (boolean): whether the output scope and artifact ledger belong in it.

    Returns:
    ----------
    block (str): the part of context that must survive summarization.
    '''

    conversation_id = state.get("conversation_id")

    sections: List[str] = [describe_output_scope(conversation_id=conversation_id)]

    if include_goal:
        goal = _conversation_goal(messages)
        if goal:
            sections.append(
                "Original request for this conversation (the standing goal):\n"
                + _shorten(goal, CONTEXT_GOAL_MAX_CHARS)
            )

    # The plan text itself stays in plan.md. Only a pointer and a progress line are
    # pinned; `plan_status` returns the authoritative copy on demand.
    if include_plan:
        plan_path = coerce_text(state.get("plan_path"))
        if plan_path:
            lines = [f"Execution plan for this conversation: {plan_path}"]
            progress = coerce_text(state.get("plan_progress"))
            if progress:
                lines.append(progress)
            lines.append(
                "That file is the authoritative record of what has been done. Read it "
                "with `plan_status` and record outcomes with `plan_update`; never "
                "restate progress from memory."
            )
            sections.append("\n".join(lines))

    constraints = [f"- {item}" for item in (state.get("approval_constraints") or []) if str(item).strip()]
    if constraints:
        sections.append(
            "Conditions the user attached when approving the plan. They override the "
            "corresponding plan steps:\n" + "\n".join(constraints)
        )

    if include_artifacts:
        artifacts = describe_output_artifacts(
            conversation_id=conversation_id,
            max_items=CONTEXT_ARTIFACT_MAX_ITEMS,
        )
        if artifacts:
            sections.append(
                "Files already produced in this conversation. Reuse them instead of "
                "regenerating equivalent work, and read them when you need their "
                "contents:\n" + artifacts
            )

    return "\n\n".join(sections)


# --- Per-role views of the transcript ------------------------------------------
#
# The specialists are **context-isolated**: a specialist sees the brief the supervisor
# wrote for it and its own working messages, and nothing else. Only the supervisor
# holds the conversation.
#
# Why: a specialist given the whole transcript spends its context on work it is not
# doing, and picks up other agents' half-finished reasoning as if it were input — a
# prediction agent that has read the planner's draft starts second-guessing which
# endpoints to run. Isolation also makes each delegation reproducible: the brief is
# the entire input, so a bad result is a bad brief, which is a thing that can be fixed.
#
# The cost is that a thin brief now fails visibly instead of being silently rescued by
# ambient context, which is why the handoff tool asks for objective, inputs and
# expected output separately.


def _specialist_span_start(messages: Sequence[BaseMessage], agent_name: str) -> int:
    '''Index of the handoff `ToolMessage` that most recently briefed this agent.

    Parameters:
    ---------
    messages (list): the whole transcript.
    agent_name (str): the specialist whose span to locate.

    Returns:
    ----------
    index (int): position of the handoff `ToolMessage` that most recently briefed it, or -1.
    '''

    target = f"transfer_to_{agent_name}"
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if isinstance(message, ToolMessage) and str(getattr(message, "name", "")) == target:
            return index
    return -1


def build_specialist_input_messages(
    state: Dict[str, Any],
    *,
    agent_name: str,
) -> List[BaseMessage]:
    '''The brief, then this specialist's own working messages. Nothing else.

    The working window starts *after* the handoff `ToolMessage`, which also means the
    supervisor's `AIMessage` carrying the handoff `tool_call` is excluded — so the
    window never opens with an orphaned tool call.

    Falls back to the shared view when there is no brief, which happens if a
    specialist is ever invoked outside a handoff; degrading to more context is safer
    than handing a model an empty prompt.

    Parameters:
    ---------
    state (dict): graph state, holding the transcript to slice.
    agent_name (str): the specialist being run.

    Returns:
    ----------
    messages (list): its brief followed by its own working messages — no conversation, no plan, no other specialist's work.
    '''

    messages = list(state.get("messages") or [])
    if not messages:
        return []

    start = _specialist_span_start(messages, agent_name)
    if start < 0:
        logger.debug("No handoff brief found for %s; falling back to the shared view", agent_name)
        return build_llm_input_messages(state)

    brief = coerce_text(getattr(messages[start], "content", None))
    working = prune_tool_traffic(messages[start + 1 :])
    return repair_tool_pairing([HumanMessage(content=brief), *working])


def collapse_specialist_spans(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
    '''Replace each specialist's working span with the result it reported.

    The supervisor reasons about outcomes, not about how a specialist got there. Its
    tool traffic — a knowledge-graph traversal returns tens of thousands of characters
    — would otherwise dominate the one context that has to hold the whole workflow.

    A span runs from a handoff `ToolMessage` until the supervisor speaks again. Only
    the specialist's final message survives it, which is exactly what its prompt asks
    it to make self-contained. A span with no such message is kept intact rather than
    silently emptied.

    Parameters:
    ---------
    messages (list): the whole transcript.

    Returns:
    ----------
    collapsed (list): each specialist's working span replaced by the result it reported, which is what the supervisor sees instead of tool traffic.
    '''

    kept: List[BaseMessage] = []
    index = 0
    total = len(messages)

    while index < total:
        message = messages[index]
        kept.append(message)

        is_handoff = isinstance(message, ToolMessage) and str(
            getattr(message, "name", "")
        ).startswith("transfer_to_")
        if not is_handoff:
            index += 1
            continue

        # Collect the span this handoff opened.
        index += 1
        span: List[BaseMessage] = []
        while index < total:
            candidate = messages[index]
            name = str(getattr(candidate, "name", "") or "")
            if name == SUPERVISOR_AGENT_NAME:
                break
            if isinstance(candidate, ToolMessage) and name.startswith("transfer_to_"):
                break
            span.append(candidate)
            index += 1

        report = next(
            (
                item
                for item in reversed(span)
                if isinstance(item, AIMessage) and not getattr(item, "tool_calls", None)
            ),
            None,
        )
        if report is not None:
            kept.append(report)
        else:
            # No summarizing message: keep the span rather than drop the evidence.
            kept.extend(span)

    return kept


def build_llm_input_messages(
    state: Dict[str, Any],
    *,
    compress: bool = True,
    collapse_delegations: bool = False,
) -> List[BaseMessage]:
    '''The message list one agent should actually be given.

    Parameters:
    ---------
    state (dict): graph state, holding the transcript and the summary.
    compress (boolean): whether to fold older turns into the carried summary.
    collapse_delegations (boolean): whether to replace specialist spans with their reported results, as the supervisor needs.

    Returns:
    ----------
    messages (list): pinned block, summary, the last completed turns verbatim, then the live turn with tool traffic bounded in place.
    '''

    messages = list(state.get("messages") or [])
    if not messages:
        return []

    if collapse_delegations:
        messages = collapse_specialist_spans(messages)

    if not compress:
        visible = [message for message in messages if not is_summary_message(message)]
        return repair_tool_pairing(prune_tool_traffic(visible))

    index, summary_text, memory = latest_summary_record(messages)
    recent = messages_since_last_summary(messages) if index >= 0 else list(messages)
    anchors = build_turn_anchors(messages, upto_index=index + 1 if index >= 0 else 0)

    prefix: List[BaseMessage] = []
    if summary_text:
        prefix.append(
            SystemMessage(
                content=(
                    "Compressed summary of earlier work in this conversation. Where it "
                    "overlaps with the verbatim exchanges that follow, those are more "
                    "reliable:\n" + summary_text
                )
            )
        )
    memory_text = _format_memory_for_prompt(memory)
    if memory_text:
        prefix.append(SystemMessage(content="Structured memory:\n" + memory_text))

    return prefix + anchors + repair_tool_pairing(prune_tool_traffic(recent))


# --- The middleware -----------------------------------------------------------


ISOLATION_NOTICE = """# WHAT YOU CAN SEE, AND WHAT IS YOURS TO DECIDE

You have been given one task brief and nothing else — not the conversation, not the plan,
not what any other agent has done. That is deliberate: the brief is your whole input, and
the supervisor is responsible for making it complete.

**The method is yours.** The brief states the outcome, the inputs and the constraints, not
the tool calls to make. You know these tools, their argument shapes and how they fail;
choose the approach, and change it if the first attempt tells you something new. If the
brief does prescribe a method that you can see is wrong for the data in front of you, do
the thing that achieves the stated outcome and say in your reply what you did instead.

- Work from the brief. Do not infer a wider goal from it and do not expand the scope.
- The brief's file list is the step; the artifact list in your context is background — it
  shows what exists in this conversation, which is useful for avoiding work that has
  already been done, but it says nothing about what any of those files contain.
- If the brief is missing something you need — an identifier, a file path, a threshold —
  say so plainly in your reply and do what you can without it. Never invent the missing
  value and never substitute a plausible-looking one.
- Make your reply self-contained. It is the only thing the supervisor will read back: what
  you did, the values and counts that matter, the full path of every file you produced,
  and anything that failed or is uncertain."""


# --- Bounding what a tool writes into the conversation -------------------------


def _bound_recorded_text(text: str) -> str:
    '''Cut an oversized tool result down to what may be *stored*.

    Parameters:
    ---------
    text (str): the tool result as the tool produced it.

    Returns:
    ----------
    text (str): the same text when it fits, otherwise its head and tail with a note in between saying how much was dropped and where the full data lives.
    '''

    if len(text) <= TOOL_RESULT_PERSIST_MAX_CHARS:
        return text
    head = int(TOOL_RESULT_PERSIST_MAX_CHARS * 0.6)
    tail = TOOL_RESULT_PERSIST_MAX_CHARS - head
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n"
        f"... [{omitted:,} characters were dropped here before this result was recorded: "
        f"it exceeded the {TOOL_RESULT_PERSIST_MAX_CHARS:,}-character ceiling on a single "
        "tool result. Write bulk data to a file in the output scope and read back the part "
        "you need.] ...\n\n"
        f"{text[-tail:]}"
    )


@wrap_tool_call(name="bound_tool_result")
async def bound_tool_result(request, handler):
    '''Cap every tool result at the point it becomes conversation state.

    The compression above bounds what a model is *shown*; this bounds what is
    *kept*. The distinction is not academic. A view is rebuilt per model call and
    costs nothing to discard, while a `ToolMessage` joins the `messages` channel —
    and LangGraph's SQLite checkpointer serialises the entire state into one blob
    per superstep, with no per-channel table like its PostgreSQL counterpart. So a
    single oversized result is not stored once, it is stored once *per step*: one
    run that ended a `python_executor` cell on a 95 798-row DataFrame produced two
    143 MB results and cost **20.6 GB** across 126 checkpoints.

    Every tool goes through here, which is the point — `python_executor` guards its
    own output too, but a per-tool cap is one a new tool can be written without.

    It **must** be async, for the same reason `build_context_middleware` is: a
    sync-only hook raises `NotImplementedError: Asynchronous implementation of
    awrap_tool_call is not available` the moment the agent is driven by `astream`,
    and every run here is streamed. The failure surfaces as a tool error inside the
    run, not as an import-time mistake.

    Parameters:
    ---------
    request (ToolCallRequest): the call about to be executed.
    handler (callable): runs the tool and returns its `ToolMessage` or `Command`.

    Returns:
    ----------
    response (ToolMessage | Command): the tool's response, with oversized text content replaced by its head and tail.
    '''

    response = await handler(request)
    content = getattr(response, "content", None)
    if isinstance(content, str) and len(content) > TOOL_RESULT_PERSIST_MAX_CHARS:
        bounded = _bound_recorded_text(content)
        logger.warning(
            "Tool %s returned %s characters; recorded %s.",
            request.tool_call.get("name", "?") if hasattr(request, "tool_call") else "?",
            f"{len(content):,}",
            f"{len(bounded):,}",
        )
        # `.model_copy` keeps the id and tool_call_id, which pairing depends on.
        return response.model_copy(update={"content": bounded})
    return response


def build_context_middleware(
    *,
    role: str,
    compress: bool = True,
    posture_block: Optional[Callable[[Dict[str, Any]], str]] = None,
    include_plan: bool = True,
    include_artifacts: bool = True,
    include_episodes: bool = False,
    include_goal: bool = True,
    isolated: bool = False,
    collapse_delegations: bool = False,
):
    '''Async `wrap_model_call` middleware for one agent role.

    Overrides the system message (role prompt + posture + pinned block) and the message
    list. Must be async: `astream` refuses a sync-only `wrap_model_call`, and every run
    here is streamed.

    `isolated=True` gives a specialist only the brief it was handed plus its own working
    messages. `collapse_delegations=True` gives the supervisor each specialist's
    reported result in place of its tool traffic.

    Parameters:
    ---------
    role (str): which prompt and view this agent gets.
    compress (boolean): whether older turns are folded into the carried summary.
    posture_block (Callable[[Dict[str, Any]], str]): callable returning the supervisor posture for `state['task_category']`.
    include_plan (boolean): whether to pin the plan path and progress line.
    include_artifacts (boolean): whether to pin the output scope and artifact ledger.
    include_episodes (boolean): whether to inject episodic precedent, which only the planner wants.
    include_goal (boolean): whether to pin the conversation goal.
    isolated (boolean): build a specialist's view: its brief and its own work only.
    collapse_delegations (boolean): build the supervisor's view: specialist results in place of their tool traffic.

    Returns:
    ----------
    middleware (callable): an async `@wrap_model_call` overriding `system_message` and `messages`. It must be async — `astream` raises `NotImplementedError` on a sync-only hook, and every run here is streamed.
    '''

    @wrap_model_call(name=f"{role}_context")
    async def middleware(request, handler):
        state = request.state or {}
        messages = list(state.get("messages") or [])

        sections: List[str] = []
        base = getattr(request.system_message, "content", "") or ""
        if base:
            sections.append(base)

        if posture_block is not None:
            posture = posture_block(state)
            if posture:
                sections.append(posture)

        if include_episodes and state.get("use_episodic_learning", True):
            episodes = episodic_examples_block(latest_user_request_text(messages))
            if episodes:
                sections.append(episodes)

        if isolated:
            sections.append(ISOLATION_NOTICE)

        # A specialist still needs its operating scope and the artifact ledger: it has
        # to know where to write, and the brief refers to files by path. It does not
        # get the conversation goal or the plan — that framing belongs in the brief,
        # and leaving it out is what stops a thin brief from being quietly rescued.
        pinned = build_pinned_context_block(
            state,
            messages,
            include_goal=include_goal,
            include_plan=include_plan,
            include_artifacts=include_artifacts,
        )
        if pinned:
            sections.append(pinned)

        if isolated:
            payload = build_specialist_input_messages(state, agent_name=role)
        else:
            payload = build_llm_input_messages(
                state, compress=compress, collapse_delegations=collapse_delegations
            )

        return await handler(
            request.override(
                system_message=SystemMessage(content="\n\n".join(sections)),
                messages=payload,
            )
        )

    return middleware
