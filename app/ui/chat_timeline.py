from __future__ import annotations

import time
from copy import deepcopy
from html import escape
from typing import Any, Dict, Iterable, List, Optional, Set

import gradio as gr
from gradio.components.chatbot import ChatMessage

try:
    from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover - optional
    MarkdownIt = None

from app.state import UIState
from app.ui import tool_display
from app.ui.formatters import derive_message_id

AGENT_TITLES = {
    "task_classifier": "Routing",
    "planning_agent": "Planning Agent",
    "human_chat": "Plan review",
    "approval_ack": "Plan Approved",
    "plan_init": "Execution Plan",
    "plan_finalize": "Execution Plan",
    "supervisor": "Supervisor",
    "execution": "Supervisor",
    "research_agent": "Research Agent",
    "prediction_agent": "Prediction Agent",
    "data_agent": "Data Agent",
    "report_agent_complex": "Report",
    "report_agent_simple": "Report",
    "report_agent_meta": "Response",
    "context_summary": "Context Summary",
    "context_summary_complex": "Context Summary",
    "context_summary_simple": "Context Summary",
    "context_summary_meta": "Context Summary",
}

# Nodes whose output never belongs in the transcript. `plan_init` / `plan_finalize`
# write the ledger, which the plan panel renders live from the file — a second copy
# in the transcript would only compete with it.
IGNORED_NODES = {
    "task_classifier",
    "human_chat",
    "plan_init",
    "plan_finalize",
    "context_summary",
    "context_summary_complex",
    "context_summary_simple",
    "context_summary_meta",
    "__start__",
    "__end__",
}

# The deliverable. Rendered as a document so the answer does not look like one more
# grey tool box. The report prompts emit a fixed heading structure and the CSS is
# built around exactly those headings — keep the two in step.
REPORT_NODES = {"report_agent_complex", "report_agent_simple", "report_agent_meta"}

# The plan under review, styled so what the user must approve is legible at a glance.
PLAN_NODES = {"planning_agent"}

TIMELINE_SNAPSHOT_VERSION = 1

# `table` is enabled explicitly: it is a GFM extension, not CommonMark, so the
# strict preset dropped every report table on the floor and rendered the pipes
# verbatim — while `.agent-message-section--report table` sat in the stylesheet
# never matching anything. `breaks` keeps a plan's one-line-per-field layout.
_MARKDOWN = (
    MarkdownIt("commonmark", {"breaks": True, "html": False}).enable("table")
    if MarkdownIt is not None
    else None
)


# --- Timeline lifecycle -------------------------------------------------------


def reset_chat_messages(state: UIState) -> None:
    state.messages = []
    state.message_lookup = {}
    state.agent_blocks = {}
    state.tool_call_block_lookup = {}
    state.streaming_message_lookup = {}
    state.last_agent_block_id = None
    state.message_seq = 0


def append_user_message(state: UIState, content: str) -> ChatMessage:
    message = ChatMessage(role="user", content=content)
    state.messages.append(message)
    state.last_agent_block_id = None
    return message


def rebuild_from_plain_messages(
    state: UIState,
    messages: Iterable[Dict[str, str]],
    *,
    skip_texts: Optional[Set[str]] = None,
) -> None:
    '''Fallback for conversations stored before structured snapshots existed.

    Parameters:
    ---------
    state (UIState): the state to rebuild.
    messages (list): the stored plain messages.
    skip_texts (set): message texts already rendered, which must not appear twice.
    '''

    reset_chat_messages(state)
    for message in messages:
        role = message.get("role")
        content = (message.get("content") or "").strip()
        if not content:
            continue
        if role == "user":
            append_user_message(state, content)
            continue
        if skip_texts and content in skip_texts:
            continue
        block = _ensure_agent_block(state, "assistant")
        block["items"].append({"type": "message", "content": content})
        _refresh_block_message(state, block["block_id"])
    finalize_active_blocks(state)


def export_timeline_snapshot(state: UIState) -> Dict[str, Any]:
    '''Serialize the rendered timeline for persistence.

    Parameters:
    ---------
    state (UIState): the state to serialize.

    Returns:
    ----------
    snapshot (dict): the rendered timeline, stored in `user_threads.ui_timeline`.
    '''

    entries: List[Dict[str, Any]] = []
    for message in state.messages:
        if message.role == "user":
            entries.append({"kind": "user", "content": str(message.content or "")})
            continue

        metadata = deepcopy(message.metadata) if isinstance(message.metadata, dict) else {}
        block_id = metadata.get("id")
        block = state.agent_blocks.get(block_id) if block_id else None
        if block:
            entries.append(
                {
                    "kind": "agent_block",
                    "block_id": block["block_id"],
                    "agent_name": block["agent_name"],
                    "metadata": metadata
                    or _build_metadata(block["agent_name"], block["block_id"], status="done"),
                    "items": deepcopy(block["items"]),
                }
            )
            continue
        entries.append(
            {
                "kind": "assistant_plain",
                "content": str(message.content or ""),
                "metadata": metadata,
            }
        )

    return {
        "version": TIMELINE_SNAPSHOT_VERSION,
        "message_seq": state.message_seq,
        "entries": entries,
    }


def rebuild_from_timeline_snapshot(state: UIState, snapshot: Dict[str, Any]) -> bool:
    '''Rebuild the UI from a persisted snapshot of rendered blocks.

    Parameters:
    ---------
    state (UIState): the state to rebuild in place.
    snapshot (dict): a persisted snapshot of rendered blocks.

    Returns:
    ----------
    rebuilt (boolean): True when the snapshot was usable.
    '''

    if not isinstance(snapshot, dict):
        return False
    entries = snapshot.get("entries")
    if not isinstance(entries, list):
        return False

    reset_chat_messages(state)
    max_seq = 0

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = entry.get("kind")

        if kind == "user":
            content = str(entry.get("content", "")).strip()
            if content:
                append_user_message(state, content)
            continue

        if kind == "assistant_plain":
            metadata = deepcopy(entry.get("metadata")) if isinstance(entry.get("metadata"), dict) else {}
            if metadata.get("status") == "pending":
                metadata["status"] = "done"
            block_id = metadata.get("id")
            state.messages.append(
                ChatMessage(
                    role="assistant",
                    content=str(entry.get("content", "")),
                    metadata=metadata or None,
                )
            )
            if block_id:
                state.message_lookup[str(block_id)] = len(state.messages) - 1
                max_seq = max(max_seq, _extract_message_seq(block_id))
            state.last_agent_block_id = None
            continue

        if kind != "agent_block":
            continue

        agent_name = str(entry.get("agent_name") or "assistant").lower()
        block_id = str(entry.get("block_id") or state.next_message_id(agent_name))
        metadata = deepcopy(entry.get("metadata")) if isinstance(entry.get("metadata"), dict) else {}
        metadata.setdefault("id", block_id)
        # A block that was still streaming when the snapshot was taken must not come
        # back with a live spinner: nothing is going to resolve it.
        if metadata.get("status") == "pending":
            metadata["status"] = "done"

        state.messages.append(ChatMessage(role="assistant", content="", metadata=metadata))
        state.message_lookup[block_id] = len(state.messages) - 1

        items = deepcopy(entry.get("items")) if isinstance(entry.get("items"), list) else []
        block = {"agent_name": agent_name, "block_id": block_id, "items": items}
        state.agent_blocks[block_id] = block
        state.last_agent_block_id = block_id
        _restore_tool_lookup_for_block(state, block)
        _refresh_block_message(state, block_id)
        max_seq = max(max_seq, _extract_message_seq(block_id))

    stored_seq = snapshot.get("message_seq")
    state.message_seq = max(max_seq, stored_seq if isinstance(stored_seq, int) else 0)
    return bool(entries)


# --- Ingesting streamed events -----------------------------------------------


def process_chunk(state: UIState, chunk: Dict[str, Any]) -> bool:
    '''Apply one streamed payload. True when the timeline changed.

    Parameters:
    ---------
    state (UIState): the state to fold the payload into.
    chunk (dict): one streamed payload.

    Returns:
    ----------
    changed (boolean): True when the timeline changed.
    '''

    updated = False
    for agent_name, payload in (chunk or {}).items():
        if not isinstance(payload, dict):
            continue
        messages = payload.get("messages") or []
        if str(agent_name).lower() in IGNORED_NODES:
            _suppress_messages(state, messages)
            continue
        for message in messages:
            if _ingest_message(state, message, agent_name=agent_name):
                updated = True
    return updated


def _suppress_messages(state: UIState, messages: Any) -> None:
    '''Drop an ignored node's messages permanently.

    Recording the ids is the whole point: a wrapping node re-emits them later under
    a name that is not ignored, and they would render there instead.

    Parameters:
    ---------
    state (UIState): the state recording the suppression.
    messages (Any): the messages of a node that must not render.
    '''

    for message in messages or []:
        message_id = derive_message_id(message)
        if message_id:
            state.processed_message_ids.add(message_id)
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            state.processed_tools_ids.add(tool_call_id)


def _ingest_message(state: UIState, raw: Any, agent_name: Optional[str]) -> bool:
    role = _get_role(raw)
    if _is_stream_chunk(raw, role):
        return _append_streaming_text(
            state, agent_name or getattr(raw, "name", None) or "assistant", raw
        )
    if role in {"human", "user"}:
        # The user's own message is already in the timeline; mark it seen so the
        # aggregate re-emission does not add a duplicate bubble.
        message_id = derive_message_id(raw)
        if message_id:
            state.processed_message_ids.add(message_id)
        return False
    if role in {"ai", "assistant"}:
        return _ingest_ai_message(state, raw, agent_name)
    if role in {"tool", "function"}:
        return _ingest_tool_result(state, raw)
    return False


def _ingest_ai_message(state: UIState, raw: Any, agent_name: Optional[str]) -> bool:
    agent_key = str(agent_name or getattr(raw, "name", None) or "assistant").lower()
    message_id = derive_message_id(raw) or state.next_message_id(agent_key)
    if message_id in state.processed_message_ids:
        return False

    block = _ensure_agent_block(state, agent_key)
    updated = False

    text = _coerce_text(getattr(raw, "content", None))
    primary_key = str(message_id)
    fallback_key = f"{agent_key}:{block['block_id']}:stream"
    stream_entry = state.streaming_message_lookup.get(
        primary_key
    ) or state.streaming_message_lookup.get(fallback_key)
    if stream_entry and primary_key not in state.streaming_message_lookup:
        state.streaming_message_lookup[primary_key] = stream_entry
        state.streaming_message_lookup.pop(fallback_key, None)

    if text:
        # Replace the accumulated streamed text rather than appending to it, so a
        # partially flushed buffer self-corrects when the message completes.
        streamed_index = (
            stream_entry.get("item_index")
            if stream_entry and stream_entry.get("block_id") == block["block_id"]
            else None
        )
        if streamed_index is not None and streamed_index < len(block["items"]):
            block["items"][streamed_index]["content"] = text
        else:
            block["items"].append({"type": "message", "content": text})
            state.streaming_message_lookup[primary_key] = {
                "block_id": block["block_id"],
                "item_index": len(block["items"]) - 1,
            }
        updated = True

    for call in getattr(raw, "tool_calls", None) or []:
        updated |= _append_tool_call(state, block, call)

    if updated:
        _refresh_block_message(state, block["block_id"])
    state.processed_message_ids.add(message_id)
    return updated


def _append_tool_call(state: UIState, block: Dict[str, Any], call: Any) -> bool:
    name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else "tool")
    args = getattr(call, "args", None) or (call.get("args") if isinstance(call, dict) else {})
    call_id = getattr(call, "id", None) or (
        call.get("id") if isinstance(call, dict) else state.next_message_id("tool_call")
    )
    if name in tool_display.SUPPRESSED_TOOLS:
        # Remember the id so the matching result is dropped too.
        state.processed_tools_ids.add(str(call_id))
        return False

    key = str(call_id)
    entry = tool_display.call_metadata(name, args)
    for item in block["items"]:
        if item.get("type") == "tool_call" and item.get("id") == key:
            # Seen twice (streamed, then committed): keep whatever the result recorded.
            item.update({k: v for k, v in entry.items() if k not in ("status", "note", "result_body")})
            state.tool_call_block_lookup[key] = block["block_id"]
            return True

    block["items"].append({"type": "tool_call", "id": key, "tool_name": name, **entry})
    state.tool_call_block_lookup[key] = block["block_id"]
    return True


def _ingest_tool_result(state: UIState, raw: Any) -> bool:
    message_id = derive_message_id(raw) or state.next_message_id("tool_result")
    if message_id in state.processed_message_ids:
        return False

    tool_call_id = getattr(raw, "tool_call_id", None)
    if tool_call_id and str(tool_call_id) in state.processed_tools_ids:
        state.processed_message_ids.add(message_id)
        return False

    tool_name = getattr(raw, "name", None) or "tool"
    if tool_name in tool_display.SUPPRESSED_TOOLS:
        state.processed_message_ids.add(message_id)
        if tool_call_id:
            state.tool_call_block_lookup.pop(str(tool_call_id), None)
            state.processed_tools_ids.add(str(tool_call_id))
        return False

    # Attribution is by tool_call_id, not by the emitting node: a handoff's result
    # is committed by the subgraph, but it belongs on the supervisor's call.
    block_id = state.tool_call_block_lookup.get(str(tool_call_id)) or state.last_agent_block_id
    block = state.agent_blocks.get(block_id) if block_id else None
    if not block:
        state.processed_message_ids.add(message_id)
        return False

    content = getattr(raw, "content", None)
    status, note = tool_display.describe_result(tool_name, content)
    body = tool_display.render_result_body(tool_name, content)

    # Fold the outcome into the call it answers, so one action reads as one line.
    merged = False
    if tool_call_id:
        for item in block["items"]:
            if item.get("type") == "tool_call" and item.get("id") == str(tool_call_id):
                item["status"] = status
                item["note"] = note
                item["result_body"] = body
                merged = True
                break

    if not merged:
        view = tool_display.describe_call(tool_name, None)
        block["items"].append(
            {
                "type": "tool_call",
                "id": str(tool_call_id) if tool_call_id else None,
                "tool_name": tool_name,
                "label": view.label,
                "status": status,
                "note": note,
                "call_body": "",
                "result_body": body,
            }
        )

    if tool_call_id:
        state.tool_call_block_lookup.pop(str(tool_call_id), None)
        state.processed_tools_ids.add(str(tool_call_id))
    _refresh_block_message(state, block["block_id"])
    state.processed_message_ids.add(message_id)
    return True


def _append_streaming_text(state: UIState, agent_name: str, chunk: Any) -> bool:
    agent_key = str(agent_name or "assistant").lower()
    text = _coerce_stream_text(getattr(chunk, "content", None))
    if not text:
        return False

    message_id = getattr(chunk, "id", None)
    if isinstance(chunk, dict):
        message_id = message_id or chunk.get("id")

    block = _ensure_agent_block(state, agent_key)
    lookup_key = str(message_id) if message_id else f"{agent_key}:{block['block_id']}:stream"
    entry = state.streaming_message_lookup.get(lookup_key)

    if entry and entry.get("block_id") == block["block_id"]:
        index = entry.get("item_index")
        if index is not None and index < len(block["items"]):
            block["items"][index]["content"] += text
        else:
            block["items"].append({"type": "message", "content": text})
            entry = {"block_id": block["block_id"], "item_index": len(block["items"]) - 1}
            state.streaming_message_lookup[lookup_key] = entry
    else:
        block["items"].append({"type": "message", "content": text})
        entry = {"block_id": block["block_id"], "item_index": len(block["items"]) - 1}
        state.streaming_message_lookup[lookup_key] = entry

    state.streaming_message_lookup[f"{agent_key}:{block['block_id']}:stream"] = entry
    _refresh_block_message(state, block["block_id"])
    return True


# --- Blocks ------------------------------------------------------------------


def _ensure_agent_block(state: UIState, agent_key: str) -> Dict[str, Any]:
    last_id = state.last_agent_block_id
    if last_id:
        block = state.agent_blocks.get(last_id)
        if block and block["agent_name"] == agent_key:
            return block

    # A different agent is taking over: resolve the previous block's spinner and
    # stamp how long it ran.
    _finalize_block(state, last_id)

    block_id = state.next_message_id(agent_key)
    state.messages.append(
        ChatMessage(
            role="assistant", content="", metadata=_build_metadata(agent_key, block_id, status="pending")
        )
    )
    state.message_lookup[block_id] = len(state.messages) - 1
    block = {
        "agent_name": agent_key,
        "block_id": block_id,
        "items": [],
        "started_at": time.time(),
    }
    state.agent_blocks[block_id] = block
    state.last_agent_block_id = block_id
    return block


def _set_block_metadata(state: UIState, block_id: str, **updates: Any) -> None:
    index = state.message_lookup.get(block_id)
    if index is None or index >= len(state.messages):
        return
    message = state.messages[index]
    metadata = dict(message.metadata) if isinstance(message.metadata, dict) else {}
    metadata.update(updates)
    message.metadata = metadata


def _finalize_block(state: UIState, block_id: Optional[str], *, status: str = "done") -> None:
    if not block_id:
        return
    index = state.message_lookup.get(block_id)
    if index is None or index >= len(state.messages):
        return
    current = state.messages[index].metadata if isinstance(state.messages[index].metadata, dict) else {}
    if current.get("status") == status and "duration" in current:
        return
    updates: Dict[str, Any] = {"status": status}
    block = state.agent_blocks.get(block_id)
    started_at = block.get("started_at") if block else None
    if started_at:
        updates["duration"] = round(max(0.0, time.time() - started_at), 1)
    _set_block_metadata(state, block_id, **updates)


def finalize_active_blocks(state: UIState, *, status: str = "done") -> None:
    '''Resolve the spinner on the block that was still streaming.

    Parameters:
    ---------
    state (UIState): the state whose streaming block to resolve.
    status (str): how the block ended, `done` or a failure.
    '''

    _finalize_block(state, state.last_agent_block_id, status=status)


def _append_card(
    state: UIState,
    *,
    kind: str,
    title: str,
    message: str,
    detail: Optional[str] = None,
) -> bool:
    finalize_active_blocks(state)
    block_id = state.next_message_id(kind)
    state.messages.append(
        ChatMessage(role="assistant", content="", metadata={"title": title, "id": block_id, "status": "done"})
    )
    state.message_lookup[block_id] = len(state.messages) - 1
    state.agent_blocks[block_id] = {
        "agent_name": kind,
        "block_id": block_id,
        "items": [{"type": kind, "title": title, "message": message, "detail": detail}],
    }
    # Reset the pointer so later content opens a fresh agent block.
    state.last_agent_block_id = None
    _refresh_block_message(state, block_id)
    return True


def append_error_block(
    state: UIState,
    message: str,
    *,
    title: str = "Run interrupted",
    detail: Optional[str] = None,
) -> bool:
    return _append_card(state, kind="error", title=title, message=message, detail=detail)


def append_notice_block(state: UIState, message: str, *, title: str = "Notice") -> bool:
    '''A neutral status card: a stopped run is not a failure.

    Parameters:
    ---------
    state (UIState): the state to append to.
    message (str): what the notice says.
    title (str): its heading.

    Returns:
    ----------
    changed (boolean): True when the notice was added. A neutral card, because a stopped run is not a failure.
    '''

    return _append_card(state, kind="notice", title=title, message=message)


def _restore_tool_lookup_for_block(state: UIState, block: Dict[str, Any]) -> None:
    '''Re-register calls still awaiting a result after a reload.

    Parameters:
    ---------
    state (UIState): the state whose lookup to repopulate.
    block (dict): one restored timeline block.
    '''

    for item in block["items"]:
        if item.get("type") != "tool_call":
            continue
        call_id = item.get("id")
        if not call_id or item.get("status") in ("ok", "error"):
            continue
        state.tool_call_block_lookup[str(call_id)] = block["block_id"]


def _refresh_block_message(state: UIState, block_id: str) -> None:
    block = state.agent_blocks.get(block_id)
    if not block:
        return
    index = state.message_lookup.get(block_id)
    if index is None or index >= len(state.messages):
        return
    # Always HTML. Mixing markdown and HTML rendering meant a block's typography
    # changed the moment it gained its first tool call.
    state.messages[index].content = gr.HTML(
        value=_render_block_html(block["items"], agent_name=block.get("agent_name", "")),
        container=False,
    )


# --- Rendering ---------------------------------------------------------------


def _render_block_html(items: List[Dict[str, Any]], *, agent_name: str = "") -> str:
    kind = (
        "report"
        if agent_name in REPORT_NODES
        else ("plan" if agent_name in PLAN_NODES else "activity")
    )
    sections: List[str] = []
    for item in items:
        item_type = item.get("type")
        if item_type == "message":
            content = item.get("content", "")
            if content:
                sections.append(_render_message_section(content, kind=kind))
        elif item_type == "tool_call":
            sections.append(
                tool_display.render_tool_entry(
                    tool_display.ToolView(
                        label=item.get("label") or item.get("tool_name") or "Tool call",
                        status=item.get("status") or "running",
                        note=item.get("note") or "",
                    ),
                    call_body=item.get("call_body", ""),
                    result_body=item.get("result_body", ""),
                )
            )
        elif item_type == "error":
            sections.append(_render_error_card(item.get("title"), item.get("message"), item.get("detail")))
        elif item_type == "notice":
            sections.append(_render_notice_card(item.get("title"), item.get("message")))
    return f"<div class='agent-block-content agent-block-content--{kind}'>{''.join(sections)}</div>"


def _render_message_section(content: str, *, kind: str = "activity") -> str:
    stripped = (content or "").strip()
    if not stripped:
        return ""
    if _MARKDOWN is not None:
        body = _MARKDOWN.render(stripped)
    else:
        body = f"<div class='agent-message-inline'>{escape(stripped)}</div>"
    return f"<section class='agent-message-section agent-message-section--{kind}'>{body}</section>"


def _render_error_card(title: Optional[str], message: Optional[str], detail: Optional[str]) -> str:
    parts = [
        "<div class='agent-error-card'>",
        f"<div class='agent-error-card__title'>{escape(str(title or 'Something went wrong'))}</div>",
    ]
    if message:
        parts.append(f"<div class='agent-error-card__message'>{escape(str(message))}</div>")
    if detail:
        parts.append(
            "<details class='agent-error-card__detail'><summary>Technical details</summary>"
            f"<pre>{escape(str(detail))}</pre></details>"
        )
    parts.append("</div>")
    return "".join(parts)


def _render_notice_card(title: Optional[str], message: Optional[str]) -> str:
    parts = [
        "<div class='agent-notice-card'>",
        f"<div class='agent-notice-card__title'>{escape(str(title or 'Notice'))}</div>",
    ]
    if message:
        parts.append(f"<div class='agent-notice-card__message'>{escape(str(message))}</div>")
    parts.append("</div>")
    return "".join(parts)


# --- Small helpers -----------------------------------------------------------


def _coerce_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "\n".join(parts).strip()
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", "")).strip()
    return str(content).strip()


def _coerce_stream_text(content: Any) -> str:
    '''Streamed text, with whitespace preserved — it is mid-token.

    Parameters:
    ---------
    content (Any): the streamed chunk's content, whose shape varies.

    Returns:
    ----------
    text (str): the text with whitespace preserved, because it is mid-token.
    '''

    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
        return "".join(parts)
    if isinstance(content, dict) and content.get("type") == "text":
        return str(content.get("text", ""))
    return str(content)


def _get_role(message: Any) -> str:
    role = getattr(message, "type", None) or getattr(message, "role", None)
    if isinstance(message, dict):
        role = role or message.get("type") or message.get("role")
    return str(role or "").lower()


def _is_stream_chunk(message: Any, role: str) -> bool:
    return type(message).__name__.lower() == "aimessagechunk" or str(role).lower() == "aimessagechunk"


def _build_metadata(agent_name: str, block_id: str, *, status: str = "pending") -> Dict[str, Any]:
    label = AGENT_TITLES.get(agent_name, agent_name.replace("_", " ").title())
    return {"title": label, "id": block_id, "status": status}


def _extract_message_seq(block_id: str) -> int:
    try:
        return int(str(block_id).split(":", 1)[1])
    except (IndexError, TypeError, ValueError):
        return 0


__all__ = [
    "AGENT_TITLES",
    "IGNORED_NODES",
    "PLAN_NODES",
    "REPORT_NODES",
    "append_error_block",
    "append_notice_block",
    "append_user_message",
    "export_timeline_snapshot",
    "finalize_active_blocks",
    "process_chunk",
    "rebuild_from_plain_messages",
    "rebuild_from_timeline_snapshot",
    "reset_chat_messages",
]
