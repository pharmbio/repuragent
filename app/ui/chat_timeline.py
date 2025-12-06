from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from html import escape

from gradio.components.chatbot import ChatMessage

from app.state import UIState
from app.ui.formatters import _derive_message_id


AGENT_TITLES = {
    "supervisor": "Supervisor",
    "research_agent": "Research Agent",
    "data_agent": "Data Agent",
    "prediction_agent": "Prediction Agent",
    "report_agent": "Report Agent",
    "planning_agent": "Planning Agent",
}

IGNORED_NODES = {"human_chat", "__start__", "__end__"}


def reset_chat_messages(state: UIState) -> None:
    """Reset timeline-related structures."""
    state.messages = []
    state.message_lookup = {}
    state.agent_blocks = {}
    state.tool_call_block_lookup = {}
    state.streaming_message_lookup = {}
    state.last_agent_block_id = None
    state.message_seq = 0


def append_user_message(state: UIState, content: str) -> ChatMessage:
    """Append a user bubble to the Chatbot timeline."""
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
    """Fallback for conversations without raw LangGraph history."""
    reset_chat_messages(state)
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            append_user_message(state, content)
        else:
            if skip_texts and content.strip() in skip_texts:
                continue
            block = _ensure_agent_block(state, "assistant")
            block["items"].append({"type": "message", "content": content})
            _refresh_block_message(state, block["block_id"])


def rebuild_from_raw_messages(
    state: UIState,
    raw_messages: Iterable[Any],
    *,
    skip_texts: Optional[Set[str]] = None,
) -> None:
    """Recreate agent blocks from LangGraph checkpoint messages."""
    reset_chat_messages(state)
    for raw in raw_messages:
        role = _get_role(raw)
        if role in {"human", "user"}:
            text = _coerce_text(getattr(raw, "content", None))
            if text:
                if skip_texts and text.strip() in skip_texts:
                    continue
                append_user_message(state, text)
            msg_id = _derive_message_id(raw)
            if msg_id:
                state.processed_message_ids.add(msg_id)
            continue
        agent_name = getattr(raw, "name", None)
        content_text = _coerce_text(getattr(raw, "content", None))
        if skip_texts and content_text and content_text.strip() in skip_texts:
            msg_id = _derive_message_id(raw)
            if msg_id:
                state.processed_message_ids.add(msg_id)
            continue
        _ingest_message(state, raw, agent_name=agent_name)


def process_chunk(state: UIState, chunk: Dict[str, Any]) -> bool:
    """Apply a LangGraph stream chunk. Returns True if timeline updated."""
    updated = False
    for agent_name, payload in chunk.items():
        if not isinstance(payload, dict):
            continue
        if agent_name.lower() in IGNORED_NODES:
            continue
        messages = payload.get("messages") or []
        for msg in messages:
            if _ingest_message(state, msg, agent_name=agent_name):
                updated = True
    return updated


def process_stream_token(state: UIState, agent_name: Optional[str], chunk: Any) -> bool:
    """Apply a token-level stream update for the given agent."""
    if not agent_name or agent_name.lower() in IGNORED_NODES:
        return False
    return _append_streaming_text(state, agent_name, chunk)


def _append_streaming_text(state: UIState, agent_name: str, chunk: Any) -> bool:
    agent_key = (agent_name or "assistant").lower()
    text = _coerce_stream_text(getattr(chunk, "content", None) if chunk is not None else None)
    if not text:
        return False

    message_id = getattr(chunk, "id", None)
    if isinstance(chunk, dict):
        message_id = message_id or chunk.get("id")

    block = _ensure_agent_block(state, agent_key)
    lookup_key = str(message_id) if message_id else f"{agent_key}:{block['block_id']}:stream"
    stream_entry = state.streaming_message_lookup.get(lookup_key)

    if stream_entry and stream_entry.get("block_id") == block["block_id"]:
        idx = stream_entry.get("item_index")
        if idx is not None and idx < len(block["items"]):
            block["items"][idx]["content"] += text
        else:
            block["items"].append({"type": "message", "content": text})
            state.streaming_message_lookup[lookup_key] = {
                "block_id": block["block_id"],
                "item_index": len(block["items"]) - 1,
            }
    else:
        block["items"].append({"type": "message", "content": text})
        state.streaming_message_lookup[lookup_key] = {
            "block_id": block["block_id"],
            "item_index": len(block["items"]) - 1,
        }

    _refresh_block_message(state, block["block_id"])
    return True


def _ingest_message(state: UIState, raw_msg: Any, agent_name: Optional[str]) -> bool:
    role = _get_role(raw_msg)
    if role in {"human", "user"}:
        msg_id = _derive_message_id(raw_msg)
        if msg_id:
            state.processed_message_ids.add(msg_id)
        return False

    if role == "ai" or role == "assistant":
        return _ingest_ai_message(state, raw_msg, agent_name)

    if role in {"tool", "function"}:
        return _ingest_tool_result(state, raw_msg)

    return False


def _ingest_ai_message(state: UIState, raw_msg: Any, agent_name: Optional[str]) -> bool:
    agent_key = (agent_name or getattr(raw_msg, "name", None) or "assistant").lower()
    message_id = _derive_message_id(raw_msg) or state.next_message_id(agent_key)
    if message_id in state.processed_message_ids:
        return False

    block = _ensure_agent_block(state, agent_key)
    updated = False

    text = _coerce_text(getattr(raw_msg, "content", None))
    primary_stream_key = str(message_id)
    fallback_stream_key = f"{agent_key}:{block['block_id']}:stream"
    stream_entry = state.streaming_message_lookup.get(primary_stream_key) or state.streaming_message_lookup.get(fallback_stream_key)
    if stream_entry and primary_stream_key not in state.streaming_message_lookup:
        state.streaming_message_lookup[primary_stream_key] = stream_entry
        state.streaming_message_lookup.pop(fallback_stream_key, None)
    if text:
        if stream_entry and stream_entry.get("block_id") == block["block_id"]:
            idx = stream_entry.get("item_index")
            if idx is not None and idx < len(block["items"]):
                block["items"][idx]["content"] = text
            else:
                block["items"].append({"type": "message", "content": text})
                state.streaming_message_lookup[str(message_id)] = {
                    "block_id": block["block_id"],
                    "item_index": len(block["items"]) - 1,
                }
        else:
            block["items"].append({"type": "message", "content": text})
            state.streaming_message_lookup[str(message_id)] = {
                "block_id": block["block_id"],
                "item_index": len(block["items"]) - 1,
            }
        updated = True

    tool_calls = getattr(raw_msg, "tool_calls", None) or []
    for call in tool_calls:
        updated |= _append_tool_call(state, block, call)

    if updated:
        _refresh_block_message(state, block["block_id"])

    state.processed_message_ids.add(message_id)
    return updated


def _append_tool_call(state: UIState, block: Dict, call: Any) -> bool:
    call_name = getattr(call, "name", None) or (call.get("name") if isinstance(call, dict) else "tool")
    call_args = getattr(call, "args", None) or (call.get("args") if isinstance(call, dict) else {})
    call_id = getattr(call, "id", None) or (call.get("id") if isinstance(call, dict) else state.next_message_id("tool_call"))

    content, is_html = _format_tool_call_body(call_name, call_args)
    block["items"].append(
        {
            "type": "tool_call",
            "tool_name": call_name,
            "content": content,
            "content_is_html": is_html,
        }
    )
    state.tool_call_block_lookup[str(call_id)] = block["block_id"]
    return True


def _ingest_tool_result(state: UIState, raw_msg: Any) -> bool:
    msg_id = _derive_message_id(raw_msg) or state.next_message_id("tool_result")
    if msg_id in state.processed_message_ids:
        return False

    tool_call_id = getattr(raw_msg, "tool_call_id", None)
    if isinstance(raw_msg, dict):
        tool_call_id = tool_call_id or raw_msg.get("tool_call_id")

    if tool_call_id:
        if tool_call_id in state.processed_tools_ids:
            state.processed_message_ids.add(msg_id)
            return False
        block_id = state.tool_call_block_lookup.get(str(tool_call_id))
    else:
        block_id = state.last_agent_block_id

    if not block_id:
        state.processed_message_ids.add(msg_id)
        return False

    block = state.agent_blocks.get(block_id)
    if not block:
        state.processed_message_ids.add(msg_id)
        return False

    tool_name = getattr(raw_msg, "name", "Tool Result")
    content, is_html = _format_tool_result_content(getattr(raw_msg, "content", None), tool_name)
    block["items"].append(
        {
            "type": "tool_result",
            "tool_name": tool_name,
            "content": content,
            "content_is_html": is_html,
        }
    )
    if tool_call_id:
        state.tool_call_block_lookup.pop(str(tool_call_id), None)
        state.processed_tools_ids.add(tool_call_id)
    _refresh_block_message(state, block_id)
    state.processed_message_ids.add(msg_id)
    return True


def _ensure_agent_block(state: UIState, agent_key: str) -> Dict:
    last_block_id = state.last_agent_block_id
    if last_block_id:
        block = state.agent_blocks.get(last_block_id)
        if block and block["agent_name"] == agent_key:
            return block

    block_id = state.next_message_id(agent_key)
    metadata = _build_metadata(agent_key, block_id)
    chat_message = ChatMessage(role="assistant", content="", metadata=metadata)
    state.messages.append(chat_message)
    state.message_lookup[block_id] = len(state.messages) - 1
    block = {"agent_name": agent_key, "block_id": block_id, "items": []}
    state.agent_blocks[block_id] = block
    state.last_agent_block_id = block_id
    return block


def _refresh_block_message(state: UIState, block_id: str) -> None:
    block = state.agent_blocks.get(block_id)
    if not block:
        return
    idx = state.message_lookup.get(block_id)
    if idx is None or idx >= len(state.messages):
        return
    state.messages[idx].content = _render_block_content(block["items"])


def _render_block_content(items: List[Dict[str, str]]) -> str:
    sections: List[str] = []
    for item in items:
        if item["type"] == "message":
            sections.append(item["content"])
        elif item["type"] == "tool_call":
            sections.append(
                _render_tool_section(
                    "Tool call",
                    item["content"],
                    item.get("tool_name"),
                    body_is_html=item.get("content_is_html", False),
                )
            )
        elif item["type"] == "tool_result":
            sections.append(
                _render_tool_section(
                    "Tool result",
                    item["content"],
                    item.get("tool_name"),
                    body_is_html=item.get("content_is_html", False),
                )
            )
    return "\n\n".join(sections).strip()


def _render_tool_section(title: str, body: str, tool_name: Optional[str], *, body_is_html: bool = False) -> str:
    label = f"{title} · {tool_name}" if tool_name else title
    escaped_label = escape(label)
    body_markup = body if body_is_html else f"<pre>{escape(body)}</pre>"
    return (
        "<details class='tool-block'>"
        f"<summary>{escaped_label}</summary>"
        f"{body_markup}"
        "</details>"
    )


def _format_tool_result_content(raw_content: Any, tool_name: Optional[str]) -> Tuple[str, bool]:
    if raw_content is None:
        return "", False
    if tool_name == "python_executor":
        code = _maybe_extract_code_from_result(raw_content)
        if code:
            return _render_code_block(code, language="python"), True
    if isinstance(raw_content, (dict, list)):
        formatted = json.dumps(raw_content, indent=2)
        return _render_code_block(formatted, language="json"), True
    return str(raw_content), False


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
    """Coerce streamed token content without trimming whitespace."""
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


def _get_role(raw_msg: Any) -> str:
    if hasattr(raw_msg, "type"):
        return getattr(raw_msg, "type")
    if hasattr(raw_msg, "role"):
        return getattr(raw_msg, "role")
    if isinstance(raw_msg, dict):
        return raw_msg.get("type") or raw_msg.get("role", "")
    return ""


def _build_metadata(agent_key: str, block_id: str) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {"id": block_id}
    title = AGENT_TITLES.get(agent_key, agent_key.replace("_", " ").title())
    if agent_key != "planning_agent":
        metadata.update({"title": title, "status": "done"})
    else:
        metadata.update({"title": title})
    return metadata


def _format_tool_call_body(call_name: str, call_args: Any) -> Tuple[str, bool]:
    parsed_args = _parse_tool_args(call_args)
    if call_name == "python_executor":
        code = _maybe_extract_python_code(parsed_args)
        if code:
            return _render_code_block(code, language="python"), True
    if isinstance(parsed_args, (dict, list)):
        try:
            return json.dumps(parsed_args, indent=2), False
        except TypeError:
            return str(parsed_args), False
    return str(parsed_args), False


def _parse_tool_args(call_args: Any) -> Any:
    if isinstance(call_args, str):
        try:
            return json.loads(call_args)
        except json.JSONDecodeError:
            return call_args
    return call_args


def _maybe_extract_python_code(call_args: Any) -> Optional[str]:
    if isinstance(call_args, dict):
        for key in ("code", "python_code", "script", "snippet"):
            code = call_args.get(key)
            if isinstance(code, str) and code.strip():
                return code.rstrip("\n")
    if isinstance(call_args, str) and call_args.strip():
        return call_args.rstrip("\n")
    return None


def _maybe_extract_code_from_result(content: Any) -> Optional[str]:
    if isinstance(content, dict):
        for key in ("stdout", "code", "output", "text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.rstrip("\n")
    if isinstance(content, str) and content.strip():
        return content.rstrip("\n")
    return None


def _render_code_block(code: str, *, language: str = "python") -> str:
    escaped = escape(code.rstrip("\n"), quote=False).replace("#", "&#35;")
    lang_label = language.upper() if language else ""
    label = f"<div class='tool-code-label'>{lang_label}</div>" if lang_label else ""
    return f"<div class='tool-code-block'>{label}<pre><code>{escaped}</code></pre></div>"
