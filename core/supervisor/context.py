from typing import Optional, Sequence, Tuple, List, Dict, Any

from langchain_core.messages import BaseMessage, SystemMessage


SUMMARY_AGENT_NAME = "summary"
SUMMARY_MEMORY_KEY = "summary_memory"


def _coerce_text(content) -> str:
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


def is_summary_message(message: BaseMessage) -> bool:
    name = getattr(message, "name", None)
    if name and str(name).lower() == SUMMARY_AGENT_NAME:
        return True
    metadata = getattr(message, "response_metadata", None) or {}
    return bool(metadata.get("is_summary"))


def latest_summary_record(
    messages: Sequence[BaseMessage],
) -> Tuple[int, Optional[str], Optional[Dict[str, Any]]]:
    for idx in range(len(messages) - 1, -1, -1):
        msg = messages[idx]
        if is_summary_message(msg):
            metadata = getattr(msg, "response_metadata", None) or {}
            memory = metadata.get(SUMMARY_MEMORY_KEY)
            return idx, _coerce_text(getattr(msg, "content", None)), memory
    return -1, None, None


def messages_since_last_summary(
    messages: Sequence[BaseMessage],
) -> List[BaseMessage]:
    if not messages:
        return []
    idx, _, _ = latest_summary_record(messages)
    if idx < 0:
        return list(messages)
    if idx + 1 >= len(messages):
        return []
    return list(messages[idx + 1 :])


def build_llm_input_messages(
    messages: Sequence[BaseMessage],
) -> List[BaseMessage]:
    """Build compact LLM input using the latest summary + messages after it."""
    if not messages:
        return []
    idx, summary_text, memory = latest_summary_record(messages)
    if idx < 0 or (not summary_text and not memory):
        return list(messages)
    recent = messages_since_last_summary(messages)
    prefix: List[BaseMessage] = []
    memory_text = _format_memory_for_prompt(memory)
    if memory_text:
        prefix.append(
            SystemMessage(
                content="Structured memory (facts/outputs):\n" + memory_text
            )
        )
    if summary_text:
        prefix.append(
            SystemMessage(
                content="Summary of previous workflow:\n" + summary_text
            )
        )
    return prefix + list(recent)


def _format_memory_for_prompt(memory: Optional[Dict[str, Any]]) -> str:
    if not isinstance(memory, dict):
        return ""
    lines: List[str] = []
    facts = memory.get("facts") or []
    outputs = memory.get("outputs") or []
    decisions = memory.get("decisions") or []
    open_questions = memory.get("open_questions") or []

    if facts:
        lines.append("Facts:")
        for item in facts:
            item_text = str(item).strip()
            if item_text:
                lines.append(f"- {item_text}")

    if outputs:
        lines.append("Outputs:")
        for item in outputs:
            if isinstance(item, dict):
                path = str(item.get("path", "")).strip()
                desc = str(item.get("description", "")).strip()
                if path and desc:
                    lines.append(f"- {path} | {desc}")
                elif path:
                    lines.append(f"- {path}")
                elif desc:
                    lines.append(f"- {desc}")
            else:
                item_text = str(item).strip()
                if item_text:
                    lines.append(f"- {item_text}")

    if decisions:
        lines.append("Decisions:")
        for item in decisions:
            item_text = str(item).strip()
            if item_text:
                lines.append(f"- {item_text}")

    if open_questions:
        lines.append("Open questions:")
        for item in open_questions:
            item_text = str(item).strip()
            if item_text:
                lines.append(f"- {item_text}")

    return "\n".join(lines).strip()
