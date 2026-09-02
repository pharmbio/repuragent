'''Small shared helpers for the UI layer.'''

from __future__ import annotations

import hashlib
from typing import Any, Optional


def derive_message_id(message: Any) -> Optional[str]:
    '''A stable identifier for one message.

    Id de-duplication is what stops a re-emitted transcript from rendering twice, so
    a message without an id still needs one. Tool results fall back to their
    `tool_call_id`, then to a content hash — never to a counter, which would differ
    between the two emissions of the same message.

    Parameters:
    ---------
    message (Any): one message from the graph.

    Returns:
    ----------
    message_id (str): a stable identifier, or None. Id de-duplication is load-bearing: a node wrapping a subgraph re-emits every message that subgraph produced.
    '''

    message_id = getattr(message, "id", None)
    if message_id:
        return str(message_id)

    if getattr(message, "type", None) == "tool":
        tool_call_id = getattr(message, "tool_call_id", None)
        if tool_call_id:
            return f"tool_call:{tool_call_id}"
        name = getattr(message, "name", "tool")
        content = getattr(message, "content", "")
        signature = f"{name}:{repr(content)[:200]}"
        digest = hashlib.sha1(signature.encode("utf-8", errors="ignore")).hexdigest()[:16]
        return f"tool_signature:{digest}"

    return None


__all__ = ["derive_message_id"]
