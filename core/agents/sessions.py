'''Specialist sessions: which session a handoff brief belongs to, and the list the supervisor sees.

Everything is derived from the briefs stored in the transcript, so there is no state of its own.
Nothing here imports from `core`, which is what lets both `handoff` and `context` use it.
'''

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from langchain_core.messages import BaseMessage, ToolMessage

from app.config import SPECIALIST_SESSION_LIST_MAX_ITEMS

# The session a brief belongs to is stored on the brief `ToolMessage` itself, under this
# key in `additional_kwargs`. It survives `Command.PARENT`, the checkpoint and a reload,
# and is never sent to the model provider, so a fresh brief reads exactly as before.
SESSION_KEY = "specialist_session"
OBJECTIVE_MAX_CHARS = 120

# The whole supervisor session list, header included.
SESSION_LIST_MAX_CHARS = 2000


# Session identity

def _short_role(agent_name: str) -> str:
    return agent_name.removesuffix("_agent")


def session_meta(message: BaseMessage) -> Optional[Dict[str, Any]]:
    '''The session a handoff brief belongs to.

    Parameters:
    ---------
    message (BaseMessage): any message from the transcript.

    Returns:
    ----------
    meta (dict): the brief's `specialist_session` record, or None for anything else — including a legacy brief without one, and a record whose id or role is not a string.
    '''

    if not isinstance(message, ToolMessage):
        return None
    meta = (getattr(message, "additional_kwargs", None) or {}).get(SESSION_KEY)
    if not isinstance(meta, dict):
        return None
    if not isinstance(meta.get("id"), str) or not isinstance(meta.get("role"), str):
        return None
    return meta


def known_sessions(messages: Sequence[BaseMessage], agent_name: str) -> List[str]:
    '''The session ids this specialist already has in the conversation.

    Parameters:
    ---------
    messages (list): the whole stored transcript.
    agent_name (str): the specialist whose sessions to list.

    Returns:
    ----------
    session_ids (list): its session ids, in the order they first appear.
    '''

    seen: List[str] = []
    for message in messages or []:
        meta = session_meta(message)
        if meta and meta["role"] == agent_name and meta["id"] not in seen:
            seen.append(meta["id"])
    return seen


def mint_session_id(
    messages: Sequence[BaseMessage], agent_name: str, tool_call_id: str
) -> str:
    '''The id of the new session a fresh handoff opens.

    Parameters:
    ---------
    messages (list): the whole stored transcript, which is where earlier sessions are recorded.
    agent_name (str): the specialist the session belongs to.
    tool_call_id (str): the `tool_call` this handoff answers.

    Returns:
    ----------
    session_id (str): the id already recorded for this `tool_call_id` when the handoff is replayed, otherwise `<short role>-<n>` with `n` one past the highest number this role has used.
    '''

    # Derived from the transcript rather than a counter, so a replayed handoff gets the
    # same id and ids survive a restart.
    for message in messages or []:
        meta = session_meta(message)
        if (
            meta
            and meta["role"] == agent_name
            and getattr(message, "tool_call_id", None) == tool_call_id
        ):
            return meta["id"]

    prefix = _short_role(agent_name)
    taken = set(known_sessions(messages, agent_name))
    pattern = re.compile(rf"^{re.escape(prefix)}-(\d+)$")
    numbers = [int(match.group(1)) for item in taken if (match := pattern.match(item))]
    number = max(numbers, default=0) + 1
    while f"{prefix}-{number}" in taken:
        number += 1
    return f"{prefix}-{number}"


def check_session_reference(
    messages: Sequence[BaseMessage], agent_name: str, session_id: str
) -> Optional[str]:
    '''Why `session_id` cannot be continued by this specialist.

    Parameters:
    ---------
    messages (list): the whole stored transcript.
    agent_name (str): the specialist being delegated to.
    session_id (str): the session the supervisor asked to continue.

    Returns:
    ----------
    problem (str): the refusal, listing this specialist's valid sessions, or None when the session can be continued.
    '''

    valid = known_sessions(messages, agent_name)
    if session_id in valid:
        return None
    listed = ", ".join(valid) if valid else "none yet"
    return (
        f"Not delivered: `session_id` {session_id!r} is not a {agent_name} session in this "
        f"conversation. Valid {agent_name} sessions: {listed}. Omit `session_id` to start "
        "a fresh session."
    )


def clip_objective(objective: str) -> str:
    '''The objective as recorded on a brief: whitespace collapsed, at most `OBJECTIVE_MAX_CHARS`.'''

    flat = " ".join((objective or "").split())
    if len(flat) <= OBJECTIVE_MAX_CHARS:
        return flat
    return flat[: OBJECTIVE_MAX_CHARS - 1].rstrip() + "…"


# The supervisor's session list

def render_session_list(messages: Sequence[BaseMessage]) -> str:
    '''The specialist sessions the supervisor can continue.

    Parameters:
    ---------
    messages (list): the whole stored transcript. Compaction only changes the supervisor's view, so sessions delegated before a summary are still listed.

    Returns:
    ----------
    block (str): the sessions, most recently used first, or an empty string when there are none. Only the supervisor's own clipped objective is shown, never text a specialist wrote, and the block stays within `SESSION_LIST_MAX_CHARS`: ids that do not fit are counted, not dropped silently.
    '''

    latest: Dict[str, Dict[str, Any]] = {}
    for message in messages:
        meta = session_meta(message)
        if meta:
            # Re-inserting moves a continued session to the end, so its last use orders it.
            latest.pop(meta["id"], None)
            latest[meta["id"]] = meta
    if not latest:
        return ""

    ordered = list(reversed(latest.values()))
    header = (
        "Specialist sessions in this conversation, most recently used first. Pass one as "
        "`session_id` to continue it; omit `session_id` to start a fresh session:"
    )
    lines = [header]
    used = len(header)
    # Room kept for the line that counts what did not fit.
    reserve = len("\n- older: ") + len(f" +{len(ordered)} more")

    detailed = 0
    for meta in ordered[: max(SPECIALIST_SESSION_LIST_MAX_ITEMS, 0)]:
        objective = clip_objective(str(meta.get("objective") or ""))
        line = f"- {meta['id']} · {meta['role']} · {objective}"
        if used + 1 + len(line) + reserve > SESSION_LIST_MAX_CHARS:
            break
        lines.append(line)
        used += 1 + len(line)
        detailed += 1

    older = [meta["id"] for meta in ordered[detailed:]]
    if older:
        budget = SESSION_LIST_MAX_CHARS - used - reserve
        shown: List[str] = []
        for session_id in older:
            if len(", ".join([*shown, session_id])) > budget:
                break
            shown.append(session_id)
        remaining = len(older) - len(shown)
        parts = [", ".join(shown)] if shown else []
        if remaining:
            parts.append(f"+{remaining} more")
        lines.append("- older: " + " ".join(parts))
    return "\n".join(lines)


__all__ = [
    "OBJECTIVE_MAX_CHARS",
    "SESSION_KEY",
    "SESSION_LIST_MAX_CHARS",
    "check_session_reference",
    "clip_objective",
    "known_sessions",
    "mint_session_id",
    "render_session_list",
    "session_meta",
]
