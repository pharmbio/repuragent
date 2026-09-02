'''Conversations in SQLite: the list, the titles, the rendered timeline.

Thin async wrapper over `ConversationRepository`, so the UI layer never writes SQL.
It is the only module that knows a conversation is a database row rather than an
entry in a JSON file, which is what earlier versions of this app used.

A thread id is a bare `uuid4`. The web build prefixes it with its owner's id so a
filesystem path can be derived from it without a second lookup; there is one user
here, so the id is the folder name as it stands — and the conversations this app
already had keep working.
'''

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from app.config import DEFAULT_CONVERSATION_TITLE, logger
from app.state import ConversationMeta
from backend.db.repository import ConversationRepository

_repo = ConversationRepository()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_thread_id() -> str:
    return str(uuid4())


def _row_to_meta(row: Dict[str, Any]) -> ConversationMeta:
    return ConversationMeta(
        thread_id=row["thread_id"],
        title=row.get("title") or DEFAULT_CONVERSATION_TITLE,
        created_at=str(row.get("created_at") or ""),
        updated_at=str(row.get("updated_at") or ""),
    )


async def load_threads() -> List[ConversationMeta]:
    await _repo.ensure_schema()
    return [_row_to_meta(row) for row in await _repo.list_threads()]


async def create_thread(title: Optional[str] = None) -> ConversationMeta:
    await _repo.ensure_schema()
    resolved_title = (title or DEFAULT_CONVERSATION_TITLE).strip() or DEFAULT_CONVERSATION_TITLE
    thread_id = new_thread_id()
    await _repo.upsert_thread(thread_id=thread_id, title=resolved_title)
    stamp = _now()
    return ConversationMeta(
        thread_id=thread_id,
        title=resolved_title,
        created_at=stamp,
        updated_at=stamp,
    )


async def update_thread_title(thread_id: str, title: str) -> None:
    await _repo.ensure_schema()
    await _repo.update_thread_title(
        thread_id, (title or "").strip() or DEFAULT_CONVERSATION_TITLE
    )


async def delete_thread(thread_id: str) -> None:
    await _repo.ensure_schema()
    await _repo.delete_thread(thread_id)


async def load_timeline(thread_id: str) -> Optional[Dict[str, Any]]:
    await _repo.ensure_schema()
    try:
        return await _repo.get_thread_timeline(thread_id)
    except Exception as exc:  # noqa: BLE001 - a missing timeline is not fatal
        logger.warning("Unable to load timeline for %s: %s", thread_id, exc)
        return None


async def save_timeline(thread_id: str, timeline: Dict[str, Any]) -> None:
    await _repo.ensure_schema()
    # A thread can be written to before it has a row when a run starts on a
    # conversation created outside this process.
    if not await _repo.get_thread(thread_id):
        await _repo.upsert_thread(thread_id=thread_id, title=DEFAULT_CONVERSATION_TITLE)
    try:
        await _repo.update_thread_timeline(thread_id, timeline or {})
    except Exception as exc:  # noqa: BLE001 - never lose a run over a snapshot write
        logger.warning("Unable to save timeline for %s: %s", thread_id, exc)


__all__ = [
    "create_thread",
    "delete_thread",
    "load_threads",
    "load_timeline",
    "new_thread_id",
    "save_timeline",
    "update_thread_title",
]
