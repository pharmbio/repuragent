'''Carrying a pre-v2 installation's conversations into the `conversations` table.

Earlier versions of this app kept the conversation list in
`MEMORY_ROOT/shortterm_memory/thread_ids.json` and, optionally, the rendered
timelines in a `ui_timeline_snapshots` table. Neither is read any more, so without
this the app would start with an empty sidebar over a database full of history.

**Both sources are read, because neither is complete.** In the installation this was
written against, `thread_ids.json` listed one conversation that has no checkpoints at
all, while the `checkpoints` table held ten that the JSON had never recorded — those
ten were invisible in the old UI and are recovered here.

The import is idempotent (`INSERT` guarded by an existence check, never an update),
so it runs at every startup and does nothing once it has run. Neither source is
deleted: they are left exactly as they were, in case something needs to be re-read.
'''

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import DEFAULT_CONVERSATION_TITLE, MEMORY_ROOT, logger
from backend.db import connection as db

LEGACY_THREAD_IDS_FILE = MEMORY_ROOT / "shortterm_memory" / "thread_ids.json"
LEGACY_TIMELINE_TABLE = "ui_timeline_snapshots"

# A conversation recovered from the checkpoint table alone gets its name from its
# opening request, cut to the same length the UI uses when it titles a new one.
TITLE_MAX_CHARS = 60


def _normalise_timestamp(value: Any) -> Optional[str]:
    '''Turn a legacy timestamp into the ISO-8601 UTC form this schema sorts on.

    The old writer used `datetime.now().strftime('%Y-%m-%d %H:%M:%S')` — naive, local
    and space-separated. Left alone it would sort *before* every ISO-8601 value,
    because a space precedes `T`, and every migrated conversation would sink to the
    bottom of the sidebar regardless of its date.

    Parameters:
    ---------
    value (str): the timestamp as it was stored, in any of the shapes used so far.

    Returns:
    ----------
    stamp (str): the same moment as an ISO-8601 UTC string, or None when it cannot be parsed.
    '''

    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(timezone.utc).isoformat()


def _read_thread_ids_file() -> List[Dict[str, Any]]:
    if not LEGACY_THREAD_IDS_FILE.exists():
        return []
    try:
        raw = json.loads(LEGACY_THREAD_IDS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read %s: %s", LEGACY_THREAD_IDS_FILE, exc)
        return []
    return [entry for entry in raw if isinstance(entry, dict) and entry.get("thread_id")]


async def _table_exists(name: str) -> bool:
    row = await db.fetch_one(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name = ?", (name,)
    )
    return row is not None


async def _checkpoint_threads() -> List[str]:
    '''Conversations the graph could actually reopen.

    Only the **root** namespace counts. A thread whose rows are all under a subgraph
    namespace (`research_agent:...`) is the residue of a run that died before its
    parent committed: there is no state for `aget_state` to return and no transcript
    to render, so listing it would put a permanently empty conversation in the
    sidebar. Its rows are left in the database either way.

    Returns:
    ----------
    thread_ids (list): every conversation with a root checkpoint.
    '''

    if not await _table_exists("checkpoints"):
        return []
    rows = await db.fetch_all("SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = ''")
    return [str(row["thread_id"]) for row in rows if row.get("thread_id")]


async def _legacy_timelines() -> Dict[str, str]:
    if not await _table_exists(LEGACY_TIMELINE_TABLE):
        return {}
    rows = await db.fetch_all(f"SELECT thread_id, snapshot_json FROM {LEGACY_TIMELINE_TABLE}")
    return {
        str(row["thread_id"]): str(row["snapshot_json"])
        for row in rows
        if row.get("thread_id") and row.get("snapshot_json")
    }


async def _describe_from_checkpoint(thread_id: str) -> Tuple[Optional[str], Optional[str]]:
    '''What a checkpoint can say about a conversation nothing else recorded.

    Parameters:
    ---------
    thread_id (str): the conversation to inspect.

    Returns:
    ----------
    description (tuple): `(title, timestamp)` taken from the latest checkpoint's opening request and its `ts`, either of which may be None.
    '''

    try:
        from backend.db.checkpointer import get_checkpointer

        checkpointer = await get_checkpointer()
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
    except Exception as exc:  # noqa: BLE001 - an unreadable checkpoint is not fatal
        logger.debug("Could not read a checkpoint for %s: %s", thread_id, exc)
        return None, None
    if not checkpoint:
        return None, None

    stamp = _normalise_timestamp(checkpoint.get("ts"))

    title = None
    for message in checkpoint.get("channel_values", {}).get("messages", []) or []:
        role = getattr(message, "type", None) or (
            message.get("type") if isinstance(message, dict) else None
        )
        if role not in ("human", "user"):
            continue
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        text = " ".join(str(content or "").split())
        if text:
            title = text[:TITLE_MAX_CHARS]
        break

    return title, stamp


async def import_legacy_conversations() -> int:
    '''Seed `conversations` from whatever an earlier version of this app left behind.

    Returns:
    ----------
    imported (int): how many conversations were added, which is zero on every run after the first.
    '''

    existing = {
        str(row["thread_id"]) for row in await db.fetch_all("SELECT thread_id FROM conversations")
    }

    candidates: Dict[str, Dict[str, Any]] = {}
    for entry in _read_thread_ids_file():
        thread_id = str(entry["thread_id"])
        candidates[thread_id] = {
            "title": entry.get("title") or DEFAULT_CONVERSATION_TITLE,
            "created_at": _normalise_timestamp(entry.get("created_at")),
        }
    for thread_id in await _checkpoint_threads():
        candidates.setdefault(thread_id, {"title": None, "created_at": None})

    pending = {tid: meta for tid, meta in candidates.items() if tid not in existing}
    if not pending:
        return 0

    timelines = await _legacy_timelines()
    fallback_stamp = datetime.now(timezone.utc).isoformat()

    imported = 0
    for thread_id, meta in pending.items():
        title, stamp = meta["title"], meta["created_at"]
        if title is None or stamp is None:
            checkpoint_title, checkpoint_stamp = await _describe_from_checkpoint(thread_id)
            title = title or checkpoint_title or f"Conversation {thread_id[:8]}"
            stamp = stamp or checkpoint_stamp or fallback_stamp
        await db.execute(
            """
            INSERT INTO conversations (thread_id, title, created_at, updated_at, ui_timeline)
            VALUES (?, ?, ?, ?, ?)
            """,
            (thread_id, title, stamp, stamp, timelines.get(thread_id, "{}")),
        )
        imported += 1

    logger.info("Imported %s conversation(s) from the pre-v2 store", imported)
    return imported


__all__ = [
    "LEGACY_THREAD_IDS_FILE",
    "LEGACY_TIMELINE_TABLE",
    "import_legacy_conversations",
]
