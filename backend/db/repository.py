'''Database access for conversations.

One table, `conversations`, holding the sidebar list and the rendered timeline. It
replaces the web build's `user_threads` and everything that scoped it by account.

Two shapes differ from PostgreSQL and are worth naming, because both are silent when
wrong:

* **timestamps are ISO-8601 UTC strings.** SQLite has no timestamp type, and
  `ORDER BY updated_at DESC` is a *lexicographic* sort — which is chronological only
  while every value has the same shape. `_now()` is the single place that mints them.
* **`ui_timeline` is TEXT holding JSON**, encoded and decoded here. Postgres `JSONB`
  hands back a dict; SQLite hands back the string it was given, so a caller that
  forgot to decode would store a JSON-encoded string on the next write and the
  timeline would nest one level deeper on every save.
'''

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.config import logger
from backend.db import connection as db

_schema_ready = False
_schema_lock: Optional[asyncio.Lock] = None

MIGRATION_PATH = Path(__file__).resolve().parent / "migrations" / "initialise_db.sql"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ConversationRepository:
    '''Execute conversation queries against the shared connection.'''

    async def ensure_schema(self) -> None:
        '''Apply the committed migration once per process.

        `migrations/initialise_db.sql` is entirely idempotent, so running it at
        startup keeps one source of truth for the schema and removes any manual
        step a fresh installation would otherwise need.
        '''

        global _schema_ready, _schema_lock

        if _schema_ready:
            return
        if _schema_lock is None:
            _schema_lock = asyncio.Lock()

        async with _schema_lock:
            if _schema_ready:
                return
            try:
                sql = MIGRATION_PATH.read_text(encoding="utf-8")
            except OSError as exc:
                logger.error("Could not read %s: %s", MIGRATION_PATH, exc)
                return
            await db.execute_script(sql)
            _schema_ready = True
            logger.info("Database schema verified")

    # --- Conversations --------------------------------------------------------

    async def upsert_thread(self, *, thread_id: str, title: str) -> None:
        stamp = _now()
        await db.execute(
            """
            INSERT INTO conversations (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE
            SET title = excluded.title,
                updated_at = excluded.updated_at
            """,
            (thread_id, title, stamp, stamp),
        )

    async def list_threads(self) -> List[Dict[str, Any]]:
        return await db.fetch_all(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            ORDER BY updated_at DESC
            """
        )

    async def get_thread(self, thread_id: str) -> Optional[Dict[str, Any]]:
        return await db.fetch_one(
            """
            SELECT thread_id, title, created_at, updated_at
            FROM conversations
            WHERE thread_id = ?
            """,
            (thread_id,),
        )

    async def update_thread_title(self, thread_id: str, title: str) -> None:
        await db.execute(
            """
            UPDATE conversations
            SET title = ?, updated_at = ?
            WHERE thread_id = ?
            """,
            (title, _now(), thread_id),
        )

    async def get_thread_timeline(self, thread_id: str) -> Optional[Dict[str, Any]]:
        row = await db.fetch_one(
            "SELECT ui_timeline FROM conversations WHERE thread_id = ?",
            (thread_id,),
        )
        if not row:
            return None
        try:
            timeline = json.loads(row.get("ui_timeline") or "{}")
        except (TypeError, ValueError) as exc:
            logger.warning("Stored timeline for %s is not valid JSON: %s", thread_id, exc)
            return None
        return timeline if isinstance(timeline, dict) else None

    async def update_thread_timeline(self, thread_id: str, timeline: Dict[str, Any]) -> None:
        payload = json.dumps(timeline or {}, ensure_ascii=True)
        await db.execute(
            """
            UPDATE conversations
            SET ui_timeline = ?, updated_at = ?
            WHERE thread_id = ?
            """,
            (payload, _now(), thread_id),
        )

    async def delete_thread(self, thread_id: str) -> None:
        '''Remove a conversation, its checkpoints and its pending writes.

        The web build deletes only the row and leaves the checkpoints in Postgres for
        an operator to prune. Here the database is a file in the user's own
        directory: a deleted conversation whose transcript is still recoverable from
        it is not deleted, and nothing else would ever reclaim the space.

        Parameters:
        ---------
        thread_id (str): the conversation to remove.
        '''

        async with db.connection() as handle:
            await handle.execute("BEGIN")
            try:
                await handle.execute("DELETE FROM conversations WHERE thread_id = ?", (thread_id,))
                # These two belong to the checkpointer. They are only ever created by
                # `AsyncSqliteSaver.setup()`, which the app runs at startup, but a
                # test or a first boot may get here before that.
                for table in ("writes", "checkpoints"):
                    try:
                        await handle.execute(f"DELETE FROM {table} WHERE thread_id = ?", (thread_id,))
                    except Exception as exc:  # noqa: BLE001 - table not created yet
                        logger.debug("Could not clear %s for %s: %s", table, thread_id, exc)
                await handle.execute("COMMIT")
            except Exception:
                await handle.execute("ROLLBACK")
                raise


__all__ = ["ConversationRepository", "MIGRATION_PATH"]
