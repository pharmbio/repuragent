'''The SQLite store: the conversations table, and what the type swap can lose.

The web build keeps this in PostgreSQL, and two of its column types have no SQLite
equivalent. Both replacements fail *silently* when they are wrong, which is why they
are pinned here rather than left to the first user who notices:

* **`JSONB` becomes `TEXT` holding JSON.** Postgres hands a dict back; SQLite hands
  back the string it was given. A caller that forgot to decode would re-encode it on
  the next save, and the timeline would nest one level deeper every time.
* **`TIMESTAMPTZ` becomes an ISO-8601 string.** `ORDER BY updated_at DESC` is then a
  *lexicographic* sort, which is chronological only while every value has the same
  shape — so a row written in another format sinks to the bottom of the sidebar
  regardless of its date.

Every test runs against a temporary database file, never the real one.
'''

from __future__ import annotations

import asyncio
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


@contextmanager
def scoped_database() -> Iterator[Path]:
    '''Point the whole database layer at an empty temporary file.

    Every module binds `DATABASE_PATH` at import, so all four are patched, and the
    cached connection and checkpointer are dropped on both sides of the test.

    Returns:
    ----------
    path (Path): the temporary database, yielded for the duration of one test.
    '''

    import app.config as config
    import backend.db.checkpointer as checkpointer
    import backend.db.connection as connection
    import backend.db.legacy_import as legacy_import

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-db-"))
    database = workdir / "repuragent.db"

    modules = (config, connection, checkpointer)
    originals = [(module, module.DATABASE_PATH) for module in modules]
    legacy_original = legacy_import.LEGACY_THREAD_IDS_FILE
    for module, _ in originals:
        module.DATABASE_PATH = database
    legacy_import.LEGACY_THREAD_IDS_FILE = workdir / "thread_ids.json"

    asyncio.run(_reset_state())
    try:
        yield database
    finally:
        asyncio.run(_reset_state())
        for module, value in originals:
            module.DATABASE_PATH = value
        legacy_import.LEGACY_THREAD_IDS_FILE = legacy_original
        shutil.rmtree(workdir, ignore_errors=True)


async def _reset_state() -> None:
    import backend.db.repository as repository
    from backend.db import close_checkpointer, close_connection

    await close_connection()
    await close_checkpointer()
    repository._schema_ready = False


def test_a_conversation_round_trips() -> None:
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await repo.upsert_thread(thread_id="t1", title="Find AML candidates")

        row = await repo.get_thread("t1")
        assert row and row["title"] == "Find AML candidates"
        assert row["created_at"] and row["updated_at"]

        await repo.update_thread_title("t1", "Renamed")
        assert (await repo.get_thread("t1"))["title"] == "Renamed"

        await repo.delete_thread("t1")
        assert await repo.get_thread("t1") is None

    with scoped_database():
        asyncio.run(scenario())


def test_the_timeline_survives_the_jsonb_to_text_swap() -> None:
    '''A dict in, the same dict out — twice, so a re-encode would show up.'''

    from backend.db.repository import ConversationRepository

    payload = {
        "timeline_state_version": 1,
        "timeline_snapshot": {"blocks": [{"role": "assistant", "text": "héllo — ok"}]},
        "processed_message_ids": ["a", "b"],
    }

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await repo.upsert_thread(thread_id="t1", title="x")

        # A conversation that has never been written to reads as an empty document,
        # not as None: the UI treats None as "no such thread".
        assert await repo.get_thread_timeline("t1") == {}

        await repo.update_thread_timeline("t1", payload)
        assert await repo.get_thread_timeline("t1") == payload
        await repo.update_thread_timeline("t1", payload)
        assert await repo.get_thread_timeline("t1") == payload

        assert await repo.get_thread_timeline("nonexistent") is None

    with scoped_database():
        asyncio.run(scenario())


def test_conversations_are_listed_newest_first() -> None:
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        for thread_id in ("first", "second", "third"):
            await repo.upsert_thread(thread_id=thread_id, title=thread_id)
        # Touching the oldest must float it to the top.
        await repo.update_thread_title("first", "first, revisited")

        order = [row["thread_id"] for row in await repo.list_threads()]
        assert order[0] == "first", order
        assert set(order) == {"first", "second", "third"}
        # Same shape for every stamp, or the sort is not chronological.
        stamps = [row["updated_at"] for row in await repo.list_threads()]
        assert all(stamp.endswith("+00:00") and "T" in stamp for stamp in stamps), stamps

    with scoped_database():
        asyncio.run(scenario())


def test_deleting_a_conversation_takes_its_checkpoints_with_it() -> None:
    '''Otherwise a deleted transcript is still recoverable from the same file.'''

    from backend.db import connection as db
    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await get_checkpointer()  # creates `checkpoints` and `writes`

        for thread_id in ("doomed", "kept"):
            await repo.upsert_thread(thread_id=thread_id, title=thread_id)
            await db.execute(
                "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)"
                " VALUES (?, '', ?, 'msgpack', ?, ?)",
                (thread_id, f"cp-{thread_id}", b"x", b"y"),
            )
            await db.execute(
                "INSERT INTO writes (thread_id, checkpoint_ns, checkpoint_id, task_id, idx, channel, type, value)"
                " VALUES (?, '', ?, 'task', 0, 'messages', 'msgpack', ?)",
                (thread_id, f"cp-{thread_id}", b"z"),
            )

        await repo.delete_thread("doomed")

        remaining = {row["thread_id"] for row in await db.fetch_all("SELECT thread_id FROM checkpoints")}
        assert remaining == {"kept"}
        writes = {row["thread_id"] for row in await db.fetch_all("SELECT thread_id FROM writes")}
        assert writes == {"kept"}

    with scoped_database():
        asyncio.run(scenario())


def test_the_schema_is_idempotent() -> None:
    '''It runs at every startup, so a second run must be a no-op, not an error.'''

    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await repo.upsert_thread(thread_id="t1", title="kept")
        # Force the once-per-process guard to let it run again.
        import backend.db.repository as repository

        repository._schema_ready = False
        await repo.ensure_schema()
        assert (await repo.get_thread("t1"))["title"] == "kept"

    with scoped_database():
        asyncio.run(scenario())


def test_the_checkpointer_and_the_app_share_one_file() -> None:
    '''One database, two connections. WAL is what makes that safe.'''

    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        await ConversationRepository().ensure_schema()
        await get_checkpointer()

    with scoped_database() as database:
        asyncio.run(scenario())
        tables = {
            row[0]
            for row in sqlite3.connect(database).execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert {"conversations", "checkpoints", "writes"} <= tables, tables
        mode = sqlite3.connect(database).execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", mode


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
