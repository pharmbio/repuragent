'''Carrying a pre-v2 installation's conversations into the `conversations` table.

This runs at every startup against a database the user already has, so the failure
modes that matter are the destructive ones: importing twice, overwriting a title
someone changed, or touching the checkpoint rows the transcripts live in.

Both sources are read because neither is complete. In the installation this was
written against, `thread_ids.json` listed one conversation with no checkpoints, and
the `checkpoints` table held nine the JSON had never recorded — invisible in the old
UI, and recovered here.
'''

from __future__ import annotations

import asyncio
import json

from tests.test_db import scoped_database


async def _seed_checkpoint(thread_id: str, *, namespace: str = "") -> None:
    from backend.db import connection as db

    await db.execute(
        "INSERT INTO checkpoints (thread_id, checkpoint_ns, checkpoint_id, type, checkpoint, metadata)"
        " VALUES (?, ?, ?, 'msgpack', ?, ?)",
        (thread_id, namespace, f"cp-{thread_id}-{namespace or 'root'}", b"x", b"y"),
    )


def test_both_sources_are_imported_and_the_import_is_idempotent() -> None:
    from backend.db import connection as db
    from backend.db import legacy_import
    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await get_checkpointer()

        legacy_import.LEGACY_THREAD_IDS_FILE.write_text(
            json.dumps(
                [
                    {
                        "thread_id": "from-json",
                        "created_at": "2026-01-19 10:32:31",
                        "title": "Conversation 2026-01-19 10:32:31",
                    }
                ]
            ),
            encoding="utf-8",
        )
        await _seed_checkpoint("from-checkpoints")

        imported = await legacy_import.import_legacy_conversations()
        threads = {row["thread_id"]: row for row in await repo.list_threads()}
        assert imported == 2, imported
        assert set(threads) == {"from-json", "from-checkpoints"}
        # The JSON's own title wins; the other gets a name derived from its id.
        assert threads["from-json"]["title"] == "Conversation 2026-01-19 10:32:31"
        assert "from-che" in threads["from-checkpoints"]["title"]

        # Someone renames a conversation, then restarts the app.
        await repo.update_thread_title("from-json", "Renamed by the user")
        assert await legacy_import.import_legacy_conversations() == 0
        assert (await repo.get_thread("from-json"))["title"] == "Renamed by the user"

        # And nothing was taken from the checkpoint tables.
        assert (await db.fetch_one("SELECT count(*) AS n FROM checkpoints"))["n"] == 1

    with scoped_database():
        asyncio.run(scenario())


def test_a_legacy_timestamp_is_normalised_so_the_sort_still_works() -> None:
    '''`"2026-01-19 10:32:31"` sorts before every ISO-8601 value: a space precedes `T`.

    Left alone, every migrated conversation would sink to the bottom of the sidebar
    however recent it was.
    '''

    from backend.db import legacy_import
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        legacy_import.LEGACY_THREAD_IDS_FILE.write_text(
            json.dumps([{"thread_id": "old", "created_at": "2026-01-19 10:32:31", "title": "Old"}]),
            encoding="utf-8",
        )
        await legacy_import.import_legacy_conversations()
        await repo.upsert_thread(thread_id="new", title="New")

        stamps = {row["thread_id"]: row["updated_at"] for row in await repo.list_threads()}
        assert stamps["old"].endswith("+00:00") and "T" in stamps["old"], stamps["old"]
        order = [row["thread_id"] for row in await repo.list_threads()]
        assert order == ["new", "old"], order

    with scoped_database():
        asyncio.run(scenario())


def test_a_subgraph_only_thread_is_not_listed() -> None:
    '''A run that died before its parent committed has no state to reopen.

    Its rows stay in the database; what it must not do is put a permanently empty
    conversation in the sidebar.
    '''

    from backend.db import connection as db
    from backend.db import legacy_import
    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await get_checkpointer()

        await _seed_checkpoint("real")
        await _seed_checkpoint("fragment", namespace="research_agent:9f312dbd")

        await legacy_import.import_legacy_conversations()
        assert [row["thread_id"] for row in await repo.list_threads()] == ["real"]
        # Nothing was deleted, only left unlisted.
        assert (await db.fetch_one("SELECT count(*) AS n FROM checkpoints"))["n"] == 2

    with scoped_database():
        asyncio.run(scenario())


def test_a_missing_or_unreadable_legacy_file_is_not_fatal() -> None:
    '''A fresh installation has neither source, and must still start.'''

    from backend.db import legacy_import
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        assert await legacy_import.import_legacy_conversations() == 0

        legacy_import.LEGACY_THREAD_IDS_FILE.write_text("{ not json", encoding="utf-8")
        assert await legacy_import.import_legacy_conversations() == 0
        assert await repo.list_threads() == []

    with scoped_database():
        asyncio.run(scenario())


def test_a_legacy_timeline_snapshot_is_carried_over() -> None:
    '''An earlier version kept rendered timelines in their own table.'''

    from backend.db import connection as db
    from backend.db import legacy_import
    from backend.db.repository import ConversationRepository

    payload = {"timeline_state_version": 1, "timeline_snapshot": {"blocks": []}}

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        await db.execute(
            f"CREATE TABLE {legacy_import.LEGACY_TIMELINE_TABLE} ("
            " thread_id TEXT PRIMARY KEY, snapshot_json TEXT NOT NULL,"
            " updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        await db.execute(
            f"INSERT INTO {legacy_import.LEGACY_TIMELINE_TABLE} (thread_id, snapshot_json, updated_at)"
            " VALUES (?, ?, ?)",
            ("old", json.dumps(payload), "2026-01-19T10:32:31+00:00"),
        )
        legacy_import.LEGACY_THREAD_IDS_FILE.write_text(
            json.dumps([{"thread_id": "old", "created_at": "2026-01-19 10:32:31", "title": "Old"}]),
            encoding="utf-8",
        )

        await legacy_import.import_legacy_conversations()
        assert await repo.get_thread_timeline("old") == payload

    with scoped_database():
        asyncio.run(scenario())


def test_a_conversation_with_only_a_checkpoint_still_shows_its_transcript() -> None:
    '''The v2 UI restores from a rendered snapshot, which an old conversation lacks.

    Without the fallback in `timeline_store.restore`, every conversation carried over
    from an earlier version would open blank with its own history sitting unread in
    the same database file. The rebuilt timeline is saved, so the fallback runs once.
    '''

    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langgraph.checkpoint.base import empty_checkpoint

    from app import timeline_store
    from app.conversation_store import load_timeline
    from app.state import UIState
    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        repo = ConversationRepository()
        await repo.ensure_schema()
        saver = await get_checkpointer()

        checkpoint = empty_checkpoint()
        checkpoint["channel_values"] = {
            "messages": [
                HumanMessage(content="rank AML candidates by hERG risk"),
                # Only tool_calls, nothing to render.
                AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "c1"}]),
                ToolMessage(content="tool output", tool_call_id="c1"),
                AIMessage(content="Here are the ten lowest-risk candidates."),
            ]
        }
        await saver.aput(
            {"configurable": {"thread_id": "old", "checkpoint_ns": ""}},
            checkpoint,
            {"source": "loop", "step": 1},
            {},
        )
        await repo.upsert_thread(thread_id="old", title="Old conversation")

        state = UIState()
        await timeline_store.restore(state, "old")

        rendered = [(message.role, str(message.content or "")) for message in state.messages]
        assert rendered, "an old conversation must not open blank"
        assert rendered[0][0] == "user" and "rank AML candidates" in rendered[0][1]
        # Tool traffic has no call to attach to once the structured timeline is gone.
        assert not any("tool output" in text for _role, text in rendered)

        # Saved, so the next open reads the snapshot instead of the checkpoint.
        assert (await load_timeline("old")).get("timeline_snapshot")

        reopened = UIState()
        await timeline_store.restore(reopened, "old")
        assert len(reopened.messages) == len(state.messages)

    with scoped_database():
        asyncio.run(scenario())


def test_a_conversation_with_neither_source_opens_empty_not_broken() -> None:
    from app import timeline_store
    from app.state import UIState
    from backend.db.checkpointer import get_checkpointer
    from backend.db.repository import ConversationRepository

    async def scenario():
        await ConversationRepository().ensure_schema()
        await get_checkpointer()
        state = UIState()
        await timeline_store.restore(state, "never-run")
        assert state.messages == []

    with scoped_database():
        asyncio.run(scenario())


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
