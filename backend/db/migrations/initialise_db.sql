-- Application schema for the local Repuragent build.
--
-- One SQLite file holds this table alongside LangGraph's own `checkpoints` and
-- `writes` tables, which the checkpointer creates for itself. Keeping them together
-- is what lets deleting a conversation remove its row and its checkpoints in one
-- transaction.
--
-- Every statement is idempotent, so this runs at every startup and there is no
-- manual migration step.

CREATE TABLE IF NOT EXISTS conversations (
    thread_id   TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    -- The rendered timeline, as a JSON document. SQLite has no JSON column type;
    -- json1 functions work on TEXT, and nothing here queries inside the document.
    ui_timeline TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS conversations_updated_idx
    ON conversations(updated_at DESC);
