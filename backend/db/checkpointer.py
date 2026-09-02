'''The LangGraph SQLite checkpointer.

A process-wide singleton over its own connection to `DATABASE_PATH`. This is what
makes conversations resumable and what makes the plan-approval `interrupt()` work:
the paused state lives in the database, not in the browser session.

The `checkpoints` / `writes` schema `AsyncSqliteSaver.setup()` creates has been stable
across the 2.x and 3.x lines of `langgraph-checkpoint-sqlite`, so an existing
database file from an earlier version of this app is opened and used as it is.
'''

from __future__ import annotations

import asyncio
from typing import Optional

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import DATABASE_PATH, DATABASE_TIMEOUT_SECONDS, logger

_checkpointer: Optional[AsyncSqliteSaver] = None
_checkpointer_connection: Optional[aiosqlite.Connection] = None
_lock: Optional[asyncio.Lock] = None


async def get_checkpointer() -> AsyncSqliteSaver:
    '''The shared checkpointer, created and set up on first use.

    Returns:
    ----------
    checkpointer (AsyncSqliteSaver): the process-wide singleton, on its own connection so its internal lock actually guards it.
    '''

    global _checkpointer, _checkpointer_connection, _lock

    if _checkpointer is not None:
        return _checkpointer

    if _lock is None:
        _lock = asyncio.Lock()

    async with _lock:
        if _checkpointer is not None:
            return _checkpointer

        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(
            str(DATABASE_PATH),
            timeout=DATABASE_TIMEOUT_SECONDS,
            isolation_level=None,
        )
        checkpointer = AsyncSqliteSaver(connection)
        await checkpointer.setup()

        _checkpointer_connection = connection
        _checkpointer = checkpointer
        logger.info("SQLite checkpointer initialized at %s", DATABASE_PATH)
        return _checkpointer


async def check_database_connection() -> bool:
    '''Diagnostic: is the checkpointer's database actually reachable?

    Returns:
    ----------
    reachable (boolean): True when the checkpointer's database answers, for use as a startup diagnostic.
    '''

    try:
        await get_checkpointer()
        if _checkpointer_connection is None:
            raise RuntimeError("Checkpointer has no connection")
        async with _checkpointer_connection.execute("SELECT 1") as cursor:
            await cursor.fetchone()
        return True
    except Exception as exc:  # noqa: BLE001 - reported as a diagnostic, not raised
        logger.error("SQLite connection check failed: %s", exc)
        return False


async def close_checkpointer() -> None:
    '''Close the checkpointer's connection so a fresh one can back a new saver.'''

    global _checkpointer, _checkpointer_connection
    connection = _checkpointer_connection
    _checkpointer = None
    _checkpointer_connection = None
    if connection is not None:
        try:
            await connection.close()
        except Exception as exc:  # noqa: BLE001 - shutdown must not raise
            logger.debug("Closing the checkpointer connection failed: %s", exc)


__all__ = ["check_database_connection", "close_checkpointer", "get_checkpointer"]
