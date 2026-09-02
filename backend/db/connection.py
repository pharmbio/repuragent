'''The application's SQLite connection.

One `aiosqlite` connection for the whole process, guarded by a lock. This is
deliberately **not** a pool: SQLite serialises writers anyway, the queries here are
single-row reads and writes against one small table, and a pool of connections to
one file buys contention rather than throughput.

It is also deliberately *not* the connection the checkpointer uses, even though both
point at the same file. `AsyncSqliteSaver` holds its own lock around its own use of
its connection; sharing the object would let an application query interleave inside
one of its multi-statement operations. Two connections plus WAL is the supported way
to have two writers on one SQLite database.
'''

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

import aiosqlite

from app.config import DATABASE_PATH, DATABASE_TIMEOUT_SECONDS, logger

_connection: Optional[aiosqlite.Connection] = None
_connection_lock: Optional[asyncio.Lock] = None
_access_lock: Optional[asyncio.Lock] = None


def _locks() -> tuple[asyncio.Lock, asyncio.Lock]:
    global _connection_lock, _access_lock
    if _connection_lock is None:
        _connection_lock = asyncio.Lock()
    if _access_lock is None:
        _access_lock = asyncio.Lock()
    return _connection_lock, _access_lock


async def get_connection() -> aiosqlite.Connection:
    '''The process-wide application connection, opened on first use.

    Returns:
    ----------
    connection (aiosqlite Connection): the shared connection, in WAL mode with a busy timeout so a concurrent write waits rather than raising "database is locked".
    '''

    global _connection
    if _connection is not None:
        return _connection

    connection_lock, _ = _locks()
    async with connection_lock:
        if _connection is not None:
            return _connection

        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        connection = await aiosqlite.connect(
            str(DATABASE_PATH),
            timeout=DATABASE_TIMEOUT_SECONDS,
            isolation_level=None,  # autocommit; every statement stands alone
        )
        connection.row_factory = aiosqlite.Row
        # WAL is what lets the checkpointer's connection and this one write to the
        # same file without blocking each other's reads. The checkpointer sets it
        # too; it is a persistent property of the database, so whoever gets there
        # first wins and the other is a no-op.
        await connection.execute("PRAGMA journal_mode=WAL")
        await connection.execute("PRAGMA foreign_keys=ON")
        await connection.execute(f"PRAGMA busy_timeout={int(DATABASE_TIMEOUT_SECONDS * 1000)}")
        _connection = connection
        logger.info("SQLite database opened at %s", DATABASE_PATH)
        return _connection


@asynccontextmanager
async def connection() -> AsyncIterator[aiosqlite.Connection]:
    '''Borrow the shared connection for one operation.

    Returns:
    ----------
    connection (async generator): yields the shared connection with the access lock held, so two coroutines cannot interleave inside one multi-statement operation.
    '''

    handle = await get_connection()
    _, access_lock = _locks()
    async with access_lock:
        yield handle


async def fetch_all(query: str, parameters: Sequence[Any] = ()) -> List[Dict[str, Any]]:
    async with connection() as handle:
        async with handle.execute(query, tuple(parameters)) as cursor:
            rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def fetch_one(query: str, parameters: Sequence[Any] = ()) -> Optional[Dict[str, Any]]:
    async with connection() as handle:
        async with handle.execute(query, tuple(parameters)) as cursor:
            row = await cursor.fetchone()
    return dict(row) if row is not None else None


async def execute(query: str, parameters: Sequence[Any] = ()) -> int:
    '''Run one statement. Returns how many rows it changed.

    Parameters:
    ---------
    query (str): the statement to run.
    parameters (list): its bound parameters.

    Returns:
    ----------
    rowcount (int): how many rows the statement changed.
    '''

    async with connection() as handle:
        async with handle.execute(query, tuple(parameters)) as cursor:
            return cursor.rowcount


async def execute_script(script: str) -> None:
    async with connection() as handle:
        await handle.executescript(script)


async def close_connection() -> None:
    '''Close the shared connection when the app shuts down.'''

    global _connection
    if _connection is not None:
        await _connection.close()
        _connection = None
        logger.info("SQLite database closed")


__all__ = [
    "close_connection",
    "connection",
    "execute",
    "execute_script",
    "fetch_all",
    "fetch_one",
    "get_connection",
]
