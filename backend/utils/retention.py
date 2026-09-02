'''Deleting conversation files once they are past their retention window.

Off by default here (`RESULT_RETENTION_DAYS` is 0): this is one person's own
machine, and expiring their work on a timer is not a favour. The web build needs it
because uploads and generated artifacts from many accounts accumulate on a shared
host, and the mechanism is kept so an installation that wants a bounded footprint —
a lab machine, a shared workstation — gets it by setting the number of days.

The sweep is **filesystem-based**: it removes files under the two managed roots whose
mtime is older than the window, then prunes the directories left empty.

What must never be swept is everything under `DATA_ROOT` that is not a conversation —
the SOP corpus and the API reference data live there. The guard is structural rather
than a list of names: only `<root>/<uuid conversation>/…` is eligible, so `data/SOP`
and `data/api_related_data` are not reachable by it.
'''

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple
from uuid import UUID

from app.config import RESULT_RETENTION_DAYS, RETENTION_INTERVAL_SECONDS, logger
from backend.utils.output_paths import get_results_root
from backend.utils.storage_paths import get_data_root


def _is_conversation_directory(path: Path) -> bool:
    '''True for a real conversation directory: its name is a thread id (a UUID).

    Everything else at that level is shipped data — `SOP`, `api_related_data` — and is
    left alone.

    Parameters:
    ---------
    path (Path): a directory directly under one of the managed roots.

    Returns:
    ----------
    is_conversation (boolean): True when its name is a thread id, so a stray folder is never swept.
    '''

    try:
        UUID(path.name)
        return True
    except (ValueError, AttributeError):
        return False


def _sweepable_conversations(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for conversation in sorted(root.iterdir()):
        if conversation.is_dir() and _is_conversation_directory(conversation):
            yield conversation


def sweep_expired_files(
    *,
    retention_days: int = RESULT_RETENTION_DAYS,
    roots: Optional[Iterable[Path]] = None,
) -> Tuple[int, int]:
    '''Delete expired files under the managed roots. Returns `(files, directories)`.

    `roots` is explicit rather than read from module globals, so a caller — the test
    suite in particular — can never accidentally point this at the live directories.
    A retention window of zero or less disables the sweep entirely, which is the
    default for this build.

    Parameters:
    ---------
    retention_days (int): how long a file may live before it is eligible for deletion.
    roots (list): the directories to sweep, defaulting to the managed roots.

    Returns:
    ----------
    counts (tuple): `(files, directories)` removed.
    '''

    if retention_days <= 0:
        return 0, 0

    # Resolved at call time, never bound at import: a module-level constant here made
    # the sweep unpatchable, which is how a test once swept the real directories.
    targets = list(roots) if roots is not None else [get_results_root(), get_data_root()]

    cutoff = time.time() - retention_days * 86400
    removed_files = 0
    removed_dirs = 0

    for root in (Path(target) for target in targets):
        for conversation in _sweepable_conversations(root):
            for path in sorted(conversation.rglob("*"), reverse=True):
                try:
                    if path.is_file():
                        if path.stat().st_mtime < cutoff:
                            path.unlink()
                            removed_files += 1
                    elif path.is_dir() and not any(path.iterdir()):
                        path.rmdir()
                        removed_dirs += 1
                except OSError as exc:
                    logger.debug("Could not remove %s: %s", path, exc)
            try:
                if not any(conversation.iterdir()):
                    conversation.rmdir()
                    removed_dirs += 1
            except OSError:
                pass

    if removed_files or removed_dirs:
        logger.info(
            "Retention sweep removed %s file(s) and %s empty directory(ies) older than %s day(s)",
            removed_files,
            removed_dirs,
            retention_days,
        )
    return removed_files, removed_dirs


class RetentionWorker:
    def __init__(
        self,
        *,
        interval_seconds: int = RETENTION_INTERVAL_SECONDS,
        retention_days: int = RESULT_RETENTION_DAYS,
    ) -> None:
        self.interval_seconds = interval_seconds
        self.retention_days = retention_days
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        '''Start the sweep loop, unless retention is switched off.'''

        if self._task and not self._task.done():
            return
        if self.retention_days <= 0:
            # Nothing to do, and a task that wakes every half hour to decide that
            # would be the only thing keeping an idle process busy.
            logger.info("File retention is disabled (RESULT_RETENTION_DAYS=%s)", self.retention_days)
            return
        self._task = asyncio.create_task(self._run(), name="retention-worker")
        logger.info("File retention worker started (%s day window)", self.retention_days)

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        logger.info("File retention worker stopped")

    async def _run(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - best effort, never fatal
                logger.error("Retention job failed: %s", exc)
            await asyncio.sleep(self.interval_seconds)

    async def run_once(self) -> None:
        # Walking the tree is blocking work, so it runs off the event loop.
        await asyncio.to_thread(
            sweep_expired_files,
            retention_days=self.retention_days,
            roots=[get_results_root(), get_data_root()],
        )


retention_worker = RetentionWorker()

__all__ = ["RetentionWorker", "retention_worker", "sweep_expired_files"]
