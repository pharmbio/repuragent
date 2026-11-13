'''The retention sweep, and the data it must never touch.

The SOP corpus and the API reference data live under `DATA_ROOT` alongside the
conversations, so an over-eager sweep would delete material a fresh clone needs.
That is the property worth testing.

The sweep is **off by default in this build** (`RESULT_RETENTION_DAYS` is 0) — this
is one person's own machine. The tests pass the window explicitly, because what is
being checked is the policy, not the default.
'''

from __future__ import annotations

import os
import time
from pathlib import Path
from uuid import uuid4

from tests.harness import scoped_roots

OLD = time.time() - 30 * 86400


def _age(path: Path) -> None:
    os.utime(path, (OLD, OLD))


def test_expired_conversation_files_are_removed() -> None:
    from backend.utils.retention import sweep_expired_files

    with scoped_roots() as (results, data):
        stale = results / str(uuid4())
        stale.mkdir(parents=True)
        for name in ("master_ranked.csv", "plan.md"):
            (stale / name).write_text("x")
            _age(stale / name)

        fresh = results / str(uuid4())
        fresh.mkdir(parents=True)
        (fresh / "candidates.csv").write_text("x")

        upload = data / str(uuid4())
        upload.mkdir(parents=True)
        (upload / "compounds.csv").write_text("x")
        _age(upload / "compounds.csv")

        files, _dirs = sweep_expired_files(retention_days=2, roots=[results, data])
        assert files == 3, files
        assert not stale.exists(), "an emptied conversation directory is pruned"
        assert (fresh / "candidates.csv").exists(), "recent work must survive"
        assert not (upload / "compounds.csv").exists()


def test_shipped_data_is_never_swept() -> None:
    '''Only `<root>/<uuid>/…` is eligible, which is what keeps the corpus safe.

    The guard is structural rather than a list of names: a conversation directory is
    named by a thread id, and a thread id is a UUID. `SOP` and `api_related_data` are
    not, so the sweep cannot reach inside them however old their contents are.
    '''

    from backend.utils.retention import sweep_expired_files

    with scoped_roots() as (_results, data):
        for directory in ("SOP", "api_related_data"):
            path = data / directory
            path.mkdir(parents=True)
            target = path / "GHS_Rev11e.pdf"
            target.write_text("committed")
            _age(target)

        files, dirs = sweep_expired_files(retention_days=1, roots=[data])
        assert files == 0 and dirs == 0
        assert (data / "SOP" / "GHS_Rev11e.pdf").exists()
        assert (data / "api_related_data" / "GHS_Rev11e.pdf").exists()


def test_a_zero_window_disables_the_sweep() -> None:
    '''Which is this build's default, so this is the path that normally runs.'''

    from backend.utils.retention import sweep_expired_files

    with scoped_roots() as (results, _data):
        directory = results / str(uuid4())
        directory.mkdir(parents=True)
        target = directory / "old.csv"
        target.write_text("x")
        _age(target)

        assert sweep_expired_files(retention_days=0, roots=[results]) == (0, 0)
        assert target.exists()


def test_the_worker_does_not_start_when_retention_is_off() -> None:
    '''A disabled sweep should cost nothing, not wake up every half hour to decide.'''

    import asyncio

    from backend.utils.retention import RetentionWorker

    async def scenario():
        worker = RetentionWorker(retention_days=0, interval_seconds=1)
        await worker.start()
        started = worker._task is not None
        await worker.stop()

        enabled = RetentionWorker(retention_days=1, interval_seconds=3600)
        await enabled.start()
        enabled_started = enabled._task is not None
        await enabled.stop()
        return started, enabled_started

    started, enabled_started = asyncio.run(scenario())
    assert not started, "a zero window must not schedule a task"
    assert enabled_started, "a positive window must"


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
