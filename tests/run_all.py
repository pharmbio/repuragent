'''Run the whole suite: `python -m tests.run_all`.

Written without pytest because the environment this project is developed in does not
have it. Every test is a plain `test_*` function with plain asserts, so `pytest tests`
works identically wherever it is installed.
'''

from __future__ import annotations

import importlib
import sys
import time
from typing import List

MODULES: List[str] = [
    "tests.test_imports",
    "tests.test_db",
    "tests.test_migration",
    "tests.test_plan_store",
    "tests.test_context",
    "tests.test_sandbox",
    "tests.test_sop_rag",
    "tests.test_literature",
    "tests.test_ui",
    "tests.test_run_controller",
    "tests.test_retention",
    "tests.test_wiring",
    "tests.test_isolation",
    "tests.test_tool_pairing",
    "tests.test_graph",
]


def main() -> int:
    from tests.harness import run_module_tests

    total_passed = 0
    total_failed = 0
    started = time.monotonic()

    for name in MODULES:
        print(f"\n{name}")
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            print(f"    FAIL could not import module: {type(exc).__name__}: {exc}")
            total_failed += 1
            continue
        passed, failed = run_module_tests(vars(module), label=f"{name}.")
        total_passed += passed
        total_failed += failed

    elapsed = time.monotonic() - started
    print(
        f"\n{total_passed} passed, {total_failed} failed in {elapsed:.1f}s"
        + ("" if total_failed else "  — all good")
    )
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
