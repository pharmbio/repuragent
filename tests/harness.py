'''A dependency-free test harness.

`pytest` is not installed in the environment this project is developed in, so the
suite runs itself: `python -m tests.run_all`. The tests are still written as plain
`test_*` functions with plain asserts, so `pytest` collects them unchanged if it is
ever available.
'''

from __future__ import annotations

import shutil
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Sequence, Tuple


@contextmanager
def scoped_results_root() -> Iterator[Path]:
    '''Point the output scope at a temporary directory for the duration of a test.

    Both `app.config` and `backend.utils.output_paths` are patched: the latter binds
    `RESULTS_ROOT` at import time, so patching only the config module would leave the
    real results directory in use.

    Returns:
    ----------
    root (Path): a temporary output scope, yielded for the duration of one test.
    '''

    import app.config as config
    import backend.utils.output_paths as output_paths

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-test-"))
    original_config = config.RESULTS_ROOT
    original_module = output_paths.RESULTS_ROOT
    config.RESULTS_ROOT = workdir
    output_paths.RESULTS_ROOT = workdir
    try:
        yield workdir
    finally:
        config.RESULTS_ROOT = original_config
        output_paths.RESULTS_ROOT = original_module
        shutil.rmtree(workdir, ignore_errors=True)


@contextmanager
def scoped_roots() -> Iterator[Tuple[Path, Path]]:
    '''Temporary results **and** data roots, as `(results, data)`.

    Returns:
    ----------
    roots (tuple): `(results, data)` temporary roots, yielded for the duration of one test.
    '''

    import app.config as config
    import backend.utils.output_paths as output_paths
    import backend.utils.storage_paths as storage_paths

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-test-"))
    results, data = workdir / "results", workdir / "data"
    results.mkdir(parents=True)
    data.mkdir(parents=True)

    originals = (config.RESULTS_ROOT, config.DATA_ROOT, output_paths.RESULTS_ROOT, storage_paths.DATA_ROOT)
    config.RESULTS_ROOT, config.DATA_ROOT = results, data
    output_paths.RESULTS_ROOT = results
    storage_paths.DATA_ROOT = data
    # These are lru_cached, so a stale cached root would defeat the patch.
    storage_paths.get_data_root.cache_clear()
    storage_paths.get_results_root.cache_clear()
    try:
        yield results, data
    finally:
        (
            config.RESULTS_ROOT,
            config.DATA_ROOT,
            output_paths.RESULTS_ROOT,
            storage_paths.DATA_ROOT,
        ) = originals
        storage_paths.get_data_root.cache_clear()
        storage_paths.get_results_root.cache_clear()
        shutil.rmtree(workdir, ignore_errors=True)


@contextmanager
def scoped_sop_index() -> Iterator[Tuple[Path, Path]]:
    '''Point the SOP index and the SOP source folder at temporary directories.

    Four modules bind these paths at import time — `config` itself plus the
    indexer, the retriever and the corpus writer — so all four are patched. The
    functions in `config` read the module's own globals, so patching the constants
    is enough to redirect `ensure_directories` and `index_exists` with them.

    Returns:
    ----------
    roots (tuple): `(sop_folder, index_dir)`, yielded for the duration of one test.
    '''

    import backend.sop_rag.bm25_corpus as bm25_corpus
    import backend.sop_rag.config as sop_config
    import backend.sop_rag.sop_indexer as sop_indexer
    import backend.sop_rag.sop_retriever as sop_retriever

    workdir = Path(tempfile.mkdtemp(prefix="repuragent-sop-"))
    sop_folder = workdir / "SOP"
    index_dir = workdir / "ensemble"
    sop_folder.mkdir(parents=True)

    patched = {
        "SOP_DATA_DIR": sop_folder,
        "SOP_MEMORY_DIR": workdir,
        "INDEX_DIR": index_dir,
        "CHROMA_PERSIST_PATH": index_dir / "chroma_db",
        "DOCSTORE_DIR": index_dir / "docstore",
        "BM25_CORPUS_PATH": index_dir / "bm25_corpus.json",
        "MANIFEST_PATH": index_dir / "manifest.json",
    }
    modules = (sop_config, sop_indexer, sop_retriever, bm25_corpus)
    originals = [
        (module, name, getattr(module, name))
        for module in modules
        for name in patched
        if hasattr(module, name)
    ]
    for module, name, _ in originals:
        setattr(module, name, patched[name])

    sop_config.ensure_directories()
    try:
        yield sop_folder, index_dir
    finally:
        for module, name, value in originals:
            setattr(module, name, value)
        sop_retriever.clear_sop_retriever_cache()
        shutil.rmtree(workdir, ignore_errors=True)


@contextmanager
def scoped_conversation(conversation_id: str = "conv") -> Iterator[None]:
    '''Bind the contextvar a tool reads to identify its conversation.

    Parameters:
    ---------
    conversation_id (str): the conversation id to bind.

    Returns:
    ----------
    context (generator): yields with the contextvar a tool reads bound, so the suite needs no graph.
    '''

    from backend.utils.output_paths import (
        reset_current_conversation_id,
        set_current_conversation_id,
    )

    conversation_token = set_current_conversation_id(conversation_id)
    try:
        yield
    finally:
        reset_current_conversation_id(conversation_token)


def tool_pairing_problems(
    messages: Sequence[Any], *, label: str = "messages"
) -> List[str]:
    '''Every reason a chat API would reject this message list, as the API checks it.

    Scripted models accept anything, so a view that only a real provider would refuse
    passes the suite silently. This applies the two rules OpenAI enforces:

    * a `tool` message must answer a `tool_call` on an *earlier* message —
      *"messages with role 'tool' must be a response to a preceeding message with
      'tool_calls'"*;
    * an assistant message's `tool_call`s must each be answered.

    Parameters:
    ---------
    messages (list): one assembled message list, as it would be sent to the provider.
    label (str): what to name this list in the problem text.

    Returns:
    ----------
    problems (list): one line per violation, naming the index the provider would name, or empty when the list is acceptable.
    '''

    problems: List[str] = []
    open_calls: List[str] = []
    for index, message in enumerate(messages):
        call_id = getattr(message, "tool_call_id", None)
        if call_id is not None:
            if call_id in open_calls:
                open_calls.remove(call_id)
            else:
                problems.append(
                    f"[{label}] messages[{index}] role 'tool' "
                    f"(name={getattr(message, 'name', None)!r}, id={call_id!r}) "
                    "does not answer any preceding message with tool_calls"
                )
            continue
        if open_calls:
            problems.append(
                f"[{label}] messages[{index}] {type(message).__name__} arrives while "
                f"tool_calls {open_calls} are still unanswered"
            )
        open_calls = [
            call["id"] for call in (getattr(message, "tool_calls", None) or []) if call.get("id")
        ]
    if open_calls:
        problems.append(f"[{label}] the list ends with unanswered tool_calls {open_calls}")
    return problems


def assert_tool_pairing(messages: Sequence[Any], *, label: str = "messages") -> None:
    problems = tool_pairing_problems(messages, label=label)
    assert not problems, "\n".join(problems)


def assert_every_model_call_is_valid(models: Dict[str, Any]) -> None:
    '''Check every message list the middleware handed to every scripted model.

    Parameters:
    ---------
    models (dict): role name -> `ScriptedChatModel`, whose `calls` hold what each invocation was given.
    '''

    problems: List[str] = []
    for role, model in models.items():
        for number, call in enumerate(getattr(model, "calls", None) or []):
            problems.extend(tool_pairing_problems(call, label=f"{role} call#{number}"))
    assert not problems, "\n".join(problems)


def collect_tests(namespace: Dict[str, Any]) -> List[Tuple[str, Callable[[], None]]]:
    return [
        (name, value)
        for name, value in sorted(namespace.items())
        if name.startswith("test_") and callable(value)
    ]


def run_module_tests(namespace: Dict[str, Any], *, label: str = "") -> Tuple[int, int]:
    '''Run every `test_*` in a namespace. Returns `(passed, failed)`.

    Parameters:
    ---------
    namespace (dict): the module's globals, scanned for `test_*` functions.
    label (str): what to print as the module's heading.

    Returns:
    ----------
    counts (tuple): `(passed, failed)`. This exists because `pytest` is not installed in the development environment.
    '''

    tests = collect_tests(namespace)
    passed = 0
    failures: List[str] = []
    for name, function in tests:
        try:
            function()
            passed += 1
            print(f"    ok   {name}")
        except Exception:  # noqa: BLE001 - reported, not raised
            failures.append(f"{label or ''}{name}\n{traceback.format_exc()}")
            print(f"    FAIL {name}")
    for failure in failures:
        print("\n" + failure)
    return passed, len(failures)


__all__ = [
    "assert_every_model_call_is_valid",
    "assert_tool_pairing",
    "collect_tests",
    "run_module_tests",
    "scoped_conversation",
    "scoped_results_root",
    "scoped_roots",
    "scoped_sop_index",
    "tool_pairing_problems",
]
