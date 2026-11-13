'''The sandbox: output scoping, the Python session, and file reads.

The properties here are multi-tenancy ones. This is a shared deployment, so a run
must not be able to write outside its own conversation, read another user's uploads,
or share an interpreter namespace with a conversation running at the same time.
'''

from __future__ import annotations

import threading

from tests.harness import scoped_conversation, scoped_results_root, scoped_roots


# --- Output scoping -----------------------------------------------------------


def test_output_scope_is_per_conversation() -> None:
    from backend.utils.output_paths import conversation_output_root

    with scoped_results_root() as results:
        first = conversation_output_root("abc")
        second = conversation_output_root("def")
        assert first != second
        # One directory level, named by the conversation: there is one user, so a
        # user level would be a single constant folder holding everything.
        assert first.relative_to(results).parts == ("abc",)


def test_write_paths_cannot_escape_the_scope() -> None:
    from backend.utils.output_paths import conversation_output_root, resolve_output_folder

    with scoped_results_root():
        scope = {"conversation_id": "abc"}
        root = conversation_output_root(scope["conversation_id"])

        assert resolve_output_folder("figures", **scope) == root / "figures"
        # Every escape attempt is clamped back to the scope root rather than raising,
        # so a clumsy path never becomes a write somewhere else.
        for attempt in ("../../elsewhere", "/etc", "../other/abc", "figures/../../..") :
            resolved = resolve_output_folder(attempt, **scope)
            assert resolved == root or root in resolved.parents, (attempt, resolved)


def test_artifact_ledger_is_read_from_disk() -> None:
    from backend.utils.output_paths import (
        conversation_output_root,
        describe_output_artifacts,
        list_task_files,
    )

    with scoped_results_root():
        scope = {"conversation_id": "abc"}
        root = conversation_output_root("abc")
        (root / "master_ranked.csv").write_text("smiles,score\nCCO,1\n")
        (root / "figures").mkdir()
        (root / "figures" / "heatmap.png").write_bytes(b"png")

        ledger = describe_output_artifacts(max_items=25, **scope)
        assert "master_ranked.csv" in ledger and "heatmap.png" in ledger
        assert {path.name for path in list_task_files("abc")} == {
            "master_ranked.csv",
            "heatmap.png",
        }


# --- The Python session -------------------------------------------------------


def test_python_sessions_are_isolated_per_conversation() -> None:
    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        InterpreterError,
        local_python_executor,
        reset_executor_state,
    )

    imports = sorted(set(BASE_BUILTIN_MODULES))
    reset_executor_state()
    assert local_python_executor("shortlist = 42", imports, session_key="a") == 42
    assert local_python_executor("shortlist", imports, session_key="a") == 42

    try:
        local_python_executor("shortlist", imports, session_key="b")
        raise AssertionError("session b must not see session a's variables")
    except InterpreterError as exc:
        assert "not defined" in str(exc)

    # A reset clears only the caller's session.
    reset_executor_state("b")
    assert local_python_executor("shortlist", imports, session_key="a") == 42
    reset_executor_state()


def test_undefined_names_suggest_rather_than_substitute() -> None:
    '''The upstream interpreter silently returned the closest-named variable.'''

    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        InterpreterError,
        local_python_executor,
        reset_executor_state,
    )

    imports = sorted(set(BASE_BUILTIN_MODULES))
    reset_executor_state("suggest")
    local_python_executor("herg_scores = [1, 2, 3]", imports, session_key="suggest")
    try:
        local_python_executor("hergs_scores", imports, session_key="suggest")
        raise AssertionError("a near-miss name must raise, not resolve")
    except InterpreterError as exc:
        message = str(exc)
        assert "Did you mean" in message and "herg_scores" in message
    reset_executor_state("suggest")


def test_a_runaway_execution_is_abandoned() -> None:
    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        ExecutionTimeout,
        local_python_executor,
    )

    try:
        local_python_executor(
            "i = 0\nwhile True:\n    i += 1",
            sorted(set(BASE_BUILTIN_MODULES)),
            session_key="timeout",
            timeout=1,
        )
        raise AssertionError("expected a timeout")
    except ExecutionTimeout as exc:
        assert "1s limit" in str(exc)


def test_stop_reaches_a_running_execution() -> None:
    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        ExecutionCancelled,
        local_python_executor,
    )

    event = threading.Event()
    threading.Timer(0.4, event.set).start()
    try:
        local_python_executor(
            "i = 0\nwhile True:\n    i += 1",
            sorted(set(BASE_BUILTIN_MODULES)),
            session_key="cancel",
            timeout=60,
            cancel_event=event,
        )
        raise AssertionError("expected cancellation")
    except ExecutionCancelled:
        pass


def test_python_tool_writes_only_inside_the_scope() -> None:
    from core.tools.python_executor import python_executor, reset_python_state

    with scoped_roots() as (results, _data):
        with scoped_conversation("abc"):
            reset_python_state.invoke({})
            written = python_executor.invoke(
                {"code": "p = prepare_output_path('candidates.csv')\nopen(p,'w').write('smiles\\nCCO')\np"}
            )
            assert str(results) in str(written)
            assert (results / "abc" / "candidates.csv").read_text().startswith("smiles")

            # A relative write lands in the scope, not the process CWD.
            python_executor.invoke({"code": "open('notes.txt','w').write('scoped')"})
            assert (results / "abc" / "notes.txt").read_text() == "scoped"

            # An absolute write outside the scope is refused.
            refused = python_executor.invoke({"code": "open('/tmp/rp-escape.txt','w').write('x')"})
            assert isinstance(refused, dict) and refused["ok"] is False
            assert "output scope" in refused["error"]
            reset_python_state.invoke({})


def test_introspection_escapes_are_refused() -> None:
    '''Every route to a dunder goes through one policy, including the indirect ones.'''

    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        InterpreterError,
        local_python_executor,
        reset_executor_state,
    )

    imports = sorted(set(BASE_BUILTIN_MODULES) | {"asyncio", "pathlib"})
    escapes = {
        "subclass walk": "().__class__.__bases__[0].__subclasses__()",
        "method fetch": "f = lambda: 1\nf.__globals__",
        "getattr builtin": "f = lambda: 1\ngetattr(f, '__glob' + 'als__')",
        "format field": "f = lambda: 1\n'{0.__globals__}'.format(f)",
        "nested format spec": "f = lambda: 1\n'{0:{1.__globals__}}'.format(1, f)",
        "module namespace": "import math\nmath.__dict__",
        "frame locals": "vars()",
        "raw builtin": "__import__('os')",
        "dunder import": "from math import __loader__",
        "dunder write": "class A:\n    pass\nA.__getattribute__ = None",
        "asyncio subprocess": "import asyncio\nasyncio.create_subprocess_shell('true')",
    }
    reset_executor_state("escape")
    for label, code in escapes.items():
        try:
            local_python_executor(code, imports, session_key="escape")
        except InterpreterError:
            continue
        raise AssertionError(f"{label} was not refused: {code!r}")

    # A module re-exported by an allowed one is not a way round the import list.
    assert local_python_executor("import pathlib\nhasattr(pathlib, 'os')", imports, session_key="escape") is False
    reset_executor_state("escape")


def test_ordinary_class_code_still_runs() -> None:
    '''The allowlist must not cost the dunders generated code actually uses.'''

    from backend.utils.local_python_executor import (
        BASE_BUILTIN_MODULES,
        local_python_executor,
        reset_executor_state,
    )

    imports = sorted(set(BASE_BUILTIN_MODULES))
    reset_executor_state("classes")
    code = (
        "class Scorer:\n"
        "    def __init__(self, weight):\n"
        "        self.weight = weight\n"
        "    def __repr__(self):\n"
        "        return f'Scorer({self.weight})'\n"
        "class Herg(Scorer):\n"
        "    def __init__(self):\n"
        "        super().__init__(0.4)\n"
        "        self.name = type(self).__name__\n"
        "s = Herg()\n"
        "(s.weight, s.__class__.__name__, s.__dict__, getattr(s, 'name'), repr(s))"
    )
    assert local_python_executor(code, imports, session_key="classes") == (
        0.4, "Herg", {"weight": 0.4, "name": "Herg"}, "Herg", "Scorer(0.4)",
    )
    reset_executor_state("classes")


def test_the_python_tool_cannot_import_the_host_or_the_app() -> None:
    from core.tools.python_executor import AUTHORIZED_IMPORTS, python_executor, reset_python_state

    for module in ("os", "sys", "app", "backend", "core"):
        assert module not in AUTHORIZED_IMPORTS, module

    with scoped_roots():
        with scoped_conversation("abc"):
            reset_python_state.invoke({})
            for code in ("import os", "import sys", "from app import config", "import backend.utils"):
                refused = python_executor.invoke({"code": code})
                assert isinstance(refused, dict) and refused["ok"] is False, code
                assert "not allowed" in refused["error"], (code, refused["error"])
            reset_python_state.invoke({})


def test_an_oversized_result_is_bounded_whatever_shape_it_is() -> None:
    '''The cap has to apply to what is stored, not to what happens to be a `str`.

    Containers used to be returned untouched, on the reasoning that a structured
    result should keep its structure. A run that returned
    `{"dfs": {"Sheet1": df.to_json()}}` for a 95 798-row frame therefore put a
    143 MB `ToolMessage` into the message list — and because LangGraph's SQLite
    checkpointer stores the whole state as one blob per superstep, those two
    messages were rewritten 126 times and cost 20.6 GB in a single conversation.
    '''

    from app.config import PYTHON_OUTPUT_MAX_CHARS
    from core.tools.python_executor import _bound_output, _measure

    # There are two regimes, on purpose. A container small enough to render is
    # elided like any other text, so its content survives at both ends — dropping
    # that for a one-line summary would tell the caller less than the unbounded
    # behaviour did. One too large to render is described instead, because `str()`
    # on it rebuilds the whole object, which is the cost being avoided.
    from core.tools.python_executor import RENDERABLE_MAX_CHARS

    renderable = {"rows": [{"smiles": "CCO", "score": 0.9, "note": "a" * 40} for _ in range(400)]}
    elided = _bound_output(renderable)
    assert isinstance(elided, str)
    assert len(elided) <= PYTHON_OUTPUT_MAX_CHARS * 1.2
    assert "smiles" in elided, "a renderable result keeps its content"
    assert "omitted from the middle" in elided
    assert "output scope" in elided

    # The shape that actually did it, at a size no one would want rendered.
    oversized = {
        "path": "/scope/annotations.xlsx",
        "sheets": ["Sheet1"],
        "dfs": {"Sheet1": '{"compound_name":' + '"x",' * 400_000 + "}"},
    }
    assert _measure(oversized, RENDERABLE_MAX_CHARS) > RENDERABLE_MAX_CHARS
    bounded = _bound_output(oversized)
    assert isinstance(bounded, str), "an oversized container must not reach the transcript"
    assert len(bounded) <= PYTHON_OUTPUT_MAX_CHARS
    # It has to say which part was the problem, or the agent cannot act on it.
    assert "dfs" in bounded
    assert "output scope" in bounded

    # A list is caught the same way.
    assert len(_bound_output(["y" * 1000] * 50_000)) <= PYTHON_OUTPUT_MAX_CHARS

    # And so is a frame that never became text at all. This is the one that a
    # size check based on `str()` misses: pandas' repr is the *truncated* view, so
    # a 95 798-row frame reprs to ~400 characters while whatever serializes it
    # downstream writes every row. Both offending cells in the incident ended on
    # one of these — a bare frame, and a dict holding one.
    import pandas as pd

    frame = pd.DataFrame({"compound_name": ["(-)-Huperzine A"] * 95_798, "value": [1.0] * 95_798})
    assert len(repr(frame)) < 1000, "the premise: repr is truncated"
    for shaped in (frame, {"path": "/x.xlsx", "dfs": {"Sheet1": frame}}):
        held = _bound_output(shaped)
        assert isinstance(held, str) and len(held) <= PYTHON_OUTPUT_MAX_CHARS
        assert "too large to keep" in held

    # A frame small enough to be useful in the transcript still comes back whole.
    small_frame = pd.DataFrame({"a": [1, 2, 3]})
    assert _bound_output(small_frame) is small_frame

    # And a structured result that fits is still handed back exactly as it is,
    # because that is what made returning containers worth allowing.
    small = {"rows": 12, "top": ["a", "b"], "score": 0.91}
    assert _bound_output(small) is small
    for value in (None, 42, True, "short", [], {}):
        assert _bound_output(value) == value


def test_every_tool_result_is_capped_before_it_becomes_state() -> None:
    '''The last line of defence, and the one a newly added tool cannot be written without.

    `python_executor` guards its own output, but a per-tool cap is a rule someone has
    to remember. This middleware wraps *every* tool call, so an oversized result
    cannot reach the `messages` channel — which matters more than it looks, because
    the SQLite checkpointer rewrites the whole state as one blob per superstep, so
    one oversized result is stored once per step rather than once.
    '''

    import asyncio

    from langchain_core.messages import ToolMessage
    from langgraph.types import Command

    from app.config import TOOL_RESULT_PERSIST_MAX_CHARS
    from core.agents.context import bound_tool_result

    class _Request:
        tool_call = {"name": "python_executor", "id": "c1"}

    # Async, because a sync-only hook raises under `astream` and every run is streamed.
    hook = bound_tool_result.awrap_tool_call

    def _returning(content):
        async def handler(request):
            return ToolMessage(content=content, tool_call_id="c1", name="python_executor")

        return handler

    async def scenario():
        oversized = "x" * (TOOL_RESULT_PERSIST_MAX_CHARS * 20)
        recorded = await hook(_Request(), _returning(oversized))
        assert len(recorded.content) < TOOL_RESULT_PERSIST_MAX_CHARS * 1.2
        assert "dropped here before this result was recorded" in recorded.content
        # Pairing depends on these surviving the rewrite.
        assert recorded.tool_call_id == "c1" and recorded.name == "python_executor"

        # Anything that fits is untouched.
        kept = await hook(_Request(), _returning("a normal result"))
        assert kept.content == "a normal result"

        # A handoff Command has no text content to bound.
        handoff = Command(goto="research_agent")

        async def hand_off(request):
            return handoff

        assert await hook(_Request(), hand_off) is handoff

    asyncio.run(scenario())


def test_the_cap_is_wired_into_every_agent() -> None:
    '''It is applied in one place, so no agent can be built without it.'''

    import inspect

    from core.agents import agents

    source = inspect.getsource(agents)
    # Exactly one middleware list, and the cap is in it.
    assert source.count("middleware=[") == 1, "agents must be built through one path"
    block = source.split("middleware=[", 1)[1]
    assert "bound_tool_result" in block.split("]", 1)[0]


def test_measuring_an_oversized_result_does_not_rebuild_it() -> None:
    '''`str()` on the thing being guarded against copies it in memory to size it.'''

    import time

    from app.config import PYTHON_OUTPUT_MAX_CHARS
    from core.tools.python_executor import _measure

    huge = {f"k{i}": "v" * 10_000 for i in range(50_000)}
    started = time.monotonic()
    size = _measure(huge, PYTHON_OUTPUT_MAX_CHARS)
    elapsed = time.monotonic() - started

    assert size > PYTHON_OUTPUT_MAX_CHARS
    # It short-circuits, so it sees a few entries rather than all 50 000.
    assert elapsed < 1.0, elapsed


def test_a_run_that_returns_a_whole_dataframe_is_bounded_end_to_end() -> None:
    '''The same guard, reached the way a run reaches it.'''

    from app.config import PYTHON_OUTPUT_MAX_CHARS
    from core.tools.python_executor import python_executor, reset_python_state

    with scoped_roots():
        with scoped_conversation("abc"):
            reset_python_state.invoke({})
            result = python_executor.invoke(
                {"code": "rows = {str(i): 'compound-name-value' for i in range(60000)}\n{'dfs': {'Sheet1': rows}}"}
            )
            assert isinstance(result, str), type(result)
            assert len(result) <= PYTHON_OUTPUT_MAX_CHARS
            assert "too large to keep" in result
            reset_python_state.invoke({})


def test_unsaved_figures_are_rescued() -> None:
    '''A headless server makes plt.show() a no-op, so an unsaved figure is lost.'''

    from core.tools.python_executor import python_executor, reset_python_state

    with scoped_roots() as (results, _data):
        with scoped_conversation("fig"):
            reset_python_state.invoke({})
            result = python_executor.invoke(
                {
                    "code": (
                        "import matplotlib.pyplot as plt\n"
                        "plt.figure()\n"
                        "plt.plot([1,2,3],[2,4,8])\n"
                        "'plotted'"
                    )
                }
            )
            assert "[figures]" in str(result)
            assert list((results / "fig").glob("figure_*.png"))
            reset_python_state.invoke({})


# --- read_files ---------------------------------------------------------------


def test_read_files_resolves_and_bounds() -> None:
    from app.config import READ_FILES_PREVIEW_THRESHOLD_CHARS
    from core.tools.read_files import read_files

    with scoped_roots() as (results, data):
        scope_dir = results / "abc"
        scope_dir.mkdir(parents=True)
        (scope_dir / "small.csv").write_text("a,b\n1,2\n3,4\n")
        (scope_dir / "big.csv").write_text(
            "col\n" + "\n".join(str(index) for index in range(READ_FILES_PREVIEW_THRESHOLD_CHARS))
        )
        upload_dir = data / "abc"
        upload_dir.mkdir(parents=True)
        (upload_dir / "uploaded.csv").write_text("smiles\nCCO\n")

        with scoped_conversation("abc"):
            # A bare name resolves against the repo, then uploads, then outputs.
            assert read_files.invoke({"file_path": "small.csv"}).startswith("a,b")
            assert read_files.invoke({"file_path": "uploaded.csv"}).startswith("smiles")
            assert "Repuragent" in read_files.invoke({"file_path": "requirements.txt"})

            # An explicit range is verbatim and never previewed.
            ranged = read_files.invoke({"file_path": "big.csv", "offset": 3, "limit": 2})
            assert ranged.splitlines()[0].startswith("[read_files] Lines 3-4")
            assert ranged.splitlines()[2:] == ["1", "2"]

            # A large file comes back as a preview that says how to get the rest.
            preview = read_files.invoke({"file_path": "big.csv"})
            assert "[read_files preview]" in preview
            assert "python_executor" in preview and "offset" in preview

            # Reads outside the repo and this conversation are refused.
            assert "Error:" in read_files.invoke({"file_path": "/etc/hosts"})
            assert "Error:" in read_files.invoke({"file_path": "nope.csv"})


def test_read_files_cannot_reach_another_conversation() -> None:
    from core.tools.read_files import read_files

    with scoped_roots() as (results, _data):
        other = results / "u2" / "secret"
        other.mkdir(parents=True)
        target = other / "private.csv"
        target.write_text("confidential\n")

        with scoped_conversation("abc"):
            answer = read_files.invoke({"file_path": str(target)})
            assert "Error:" in answer and "this conversation" in answer


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
