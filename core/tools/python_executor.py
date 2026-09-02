'''The data agent's work engine: Python in a restricted interpreter.

Code runs through `backend/utils/local_python_executor.py` (an AST-walking
interpreter adapted from smolagents), not `exec`, and only modules in
`DEFAULT_AUTHORIZED_IMPORTS` can be imported.

Three properties matter more than the sandbox itself:

* **State is per conversation.** Interpreters are keyed by conversation with a
  per-session lock, so two conversations never share a namespace and
  `reset_python_state` clears only the caller's session. It used to be one
  namespace for the whole process.
* **Writes are scoped.** Every call is handed `output_root`,
  `prepare_output_path`, `ensure_output_dir` and a sandboxed `open`, and a write
  outside the conversation's output directory raises.
* **Figures are not lost.** The server is headless, so `plt.show()` does nothing;
  a figure the run leaves unsaved is written into the output scope and its path
  reported back.

Output is capped, because a tool result is replayed to the model on every later
call in the run.
'''

from __future__ import annotations

import builtins
import os
from pathlib import Path
from typing import Any, Dict

from langchain_core.tools import tool

from app.config import (
    DATA_ROOT,
    FIGURE_AUTOSAVE,
    MEMORY_ROOT,
    PYTHON_EXEC_TIMEOUT_SECONDS,
    PYTHON_OUTPUT_MAX_CHARS,
    PYTHON_SESSION_CACHE_SIZE,
    REPO_ROOT,
    RESULTS_ROOT,
)
from backend.utils import figure_capture
from backend.utils.cancellation import ExecutionCancelled as ToolCancelled
from backend.utils.cancellation import cancel_event
from backend.utils.local_python_executor import (
    BASE_BUILTIN_MODULES,
    ExecutionCancelled,
    ExecutionTimeout,
    InterpreterError,
    local_python_executor,
    reset_executor_state,
    set_max_executor_sessions,
)
from backend.utils.output_paths import (
    DEFAULT_CONVERSATION,
    conversation_output_root,
    describe_output_scope,
    get_current_conversation_id,
    resolve_output_folder,
    task_file_path,
)
from backend.utils.storage_paths import thread_data_root

set_max_executor_sessions(PYTHON_SESSION_CACHE_SIZE)
if FIGURE_AUTOSAVE:
    figure_capture.install()


DEFAULT_AUTHORIZED_IMPORTS = [
    # the repository itself, so the agent can reuse project code
    "app",
    "backend",
    "core",
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
    "rdkit",
    "pybel",
    "networkx",
    "openpyxl",
    "chembl_webresource_client",
    "pubchempy",
    "Bio",
    "asyncio",
    "dotenv",
    "fuzzywuzzy",
    "httpx",
    "json",
    "ntpath",
    "os",
    "pathlib",
    "posixpath",
    "requests",
    "sqlalchemy",
    "sys",
    "textwrap",
    "pickle",
    "inspect",
]
AUTHORIZED_IMPORTS = sorted(set(BASE_BUILTIN_MODULES) | set(DEFAULT_AUTHORIZED_IMPORTS))


def _session_key() -> str:
    '''Interpreter identity: one namespace per conversation.

    Returns:
    ----------
    key (str): the conversation id, so no two conversations share an interpreter.
    '''

    return get_current_conversation_id() or DEFAULT_CONVERSATION


def _measure(value, budget: int) -> int:
    '''Roughly how many characters `value` renders to, giving up once over `budget`.

    Walks the structure instead of calling `str()` on it, because `str()` on the
    thing this guards against — a 95 000-row DataFrame that a run turned into JSON
    inside a dict — builds a second copy of it in memory just to find out it is too
    big. The number is approximate; it is only ever compared against a threshold.

    Parameters:
    ---------
    value (any): whatever the interpreter returned, at any depth.
    budget (int): stop counting once the total passes this.

    Returns:
    ----------
    size (int): the rendered size, or any value greater than `budget` once it is exceeded.
    '''

    if isinstance(value, str):
        return len(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, dict):
        total = 2
        for key, item in value.items():
            total += _measure(key, budget) + _measure(item, budget) + 4
            if total > budget:
                return total
        return total
    if isinstance(value, (list, tuple, set, frozenset)):
        total = 2
        for item in value:
            total += _measure(item, budget) + 2
            if total > budget:
                return total
        return total

    # A DataFrame, Series or ndarray is sized from its shape, never from `repr`:
    # pandas' repr is the *truncated* view, so a 95 798-row frame reprs to ~400
    # characters and would sail under the cap — while whatever serializes it
    # downstream expands every row. Roughly a dozen characters per cell is enough
    # precision for a threshold test.
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape and all(isinstance(dim, int) for dim in shape):
        cells = 1
        for dim in shape:
            cells *= max(dim, 1)
        return cells * 12

    return len(repr(value))


def _describe_shape(value) -> str:
    '''One line naming what an oversized container held, so the size is actionable.

    "168 MB of output" tells the agent nothing it can act on. "dict with keys path,
    sheets, dfs — dfs is 143 MB" tells it exactly which part to write to a file.

    Parameters:
    ---------
    value (any): the oversized result.

    Returns:
    ----------
    shape (str): a description of the container and, for a dict, the size of each entry.
    '''

    if isinstance(value, dict):
        entries = sorted(
            ((_measure(item, PYTHON_OUTPUT_MAX_CHARS * 64), str(key)) for key, item in value.items()),
            reverse=True,
        )
        rendered = ", ".join(f"{key} (~{size:,} chars)" for size, key in entries[:8])
        more = f", and {len(entries) - 8} more" if len(entries) > 8 else ""
        return f"a dict of {len(value)} entries: {rendered}{more}"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"a {type(value).__name__} of {len(value)} items"
    shape = getattr(value, "shape", None)
    if isinstance(shape, tuple) and shape:
        return f"a {type(value).__name__} of shape {shape}"
    return f"a {type(value).__name__}"


# Above this, a result is described rather than rendered: `str()` on it would
# rebuild the whole object in memory, which is the cost this guard exists to avoid.
# Below it, rendering is cheap and the caller is better served by seeing the data.
RENDERABLE_MAX_CHARS = PYTHON_OUTPUT_MAX_CHARS * 50

# What the agent is told to do instead. The same advice for both shapes, because
# the remedy is the same one: the output scope is the right place for bulk data,
# and `read_files` reads it back in bounded pieces.
_TOO_BIG_ADVICE = (
    "Narrow the selection, aggregate, or write the full result to a file under the "
    "output scope with prepare_output_path(...) and read it back in pieces."
)


def _bound_output(result):
    '''Keep one tool result from dominating the transcript — and the checkpoint.

    The middle of an oversized dump is dropped rather than the tail, because the
    tail usually holds the answer.

    **Containers are bounded too, and that is the whole point of this function.**
    They used to be returned untouched on the reasoning that a structured result
    should keep its structure. What that actually bought was an unbounded write
    path: a run that returned `{"dfs": {"Sheet1": df.to_json()}}` for a 95 798-row
    frame put a 143 MB `ToolMessage` into the message list, and LangGraph's SQLite
    checkpointer stores the whole state as one blob per superstep — so those two
    messages were rewritten 126 times and cost 20.6 GB. The cap has to apply to
    what is *stored*, not to what happens to be a `str`.

    Parameters:
    ---------
    result (any): whatever the interpreter returned.

    Returns:
    ----------
    bounded (any): the result unchanged when it fits; otherwise text whose middle — or, for a container, whose bulk — is replaced by a note saying what was dropped and what to do instead.
    '''

    if result is None:
        return result

    if not isinstance(result, str):
        # Everything non-textual is measured, not stringified. A bare DataFrame is
        # the case that matters: `str(df)` is the truncated pandas view, so sizing
        # it that way would let a 95 798-row frame through as ~400 characters.
        measured = _measure(result, RENDERABLE_MAX_CHARS)
        if measured <= PYTHON_OUTPUT_MAX_CHARS:
            return result
        if measured > RENDERABLE_MAX_CHARS:
            # Rendering this one just to elide it would rebuild the whole thing in
            # memory, which is the cost being avoided. Its shape is what the caller
            # can act on anyway — the first 20 000 characters of a serialised
            # 95 798-row frame are 20 000 characters of the same repeated column.
            return (
                f"[This result was too large to keep: {_describe_shape(result)}, over the "
                f"{PYTHON_OUTPUT_MAX_CHARS:,}-character limit for one python_executor "
                f"result. Nothing was written to disk and your session still holds the "
                f"value — it was only kept out of the transcript.] {_TOO_BIG_ADVICE}"
            )
        # Merely oversized: small enough to render, so it is elided like any other
        # text rather than replaced by a description. A structured result a little
        # over the limit still has readable content at both ends, and dropping it
        # for a one-line summary would tell the caller less than the old, unbounded
        # behaviour did.
        text = str(result)
    else:
        text = result

    if len(text) <= PYTHON_OUTPUT_MAX_CHARS:
        return result
    head = int(PYTHON_OUTPUT_MAX_CHARS * 0.6)
    tail = PYTHON_OUTPUT_MAX_CHARS - head
    omitted = len(text) - head - tail
    return (
        f"{text[:head]}\n\n"
        f"... [{omitted:,} characters omitted from the middle of this output. "
        f"{_TOO_BIG_ADVICE}] ...\n\n"
        f"{text[-tail:]}"
    )


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_managed_persistence_path(path: Path) -> bool:
    return any(_is_relative_to(path, root) for root in (DATA_ROOT, RESULTS_ROOT, MEMORY_ROOT))


def _mode_requires_write(mode: str) -> bool:
    return any(flag in mode for flag in ("w", "a", "x", "+"))


def _build_execution_context() -> Dict[str, Any]:
    '''Values injected into the interpreter namespace on every call.

    Returns:
    ----------
    context (dict): `output_root`, `prepare_output_path`, `ensure_output_dir` and the sandboxed `open`, bound to this conversation's scope.
    '''

    conversation_id = get_current_conversation_id() or DEFAULT_CONVERSATION
    repo_root = REPO_ROOT.resolve()
    output_root = conversation_output_root(conversation_id).resolve()
    data_root = thread_data_root(conversation_id, create=False).resolve()

    def ensure_output_dir(subfolder: str = "") -> str:
        return str(resolve_output_folder(subfolder or None, conversation_id=conversation_id))

    def prepare_output_path(filename: str, subfolder: str = "") -> str:
        folder = resolve_output_folder(subfolder or None, conversation_id=conversation_id)
        return str(
            task_file_path(filename, output_folder=folder, conversation_id=conversation_id)
        )

    def scoped_open(
        file,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
        closefd: bool = True,
        opener=None,
    ):
        '''`open`, restricted to the repository and this conversation's files.

        Reads resolve against the repo, then the conversation's uploads, then its
        outputs — so `open("candidates.csv")` finds the upload the user just made
        without an absolute path. Writes may only land in the output scope.

        Parameters:
        ---------
        file (str or Path): path to open, resolved against the repository, this conversation's uploads, then its outputs.
        mode (str): as builtin `open`; any write mode is confined to the output scope.
        buffering (int): as builtin `open`.
        encoding (str): as builtin `open`.
        errors (str): as builtin `open`.
        newline (str): as builtin `open`.
        closefd (boolean): as builtin `open`.
        opener (callable): as builtin `open`.

        Returns:
        ----------
        handle (file object): the opened file, or a `PermissionError` when the path escapes the scope.
        '''

        if not isinstance(mode, str) or not mode:
            raise ValueError("open() mode must be a non-empty string.")
        if opener is not None:
            raise ValueError("Custom file openers are not supported in python_executor.")

        candidate = Path(os.fspath(file)).expanduser()
        writing = _mode_requires_write(mode)

        if candidate.is_absolute():
            resolved = candidate.resolve()
        elif writing:
            resolved = (output_root / candidate).resolve()
        else:
            options = [
                (repo_root / candidate).resolve(),
                (data_root / candidate).resolve(),
                (output_root / candidate).resolve(),
            ]
            resolved = next((option for option in options if option.exists()), options[-1])

        if writing:
            if not _is_relative_to(resolved, output_root):
                raise ValueError(f"Write access is limited to the output scope: {output_root}")
            resolved.parent.mkdir(parents=True, exist_ok=True)
        else:
            allowed = (
                _is_relative_to(resolved, data_root)
                or _is_relative_to(resolved, output_root)
                or (not _is_managed_persistence_path(resolved) and _is_relative_to(resolved, repo_root))
            )
            if not allowed:
                raise ValueError(
                    "Read access is limited to the repository and this conversation's "
                    f"files: {repo_root}, {data_root}, {output_root}"
                )

        return builtins.open(
            resolved,
            mode=mode,
            buffering=buffering,
            encoding=encoding,
            errors=errors,
            newline=newline,
            closefd=closefd,
        )

    return {
        "conversation_id": conversation_id,
        "output_root": str(output_root),
        "output_root_path": output_root,
        "output_scope": describe_output_scope(conversation_id=conversation_id),
        "ensure_output_dir": ensure_output_dir,
        "prepare_output_path": prepare_output_path,
        "open": scoped_open,
        "Path": Path,
    }


@tool
def python_executor(code: str):
    '''Run Python for data work: inspection, analysis, scoring, plotting, file output.

    Variables, imports and functions persist across calls within this conversation,
    so build up state in small, inspectable steps rather than one large script.

    Every call is given, already scoped to this conversation:
    `output_root`, `output_root_path`, `output_scope`, `conversation_id`,
    `prepare_output_path(filename, subfolder="")`,
    `ensure_output_dir(subfolder="")`, and an `open` that can only write inside the
    output scope. Use those helpers for any file you produce.

    Matplotlib figures you do not save yourself are saved into the output scope and
    their paths reported back — this server is headless, so `plt.show()` displays
    nothing.

    **What the cell evaluates to is what gets stored**, so end it on a summary, not
    on data. Inspect with `print(df.shape)`, `df.head()` and `df.columns.tolist()`;
    keep whole tables in session variables and write them to the output scope. A
    cell that ends on a frame — or on a dict or list holding one — is asking for the
    entire table to be kept in the transcript, and an oversized result is replaced
    by a note rather than stored.

    Parameters:
    ---------
    code (str): The Python to execute.

    Returns:
    ----------
    result (any): The value of the last statement, or its printed output. On failure, a dict
    with `ok: false`, the error type, the message and a recovery hint.
    '''

    session_key = _session_key()
    context = _build_execution_context()
    before_figures = figure_capture.snapshot_open_figures() if FIGURE_AUTOSAVE else ()

    try:
        result = local_python_executor(
            code,
            AUTHORIZED_IMPORTS,
            variables=context,
            session_key=session_key,
            timeout=PYTHON_EXEC_TIMEOUT_SECONDS or None,
            cancel_event=cancel_event(),
        )
    except ExecutionCancelled:
        figure_capture.forget_session(session_key)
        # Propagate: the run is being torn down, so there is nobody to hand a
        # recovery hint to.
        raise ToolCancelled("Python execution stopped at your request.")
    except ExecutionTimeout as exc:
        figure_capture.forget_session(session_key)
        return {
            "ok": False,
            "error_type": "ExecutionTimeout",
            "error": str(exc),
            "input_code": code,
            "recovery": (
                "The Python session was reset, so variables from before the timeout "
                "are gone. Split the work into smaller calls, narrow the query, pass "
                "explicit timeouts to network calls, and write intermediate results "
                "to files under the output scope so progress survives."
            ),
        }
    except InterpreterError as exc:
        return {
            "ok": False,
            "error_type": "InterpreterError",
            "error": str(exc),
            "input_code": code,
            "authorized_imports": AUTHORIZED_IMPORTS,
        }

    if FIGURE_AUTOSAVE:
        saved = figure_capture.capture_unsaved_figures(
            session_key=session_key,
            before=before_figures,
            prepare_output_path=context["prepare_output_path"],
        )
        if saved:
            listing = "\n".join(f"- {path}" for path in saved)
            note = (
                "\n\n[figures] These figures were still unsaved, so they were written "
                f"to the output scope for you:\n{listing}\n"
                "Reference these paths, or call savefig yourself to control the "
                "filename and format."
            )
            result = (result if isinstance(result, str) else str(result or "")) + note

    return _bound_output(result)


@tool
def reset_python_state():
    '''Clear this conversation's Python session: all variables and definitions.

    Use it when accumulated state has become misleading — a stale DataFrame, a
    shadowed name, a repeated failure that survives your fixes. Only this
    conversation's interpreter is affected.

    Returns:
    ----------
    message (str): confirmation that this conversation's variables and definitions are gone.
    '''

    session_key = _session_key()
    reset_executor_state(session_key)
    figure_capture.forget_session(session_key)
    return "Python session reset. All previously defined variables and functions are gone."


__all__ = ["AUTHORIZED_IMPORTS", "python_executor", "reset_python_state"]
