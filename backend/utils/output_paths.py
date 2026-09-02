'''The output scope: where the agents are allowed to write, and what they wrote.

Every artifact a run produces lands in `RESULTS_ROOT/<conversation>/`. Which
conversation is carried in a **contextvar**, so a tool deep inside a graph node can
resolve its scope without every layer threading another argument through.

Note the asymmetry that this creates and that has bitten before: **graph nodes read
the scope from graph state, tools read it from the contextvar.** When the two
disagree the plan file is written in one place and looked for in another, so the
coroutine driving a run must pin the scope into an explicit `contextvars.Context`
(see `app/run_controller.py::build_conversation_context`) rather than merely setting
it inside a generator.
'''

from __future__ import annotations

import contextvars
import shutil
import time
from pathlib import Path
from typing import List, Optional

from app.config import RESULTS_ROOT
from backend.utils.storage_paths import thread_folder_name

DEFAULT_CONVERSATION = "default-thread"

_conversation_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "repuragent_conversation_id",
    default=None,
)


def get_results_root() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


# --- Scope --------------------------------------------------------------------


def set_current_conversation_id(conversation_id: Optional[str]):
    if conversation_id is None:
        return None
    return _conversation_id_var.set(conversation_id)


def reset_current_conversation_id(token) -> None:
    if token is None:
        _conversation_id_var.set(None)
        return
    try:
        _conversation_id_var.reset(token)
    except ValueError:
        # Token minted in a different asyncio context; clear instead.
        _conversation_id_var.set(None)


def get_current_conversation_id() -> Optional[str]:
    return _conversation_id_var.get()


# Legacy aliases: earlier versions called a conversation a "task".
set_current_task_id = set_current_conversation_id
reset_current_task_id = reset_current_conversation_id
get_current_task_id = get_current_conversation_id


# --- Directories --------------------------------------------------------------


def _resolve_conversation(conversation_id: Optional[str] = None) -> str:
    return conversation_id or get_current_conversation_id() or DEFAULT_CONVERSATION


def conversation_output_root(conversation_id: Optional[str] = None) -> Path:
    '''The one directory this conversation may write into.

    Parameters:
    ---------
    conversation_id (str): the conversation, defaulting to the ambient scope.

    Returns:
    ----------
    root (Path): the one directory this conversation may write into.
    '''

    path = get_results_root() / thread_folder_name(_resolve_conversation(conversation_id))
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_folder(
    preferred_folder: Optional[str] = None,
    *,
    conversation_id: Optional[str] = None,
) -> Path:
    '''Clamp a requested folder back inside the conversation's output scope.

    A relative path is taken as a subfolder of the scope; an absolute path or one
    that climbs out with `..` is refused and the scope root is returned instead.
    This is the only place that decides where a write may land, so tool code must go
    through it rather than building paths itself.

    Parameters:
    ---------
    preferred_folder (str): the folder the caller asked for, which may try to escape.
    conversation_id (str): the conversation, defaulting to the ambient scope.

    Returns:
    ----------
    folder (Path): the requested folder clamped back inside the scope root. Tool code must go through this rather than building paths itself.
    '''

    base_dir = conversation_output_root(conversation_id)
    if not preferred_folder:
        return base_dir

    candidate = Path(preferred_folder)
    if not candidate.is_absolute():
        parts = [part for part in candidate.parts if part not in ("", ".")]
        # Tolerate a hint that repeats the scope root, e.g. "results/figures".
        while parts and parts[0] in {base_dir.name, get_results_root().name}:
            parts.pop(0)
        candidate = base_dir / Path(*parts) if parts else base_dir

    try:
        candidate.resolve().relative_to(base_dir.resolve())
    except ValueError:
        return base_dir

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def task_file_path(
    filename: str,
    *,
    output_folder: Optional[str | Path] = None,
    conversation_id: Optional[str] = None,
) -> Path:
    '''Path for one file inside the conversation's output scope.

    Parameters:
    ---------
    filename (str): the file to place.
    output_folder (str | Path): a subfolder inside the scope, or None for its root.
    conversation_id (str): the conversation, defaulting to the ambient scope.

    Returns:
    ----------
    path (Path): the absolute path for that file inside the conversation's output scope.
    '''

    if isinstance(output_folder, Path):
        folder = output_folder
    else:
        folder = resolve_output_folder(
            str(output_folder) if output_folder else None,
            conversation_id=conversation_id,
        )
    folder.mkdir(parents=True, exist_ok=True)
    return folder / filename


def list_task_files(conversation_id: str) -> List[Path]:
    '''Every output file of one conversation, newest first.

    Parameters:
    ---------
    conversation_id (str): the conversation whose outputs to list.

    Returns:
    ----------
    paths (list): every output file of that conversation, newest first.
    '''

    directory = get_results_root() / thread_folder_name(_resolve_conversation(conversation_id))
    if not directory.exists():
        return []
    files = [path for path in directory.rglob("*") if path.is_file()]

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            # The agent may replace a file between listing and stat.
            return 0.0

    files.sort(key=_mtime, reverse=True)
    return files


def remove_task_dir(conversation_id: str) -> None:
    directory = get_results_root() / thread_folder_name(_resolve_conversation(conversation_id))
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)


# --- What the model is told about the scope ------------------------------------


def describe_output_scope(*, conversation_id: Optional[str] = None) -> str:
    resolved_conversation = _resolve_conversation(conversation_id)
    root = conversation_output_root(resolved_conversation)
    return (
        "Active output scope:\n"
        f"- conversation_id: {resolved_conversation}\n"
        f"- output_root: {root}\n"
        "Every generated file goes under this directory. In python_executor use "
        "the injected helpers `prepare_output_path(filename)` and "
        "`ensure_output_dir(subfolder)` rather than composing paths yourself."
    )


_ARTIFACT_CACHE_TTL_SECONDS = 2.0
_artifact_cache: dict[tuple, tuple[float, str]] = {}


def describe_output_artifacts(
    *,
    conversation_id: Optional[str] = None,
    max_items: int = 25,
) -> str:
    '''The files this conversation has already produced, newest first.

    Read from disk rather than from a list the model maintains, so a follow-up turn
    can still refer to earlier artifacts after the narrative summary has dropped
    them. Briefly cached because this runs before every model call and a run that
    writes many files would otherwise re-walk the tree each time; the TTL is short
    enough that a file written mid-run still shows up.

    Parameters:
    ---------
    conversation_id (str): the conversation to describe, defaulting to the ambient scope.
    max_items (int): how many files to list.

    Returns:
    ----------
    ledger (str): the artifact ledger pinned into context — paths with byte counts, newest first.
    '''

    resolved_conversation = _resolve_conversation(conversation_id)

    cache_key = (resolved_conversation, max_items)
    now = time.monotonic()
    cached = _artifact_cache.get(cache_key)
    if cached is not None and now - cached[0] < _ARTIFACT_CACHE_TTL_SECONDS:
        return cached[1]

    try:
        root = conversation_output_root(resolved_conversation)
        entries = [path for path in root.rglob("*") if path.is_file()]
    except OSError:
        _artifact_cache[cache_key] = (now, "")
        return ""
    if not entries:
        _artifact_cache[cache_key] = (now, "")
        return ""

    def _mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    entries.sort(key=_mtime, reverse=True)
    lines = []
    for path in entries[:max_items]:
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        lines.append(f"- {path} ({size:,} bytes)")
    remaining = len(entries) - max_items
    if remaining > 0:
        lines.append(f"- ... and {remaining} more file(s) in this output scope")

    rendered = "\n".join(lines)
    _artifact_cache[cache_key] = (now, rendered)
    return rendered


__all__ = [
    "DEFAULT_CONVERSATION",
    "conversation_output_root",
    "describe_output_artifacts",
    "describe_output_scope",
    "get_current_conversation_id",
    "get_current_task_id",
    "get_results_root",
    "list_task_files",
    "remove_task_dir",
    "reset_current_conversation_id",
    "reset_current_task_id",
    "resolve_output_folder",
    "set_current_conversation_id",
    "set_current_task_id",
    "task_file_path",
]
