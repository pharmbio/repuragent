"""Utility helpers for managing per-conversation result directories."""

from __future__ import annotations

import contextvars
import shutil
from pathlib import Path
from typing import List, Optional

RESULTS_ROOT = Path("results")
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

_task_id_var: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "repuragent_task_id",
    default=None,
)


def get_results_root() -> Path:
    """Return the root results directory, ensuring it exists."""
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def set_current_task_id(task_id: Optional[str]):
    """Push the active task/conversation id into a context variable."""
    if task_id is None:
        return None
    return _task_id_var.set(task_id)


def reset_current_task_id(token) -> None:
    """Reset the task context using a token returned from set_current_task_id."""
    if token is None:
        return
    _task_id_var.reset(token)


def get_current_task_id() -> Optional[str]:
    """Get the task id currently bound to this execution context."""
    return _task_id_var.get()


def ensure_task_dir(task_id: Optional[str] = None) -> Path:
    """Return the directory for the provided (or current) task, creating it if needed."""
    tid = task_id or get_current_task_id()
    if not tid:
        return get_results_root()
    path = get_results_root() / tid
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_folder(
    preferred_folder: Optional[str] = None,
    *,
    task_id: Optional[str] = None,
) -> Path:
    """
    Resolve an output directory that stays scoped under the results root.

    Args:
        preferred_folder: Optional folder hint (absolute or relative). Relative paths are
            always resolved beneath the results root. Absolute paths that escape the root
            are ignored for safety.
        task_id: Explicit task id override.
    """
    base_dir = ensure_task_dir(task_id)
    if not preferred_folder:
        return base_dir

    candidate = Path(preferred_folder)
    results_root = get_results_root()

    if not candidate.is_absolute():
        parts = list(candidate.parts)
        root_name = results_root.name
        while parts and parts[0] in ("", ".", root_name):
            parts.pop(0)
        if parts:
            candidate = results_root / Path(*parts)
        else:
            candidate = results_root

    try:
        candidate.relative_to(results_root)
    except ValueError:
        # Never allow writes outside of the managed results directory.
        return base_dir

    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def task_file_path(
    filename: str,
    *,
    output_folder: Optional[Path | str] = None,
    task_id: Optional[str] = None,
) -> Path:
    """Build a file path inside the active task's directory (or provided folder)."""
    if isinstance(output_folder, str):
        folder_path = resolve_output_folder(output_folder, task_id=task_id)
    elif isinstance(output_folder, Path):
        folder_path = output_folder
    else:
        folder_path = ensure_task_dir(task_id)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path / filename


def list_task_files(task_id: str) -> List[Path]:
    """List files that belong to a task, newest first."""
    directory = get_results_root() / task_id
    if not directory.exists():
        return []
    files = [path for path in directory.rglob("*") if path.is_file()]
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def remove_task_dir(task_id: str) -> None:
    """Remove every artifact for a task."""
    directory = get_results_root() / task_id
    if directory.exists():
        shutil.rmtree(directory, ignore_errors=True)
