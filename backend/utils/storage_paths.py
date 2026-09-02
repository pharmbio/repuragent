'''Where persisted data lives on disk.

Three managed roots, all per-conversation below the top level:

    DATA_ROOT/<conversation>/      uploads
    RESULTS_ROOT/<conversation>/   agent outputs (see output_paths)
    MEMORY_ROOT/                   vector stores, the SQLite database

There is no per-user level. The web build partitions both roots by account
(`DATA_ROOT/<user>/<thread>/`); this build has exactly one user, so that level would
be a single constant directory holding everything — a path segment that says nothing
and that every conversation directory would sit under.
'''

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from app.config import DATA_ROOT, MEMORY_ROOT, RESULTS_ROOT

# What a conversation's directory is called when nothing named one.
UNASSIGNED_CONVERSATION = "unassigned"


@lru_cache(maxsize=1)
def get_data_root() -> Path:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    return DATA_ROOT


@lru_cache(maxsize=1)
def get_memory_root() -> Path:
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    return MEMORY_ROOT


@lru_cache(maxsize=1)
def get_results_root() -> Path:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    return RESULTS_ROOT


def thread_folder_name(thread_id: Optional[str]) -> str:
    '''Folder name for a conversation.

    Parameters:
    ---------
    thread_id (str): the conversation id, or None.

    Returns:
    ----------
    name (str): the folder name for that conversation.
    '''

    return thread_id or UNASSIGNED_CONVERSATION


def thread_data_root(thread_id: Optional[str] = None, *, create: bool = True) -> Path:
    '''Upload directory for one conversation.

    Parameters:
    ---------
    thread_id (str): the conversation whose uploads to locate.
    create (boolean): create the directory if it is missing.

    Returns:
    ----------
    root (Path): the upload directory for that conversation.
    '''

    path = get_data_root() / thread_folder_name(thread_id)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "UNASSIGNED_CONVERSATION",
    "get_data_root",
    "get_memory_root",
    "get_results_root",
    "thread_data_root",
    "thread_folder_name",
]
