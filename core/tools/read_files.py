'''Reading a text file without pulling a whole dataset into the transcript.

Repurposing runs produce large tables — every candidate drug, every protein,
every ADMET endpoint. Handing one of those to the model verbatim wastes the
context it needs for reasoning and is replayed on every later call in the run.

So a large file comes back as a **preview envelope** — its size, shape, apparent
type, head and tail, plus the two ways to get the rest — and the agent chooses an
access method deliberately: derive the answer with code in `python_executor` when
it depends on the file as a whole, or read one exact region with `offset`/`limit`
when a specific passage is what matters. An explicitly requested line range is
always returned verbatim, never previewed.
'''

from __future__ import annotations

from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

from app.config import (
    DATA_ROOT,
    MEMORY_ROOT,
    READ_FILES_PREVIEW_HEAD_LINES,
    READ_FILES_PREVIEW_TAIL_LINES,
    READ_FILES_PREVIEW_THRESHOLD_CHARS,
    REPO_ROOT,
    RESULTS_ROOT,
)
from backend.utils.output_paths import (
    DEFAULT_CONVERSATION,
    conversation_output_root,
    get_current_conversation_id,
)
from backend.utils.storage_paths import thread_data_root

# Formats whose content is record-structured or machine-generated. Reading one end
# to end is almost never how the answer comes out of it.
_STRUCTURED_KINDS = {
    ".csv": "CSV (delimited records)",
    ".tsv": "TSV (delimited records)",
    ".json": "JSON",
    ".jsonl": "JSON Lines (one record per line)",
    ".ndjson": "JSON Lines (one record per line)",
    ".xml": "XML",
    ".html": "HTML",
    ".htm": "HTML",
    ".yaml": "YAML",
    ".yml": "YAML",
    ".log": "log file",
    ".sdf": "SDF (chemical records)",
    ".smi": "SMILES list",
    ".mol": "MOL (chemical structure)",
    ".pdb": "PDB (structure)",
    ".fasta": "FASTA sequences",
    ".parquet": "Parquet (binary, columnar)",
    ".pkl": "pickled Python object (binary)",
    ".xlsx": "Excel workbook (binary)",
}


def _scope() -> str:
    return get_current_conversation_id() or DEFAULT_CONVERSATION


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _assert_allowed(path: Path) -> None:
    '''Repository files and this conversation's own files only.

    The managed persistence roots are otherwise off limits, so one conversation
    cannot read another's uploads or results by absolute path.

    Parameters:
    ---------
    path (Path): an already-resolved absolute path to check.
    '''

    resolved = path.resolve()
    conversation_id = _scope()
    data_root = thread_data_root(conversation_id, create=False).resolve()
    output_root = conversation_output_root(conversation_id).resolve()

    if _is_relative_to(resolved, data_root) or _is_relative_to(resolved, output_root):
        return

    if any(_is_relative_to(resolved, root) for root in (DATA_ROOT, RESULTS_ROOT, MEMORY_ROOT)):
        raise PermissionError(
            "Reading persisted data is limited to this conversation's uploads and outputs."
        )

    if not _is_relative_to(resolved, REPO_ROOT):
        raise PermissionError(
            "Reading is limited to repository files and this conversation's own files."
        )


def _resolve(file_path: str) -> Path:
    '''Absolute, or relative to the repo, this conversation's uploads, or its outputs.

    Parameters:
    ---------
    file_path (str): absolute, or relative to the repository, this conversation's uploads, or its outputs.

    Returns:
    ----------
    path (Path): the resolved absolute path.
    '''

    candidate = Path(file_path).expanduser()
    if candidate.is_absolute():
        resolved = candidate.resolve(strict=True)
    else:
        conversation_id = _scope()
        options = [
            REPO_ROOT / candidate,
            thread_data_root(conversation_id, create=False) / candidate,
            conversation_output_root(conversation_id) / candidate,
        ]
        found = next((option for option in options if option.exists()), None)
        if found is None:
            raise FileNotFoundError(file_path)
        resolved = found.resolve()
    _assert_allowed(resolved)
    return resolved


def _describe_kind(path: Path) -> str:
    return _STRUCTURED_KINDS.get(path.suffix.lower(), "")


def _slice_lines(text: str, offset: int, limit: Optional[int]) -> str:
    '''An explicitly requested line range, 1-indexed and inclusive.

    Parameters:
    ---------
    text (str): the whole file contents.
    offset (int): first line to return, 1-indexed.
    limit (int): how many lines to return, or None for the rest of the file.

    Returns:
    ----------
    excerpt (str): the requested range, returned verbatim rather than as a preview.
    '''

    lines = text.splitlines()
    total = len(lines)
    start = max(1, offset) - 1
    if start >= total:
        return (
            f"[read_files] Requested offset {max(1, offset)} is past the end of the "
            f"file ({total:,} lines)."
        )
    end = total if limit is None else min(total, start + max(1, limit))
    header = f"[read_files] Lines {start + 1:,}-{end:,} of {total:,}." + (
        "" if end >= total else f" Continue with offset={end + 1}."
    )
    return f"{header}\n\n" + "\n".join(lines[start:end])


def _preview(path: Path, display_path: str, text: str) -> str:
    lines = text.splitlines()
    total = len(lines)
    head = lines[:READ_FILES_PREVIEW_HEAD_LINES]
    tail = lines[-READ_FILES_PREVIEW_TAIL_LINES:] if total > len(head) else []
    kind = _describe_kind(path)

    parts = [
        f"[read_files preview] {display_path}",
        f"Size: {len(text):,} characters · {total:,} lines"
        + (f" · looks like {kind}" if kind else ""),
        f"Previewed rather than returned whole because it exceeds "
        f"{READ_FILES_PREVIEW_THRESHOLD_CHARS:,} characters.",
        "",
        "To get what you actually need from it:",
        "- Derive the answer with code in `python_executor` (parse, filter, "
        "aggregate, then report the result or write it to a file). This is the "
        "normal path for record-structured files and for anything feeding a table, "
        "ranking or figure — never pull the records themselves into the conversation.",
        "- Read one region exactly with "
        "`read_files(file_path, offset=<first line>, limit=<line count>)`, which is "
        "returned verbatim and never previewed.",
        "",
        f"--- first {len(head):,} lines ---",
        "\n".join(head),
    ]
    if tail:
        parts += [f"--- last {len(tail):,} lines (of {total:,}) ---", "\n".join(tail)]
    return "\n".join(parts)


@tool
def read_files(file_path: str, offset: int = 0, limit: Optional[int] = None):
    '''Read a UTF-8 text file, or one line range of it.

    Accepts an absolute path, a repository-relative path, or a bare filename,
    which is resolved against the repository, then this conversation's uploads,
    then its outputs — so `read_files("candidates.csv")` finds the file the
    workflow just wrote.

    A large file is returned as a preview (size, shape, head and tail) rather than
    in full; pass `offset`/`limit` to read any region exactly, or parse it with
    `python_executor` when the answer depends on the whole file.

    Parameters:
    ---------
    file_path (str): Path or filename to read.
    offset (int): First line to return, 1-indexed. Omit to start at the beginning.
    limit (int): How many lines to return from `offset`. Omit for the rest.

    Returns:
    ----------
    contents (str): the requested range verbatim, or a preview envelope when the file is larger than `READ_FILES_PREVIEW_THRESHOLD_CHARS`.
    '''

    try:
        resolved = _resolve(file_path)
        if not resolved.is_file():
            return f"Error: {file_path} is not a file."
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"Error: {file_path} does not exist."
    except (PermissionError, ValueError) as exc:
        return f"Error: {exc}"
    except UnicodeDecodeError:
        kind = _describe_kind(Path(file_path))
        hint = (
            f" It looks like {kind}; open it in `python_executor` with a library that "
            "understands the format."
            if kind
            else " Read it with `python_executor` if it is a binary format."
        )
        return f"Error: {file_path} is not a UTF-8 text file.{hint}"
    except OSError as exc:
        return f"Error reading {file_path}: {exc}"

    if offset or limit is not None:
        return _slice_lines(text, offset, limit)
    if len(text) > READ_FILES_PREVIEW_THRESHOLD_CHARS:
        return _preview(resolved, file_path, text)
    return text


__all__ = ["read_files"]
