#!/usr/bin/env python3
'''Re-run the SOP indexer from the folder the SOP PDFs live in.

Drop a new PDF in beside this script and run it:

    python reindex.py                    # index whatever is new, edited or deleted
    python reindex.py --dry-run          # say what would change, write nothing
    python reindex.py --files new.pdf    # just this one document
    python reindex.py --rebuild          # discard the index and start again
    python reindex.py --no-images        # skip figure descriptions (no LLM calls)
    python reindex.py --bm25-only        # rebuild only the keyword arm's corpus

The work all happens in `backend.sop_rag.sop_indexer`; this exists so that adding
a document does not also mean remembering a module path and a working directory.
Two things have to be arranged before that import will work, and both are why the
script is here rather than being a shell alias:

    1. the repository root has to be on `sys.path`, because `backend.sop_rag`
       imports `app.config`;
    2. `OPENAI_API_KEY` has to be loaded from the repository's `.env`, which
       `app.config` does on import — but only if it can find the file, and it
       locates it relative to itself, so this only works once (1) is done.

It indexes the folder this script is in, not `$DATA_ROOT/SOP`, so a copy of the
corpus somewhere else can be indexed by putting a copy of this script beside it.
'''

from __future__ import annotations

import sys
from pathlib import Path

SOP_DIR = Path(__file__).resolve().parent


def _find_repo_root() -> Path:
    '''Walk up from this script until the repository root is recognisable.

    Returns:
    ----------
    root (Path): the first parent directory holding `app/config.py`.
    '''

    for candidate in SOP_DIR.parents:
        if (candidate / "app" / "config.py").exists():
            return candidate
    raise SystemExit(
        f"Could not find the repository root above {SOP_DIR}. "
        "Run this from inside a checkout, or use `python -m backend.sop_rag.sop_indexer`."
    )


def main() -> int:
    '''Hand this script's arguments to the indexer, with the SOP folder set to this one.

    Returns:
    ----------
    status (int): the indexer's exit status.
    '''

    repo_root = _find_repo_root()
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from backend.sop_rag.sop_indexer import main as index_main

    argv = sys.argv[1:]
    # Only when the caller has not named a directory of their own: --directory
    # stays usable for indexing some other folder from here.
    if not any(argument.startswith("--directory") for argument in argv):
        argv = ["--directory", str(SOP_DIR), *argv]

    return index_main(argv)


if __name__ == "__main__":
    sys.exit(main())
