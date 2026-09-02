from __future__ import annotations

import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain_core.documents import Document

from backend.sop_rag.config import BM25_CORPUS_PATH, DOCSTORE_DIR, ID_KEY

CORPUS_VERSION = 1

def load_parent_documents(docstore: Any = None) -> List[Tuple[str, Document]]:
    '''Read every parent document out of the docstore.

    Parameters:
    ---------
    docstore (BaseStore): the parent store, opened from `DOCSTORE_DIR` when omitted.

    Returns:
    ----------
    parents (list): `(id, Document)` pairs, one per stored parent.
    '''

    if docstore is None:
        from langchain_classic.storage import LocalFileStore, create_kv_docstore

        if not Path(DOCSTORE_DIR).exists():
            raise FileNotFoundError(
                f"Parent docstore not found at {DOCSTORE_DIR}. "
                "Run `python -m backend.sop_rag.sop_indexer` first."
            )
        docstore = create_kv_docstore(LocalFileStore(str(DOCSTORE_DIR)))

    keys = sorted(docstore.yield_keys())
    documents = docstore.mget(keys)

    parents: List[Tuple[str, Document]] = []
    for key, document in zip(keys, documents):
        if document is None:
            continue
        metadata = dict(document.metadata or {})
        # An index built before parent ids were minted by the indexer has the id
        # only as the docstore key, and without it fusion cannot dedup.
        metadata.setdefault(ID_KEY, key)
        parents.append((key, Document(page_content=document.page_content, metadata=metadata)))

    return parents


def write_bm25_corpus(
    docstore: Any = None,
    path: Optional[str | Path] = None,
) -> int:
    '''Write the BM25 corpus from whatever is currently in the parent docstore.

    Parameters:
    ---------
    docstore (BaseStore): the parent store, opened from `DOCSTORE_DIR` when omitted.
    path (str): where to write the corpus, defaulting to the configured location.

    Returns:
    ----------
    total (int): how many parents were written.
    '''

    parents = load_parent_documents(docstore)
    if not parents:
        raise RuntimeError(
            "No parent documents found; there is nothing for the BM25 arm to search."
        )

    payload = {
        "corpus_version": CORPUS_VERSION,
        "documents": [
            {"page_content": document.page_content, "metadata": document.metadata}
            for _, document in parents
        ],
    }

    corpus_path = Path(path or BM25_CORPUS_PATH)
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = corpus_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    temporary.replace(corpus_path)

    return len(parents)


def read_bm25_corpus(path: Optional[str | Path] = None) -> List[Document]:
    '''Read the BM25 corpus back as Documents.

    Parameters:
    ---------
    path (str): where the corpus lives, defaulting to the configured location.

    Returns:
    ----------
    documents (list): the parent documents the sparse arm searches.
    '''

    corpus_path = Path(path or BM25_CORPUS_PATH)
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"BM25 corpus not found at {corpus_path}. "
            "Run `python -m backend.sop_rag.sop_indexer --bm25-only` to build it."
        )

    try:
        payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"BM25 corpus at {corpus_path} is unreadable ({exc}); rebuild it with "
            "`python -m backend.sop_rag.sop_indexer --bm25-only`."
        ) from None

    entries: Optional[List[dict]] = payload.get("documents") if isinstance(payload, dict) else payload
    if not entries:
        raise ValueError(
            f"BM25 corpus at {corpus_path} holds no documents; rebuild it with "
            "`python -m backend.sop_rag.sop_indexer --bm25-only`."
        )

    return [
        Document(
            page_content=entry.get("page_content", ""),
            metadata=dict(entry.get("metadata") or {}),
        )
        for entry in entries
    ]


__all__ = ["CORPUS_VERSION", "load_parent_documents", "read_bm25_corpus", "write_bm25_corpus"]
