from __future__ import annotations

from pathlib import Path

from app.config import SOP_EMBEDDING_MODEL, SOP_IMAGE_DESCRIPTION_MODEL
from backend.utils.storage_paths import get_data_root, get_memory_root

# Fundamental dir path
DATA_DIR = get_data_root()
MEMORY_DIR = get_memory_root()
SOP_DATA_DIR = DATA_DIR / "SOP"
SOP_MEMORY_DIR = MEMORY_DIR / "sop_documents"

# Dir path for SOP RAG system
INDEX_DIR = SOP_MEMORY_DIR / "ensemble"
CHROMA_PERSIST_PATH = INDEX_DIR / "chroma_db"
DOCSTORE_DIR = INDEX_DIR / "docstore"
BM25_CORPUS_PATH = INDEX_DIR / "bm25_corpus.json"
MANIFEST_PATH = INDEX_DIR / "manifest.json"


def ensure_directories() -> None:
    '''Create the index directories, so a first run has somewhere to write.'''
    for directory in (SOP_DATA_DIR, SOP_MEMORY_DIR, INDEX_DIR, CHROMA_PERSIST_PATH, DOCSTORE_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)


def index_exists() -> bool:
    '''Whether all three parts of the index are present.

    An index is not one artifact. The dense arm needs its Chroma collection *and*
    the parent docstore, the sparse arm is rebuilt at process start from
    `bm25_corpus.json`, and the manifest is what makes the next run incremental. Any
    one of them missing means the retriever cannot be constructed, so all four are
    checked together rather than at the point of use.

    Returns:
    ----------
    exists (boolean): True when every part of a usable index is on disk.
    '''

    return all(
        Path(part).exists()
        for part in (CHROMA_PERSIST_PATH, DOCSTORE_DIR, BM25_CORPUS_PATH, MANIFEST_PATH)
    )


# `doc_id` links a child vector to its parent
ID_KEY = "doc_id"
# `source` is the PDF filename, and is what makes one document's chunks findable.
SOURCE_KEY = "source"

# Models' config
EMBEDDING_MODEL = SOP_EMBEDDING_MODEL
LLM_CONFIG = {
    "image_description_model": SOP_IMAGE_DESCRIPTION_MODEL,
}

# ParentDocumentRetriever child splitter config.
#
# There is deliberately no parent splitter beside it: a parent is exactly one
# `unstructured` section, whatever length that comes out at, so that parent
# boundaries match the sibling ChemSafeAgent pipeline the retrieval defaults
# below were swept on. `PDF_PROCESSING_CONFIG`'s `max_characters` is therefore
# the only bound on a parent, and it is applied before figure descriptions are
# written in — so a figure-heavy section can exceed it.
CHILD_SPLITTER_CONFIG = {
    "chunk_size": 200,
    "chunk_overlap": 50,
}


def _slug(name: str) -> str:
    '''Filesystem- and collection-safe form of a model id.'''
    normalized = name.replace("/", "_").replace(":", "_").replace(".", "_")
    return normalized


COLLECTION_NAME = (
    f"sop_rag_{_slug(SOP_EMBEDDING_MODEL)}"
    f"_c{CHILD_SPLITTER_CONFIG['chunk_size']}_o{CHILD_SPLITTER_CONFIG['chunk_overlap']}"
)

# `unstructured` hi-res partitioning config
PDF_PROCESSING_CONFIG = {
    "strategy": "hi_res",
    "infer_table_structure": True,
    "extract_image_block_types": ["Image"],
    "extract_image_block_to_payload": True,
    "chunking_strategy": "by_title",
    "max_characters": 10000,
    "combine_text_under_n_chars": 2000,
    "new_after_n_chars": 6000,
}

PDF_FALLBACK_STRATEGY = "fast"


# EnsembleRetriever config
ENSEMBLE_CONFIG = {
    "bm25_weight": 0.6,
    "dense_weight": 0.4,
    "bm25_k1": 1.0,
    "bm25_b": 0.75,
    "rrf_c": 60,
}

RETRIEVAL_CONFIG = {
    "search_type": "similarity",
    "default_score_threshold": 0.0,
    "fetch_k": 20,
    "max_results": 3,
    "fuse_func": "combsum",
}
