from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from tempfile import gettempdir
from typing import Any, Dict, Iterable, List, Optional, Tuple
from uuid import uuid4

TMP_ROOT = Path(gettempdir()) / "repuragent-sop-index"
TMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(TMP_ROOT / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(TMP_ROOT / "xdg-cache"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(TMP_ROOT / "numba-cache"))

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import OPENAI_API_KEY
from backend.sop_rag.config import (
    BM25_CORPUS_PATH,
    CHILD_SPLITTER_CONFIG,
    CHROMA_PERSIST_PATH,
    COLLECTION_NAME,
    DOCSTORE_DIR,
    EMBEDDING_MODEL,
    ID_KEY,
    INDEX_DIR,
    LLM_CONFIG,
    MANIFEST_PATH,
    PDF_FALLBACK_STRATEGY,
    PDF_PROCESSING_CONFIG,
    SOP_DATA_DIR,
    SOURCE_KEY,
    ensure_directories,
)
from backend.sop_rag.bm25_corpus import write_bm25_corpus

MANIFEST_VERSION = 6
PARENT_BATCH = 100
IMAGE_DESCRIBE_CONCURRENCY = 4
MIN_IMAGE_PAYLOAD_CHARS = 6000
IMAGE_SKIP_TOKEN = "SKIP"
_PLACEHOLDER = re.compile(r"\[[^\]]{2,60}\]")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Discovery and fingerprinting
def discover_pdf_files(directory: str | Path) -> List[Path]:
    '''Discover all PDF files in the specified directory.'''
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory {directory} does not exist")

    pdfs = [p for p in directory_path.glob("*.pdf") if p.is_file() and not p.name.startswith(".")]
    return sorted(pdfs)


def file_digest(path: str | Path) -> str:
    '''Content hash of one file, which is what decides whether it needs re-indexing.'''

    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


# The manifest
def empty_manifest() -> Dict[str, Any]:
    '''A manifest describing an index that holds nothing yet.'''
    return {
        "manifest_version": MANIFEST_VERSION,
        "embedding_model": EMBEDDING_MODEL,
        "collection_name": COLLECTION_NAME,
        "child_splitter": dict(CHILD_SPLITTER_CONFIG),
        "updated_at": None,
        "sources": {},
    }


def load_manifest(path: Optional[str | Path] = None) -> Dict[str, Any]:
    '''Read the manifest, or return an empty one when there is no index yet.

    Parameters:
    ---------
    path (str): where the manifest lives, defaulting to the configured location.

    Returns:
    ----------
    manifest (dict): what is currently indexed, or an empty manifest.
    '''

    manifest_path = Path(path or MANIFEST_PATH)
    if not manifest_path.exists():
        return empty_manifest()

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise RuntimeError(
            f"Manifest at {manifest_path} is unreadable ({exc}). "
            "Re-run with --rebuild to start again."
        ) from None

    manifest.setdefault("sources", {})
    return manifest


def save_manifest(manifest: Dict[str, Any], path: Optional[str | Path] = None) -> None:
    '''Write the manifest atomically, so an interrupted run cannot truncate it.'''

    manifest["updated_at"] = _now()
    manifest_path = Path(path or MANIFEST_PATH)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(manifest_path)


def manifest_settings_conflict(manifest: Dict[str, Any]) -> Optional[str]:
    '''Describe how a manifest's index-wide settings disagree with the current config.

    Parameters:
    ---------
    manifest (dict): the manifest read from disk.

    Returns:
    ----------
    conflict (str): what disagrees, or None when the settings match or nothing is indexed yet.
    '''

    if not manifest.get("sources"):
        return None

    mismatches = []
    if int(manifest.get("manifest_version") or 0) != MANIFEST_VERSION:
        mismatches.append(
            f"index format {manifest.get('manifest_version')} -> {MANIFEST_VERSION}"
        )
    if manifest.get("embedding_model") != EMBEDDING_MODEL:
        mismatches.append(
            f"embedding model {manifest.get('embedding_model')!r} -> {EMBEDDING_MODEL!r}"
        )
    if manifest.get("collection_name") != COLLECTION_NAME:
        mismatches.append(
            f"collection {manifest.get('collection_name')!r} -> {COLLECTION_NAME!r}"
        )
    if dict(manifest.get("child_splitter") or {}) != dict(CHILD_SPLITTER_CONFIG):
        mismatches.append(
            f"child splitter {manifest.get('child_splitter')} -> {dict(CHILD_SPLITTER_CONFIG)}"
        )

    if not mismatches:
        return None
    return "; ".join(mismatches)


# Planning for better resume or idexing new docs
@dataclass
class IndexPlan:
    '''What one indexer run has to do, decided before anything is parsed.'''
    added: List[Path] = field(default_factory=list)
    changed: List[Path] = field(default_factory=list)
    removed: List[str] = field(default_factory=list)
    unchanged: List[str] = field(default_factory=list)
    digests: Dict[str, str] = field(default_factory=dict)

    @property
    def to_index(self) -> List[Path]:
        return [*self.added, *self.changed]

    @property
    def to_delete(self) -> List[str]:
        return [*self.removed, *(path.name for path in self.changed)]

    def is_empty(self) -> bool:
        return not self.to_index and not self.removed

    def describe(self) -> str:
        lines = []
        for path in self.added:
            lines.append(f"  + new       {path.name}")
        for path in self.changed:
            lines.append(f"  ~ changed   {path.name}")
        for name in self.removed:
            lines.append(f"  - removed   {name}")
        for name in self.unchanged:
            lines.append(f"  = unchanged {name}")
        return "\n".join(lines) if lines else "  (nothing)"

def plan_index_update(
    directory: Optional[str | Path] = None,
    manifest: Optional[Dict[str, Any]] = None,
    only: Optional[Iterable[str]] = None,
    rebuild: bool = False,
) -> IndexPlan:
    '''Check differeneces the SOP folder against the manifest.

    Parameters:
    ---------
    directory (str): the folder of SOP PDFs, defaulting to `SOP_DATA_DIR`.
    manifest (dict): what is currently indexed; read from disk when omitted.
    only (list): restrict the run to these filenames, leaving every other document alone.
    rebuild (boolean): treat every PDF as new, which is what `--rebuild` does.

    Returns:
    ----------
    plan (IndexPlan): the documents to add, re-index, drop and skip.
    '''

    directory = SOP_DATA_DIR if directory is None else directory
    manifest = load_manifest() if manifest is None else manifest
    indexed = dict(manifest.get("sources") or {})
    present = discover_pdf_files(directory)

    pdf_files = present
    if only is not None:
        wanted = {Path(name).name for name in only}
        unknown = wanted - {path.name for path in present}
        if unknown:
            raise FileNotFoundError(f"Not found in {directory}: {sorted(unknown)}")
        pdf_files = [path for path in present if path.name in wanted]

    plan = IndexPlan()
    for path in pdf_files:
        digest = file_digest(path)
        plan.digests[path.name] = digest
        recorded = indexed.get(path.name)

        if rebuild or recorded is None:
            plan.added.append(path)
        elif recorded.get("sha256") != digest:
            plan.changed.append(path)
        else:
            plan.unchanged.append(path.name)

    if only is None:
        names = {path.name for path in present}
        plan.removed = sorted(name for name in indexed if name not in names)

    return plan


# Parsing PDFs into parent documents
def _create_image_describer():
    '''Turns one base64 image into a textual description.'''

    prompt_images = f'''Describe this figure from a standard operating procedure,
regulatory guidance document or laboratory protocol. State what it depicts and
what a reader is meant to take from it: a decision point and the threshold it
turns on, the order of a workflow, the values in a table, an axis and its units.

Keep it under 80 words, and write the description itself — not a template, and
no square-bracketed placeholders. What you write is inserted into the
surrounding section of the document, and a procedure with forty screenshots in
it must not become a passage nobody can read.

If the image carries nothing a reader could use — a logo, a header decoration, a
page-number stub, a cropped fragment of a word — reply with the single word
{IMAGE_SKIP_TOKEN}.'''

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "user",
                [
                    {"type": "text", "text": prompt_images},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image}"},
                    },
                ],
            )
        ]
    )
    llm = ChatOpenAI(model=LLM_CONFIG["image_description_model"], api_key=OPENAI_API_KEY)
    return prompt | llm | StrOutputParser()


def _partition(path: Path, strategy: Optional[str] = None) -> List[Any]:
    '''Partition one PDF into chunked elements

    Parameters:
    ---------
    path (str): the PDF to read.
    strategy (str): override the configured `unstructured` strategy.

    Returns:
    ----------
    chunks (list): its text, tables and figures as chunked elements.
    '''

    from unstructured.partition.pdf import partition_pdf

    config = dict(PDF_PROCESSING_CONFIG)
    if strategy:
        config["strategy"] = strategy

    try:
        return partition_pdf(filename=str(path), **config)
    except Exception as exc:
        if config["strategy"] == PDF_FALLBACK_STRATEGY:
            raise
        print(
            f"  ! {config['strategy']} partitioning failed ({type(exc).__name__}: {exc});"
            f" retrying with strategy={PDF_FALLBACK_STRATEGY}"
        )
        # Fallback strategy when hi-res not work
        config["strategy"] = PDF_FALLBACK_STRATEGY
        # Remove redundant param for fallback strategy
        for key in ("infer_table_structure", "extract_image_block_types", "extract_image_block_to_payload"):
            config.pop(key, None)
        return partition_pdf(filename=str(path), **config)


def _chunk_images(chunk: Any) -> List[str]:
    '''Base64 figures grouped into one chunk by `unstructured`.

    Parameters:
    ---------
    chunk (unstructured element): the chunk to look inside.

    Returns:
    ----------
    images (list): the base64-encoded figures it contains.
    '''

    original_elements = getattr(getattr(chunk, "metadata", None), "orig_elements", None) or []
    images = [
        element.metadata.image_base64
        for element in original_elements
        if "Image" in type(element).__name__
        and getattr(element.metadata, "image_base64", None)
    ]
    return [image for image in images if len(image) >= MIN_IMAGE_PAYLOAD_CHARS]


def _usable_description(description: Optional[str]) -> Optional[str]:
    '''The figure description if it says something, or None if it says nothing.'''

    text = (description or "").strip()
    if not text:
        return None
    if text.upper().startswith(IMAGE_SKIP_TOKEN):
        return None
    if _PLACEHOLDER.search(text):
        return None
    return text


def _chunk_to_text(chunk: Any, image_describer: Any) -> str:
    '''Flatten one chunk, i.e. text, table HTML and figure descriptions, into parent text.

    Parameters:
    ---------
    chunk (unstructured element): the chunk to flatten.
    image_describer (Runnable): describes any figures it contains, or None to skip them.

    Returns:
    ----------
    text (str): the chunk as one block of text.
    '''

    parts: List[str] = []

    if "Table" in type(chunk).__name__:
        html = getattr(getattr(chunk, "metadata", None), "text_as_html", None)
        return html or getattr(chunk, "text", "") or ""

    if getattr(chunk, "text", None):
        parts.append(chunk.text)

    images = _chunk_images(chunk) if image_describer is not None else []
    if images:
        try:
            descriptions = image_describer.batch(
                [{"image": image} for image in images],
                config={"max_concurrency": IMAGE_DESCRIBE_CONCURRENCY},
            )
        except Exception as exc:
            print(f"  ! figure description failed ({type(exc).__name__}: {exc}); keeping the text only")
            descriptions = []
        for description in descriptions:
            usable = _usable_description(description)
            if usable:
                parts.append(f"[Figure] {usable}")

    return "\n".join(part for part in parts if part)


def _source_header(filename: str) -> str:
    '''The `[Source: <filename>]` line every parent opens with.

    Load-bearing, not decoration: `SOP-INT-NA-1_3` appears nowhere inside
    "SOP-INT-NA-1_3 Drug combination screening.pdf" — that document calls itself
    `SOP-R4A-1.1` throughout — so a search for the number a person would cite
    returned the wrong document at rank 1 until the name was part of the text.

    Parameters:
    ---------
    filename (str): the PDF's name.

    Returns:
    ----------
    header (str): the prefix to put at the top of each of its parents.
    '''

    return f"[Source: {filename}]\n\n"


def build_parent_documents(
    pdf_path: str | Path,
    image_describer: Any = None,
    strategy: Optional[str] = None,
) -> List[Document]:
    '''Parse one PDF into parent documents, one per chunked section.

    Parameters:
    ---------
    pdf_path (str): the PDF to parse.
    image_describer (Runnable): describes figures found inside the chunks, or None to skip them.
    strategy (str): override the configured `unstructured` strategy.

    Returns:
    ----------
    parents (list): its sections as Documents carrying `source`, `page` and `element_type`.
    '''

    path = Path(pdf_path)
    chunks = _partition(path, strategy=strategy)

    parents: List[Document] = []
    for chunk in chunks:
        text = _chunk_to_text(chunk, image_describer)
        if not text.strip():
            continue
        metadata = {
            SOURCE_KEY: path.name,
            "element_type": type(chunk).__name__,
        }
        page = getattr(getattr(chunk, "metadata", None), "page_number", None)
        if page is not None:
            metadata["page"] = int(page)

        # Also add the file name to the index text, so the retriever will be file-name aware
        parents.append(
            Document(page_content=_source_header(path.name) + text, metadata=metadata)
        )
    return parents


# The stores/database
def open_vectorstore(embeddings: Any = None) -> Chroma:
    '''Open (or create) the Chroma collection the child vectors live in.'''
    if embeddings is None:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_PATH),
    )

def open_docstore() -> Any:
    '''Open the parent docstore.'''
    return create_kv_docstore(LocalFileStore(str(DOCSTORE_DIR)))


def build_retriever(embeddings: Any = None) -> ParentDocumentRetriever:
    '''Assemble the writing half of the dense arm: children to Chroma, parents to the docstore.

    Parameters:
    ---------
    embeddings (Embeddings): the embedding function, defaulting to the configured OpenAI model.

    Returns:
    ----------
    retriever (ParentDocumentRetriever): configured to embed children and store parents.
    '''

    ensure_directories()
    return ParentDocumentRetriever(
        vectorstore=open_vectorstore(embeddings),
        docstore=open_docstore(),
        child_splitter=RecursiveCharacterTextSplitter(**CHILD_SPLITTER_CONFIG),
        id_key=ID_KEY,
    )


# Helper function for the main indexer options
def _drop_chroma_client_cache() -> None:
    '''Forget `chromadb`'s cached client for every persist path in this process.'''

    try:
        from chromadb.api.shared_system_client import SharedSystemClient

        SharedSystemClient.clear_system_cache()
    except Exception:  # pragma: no cover - a chromadb without the private API
        pass


def clear_index(embeddings: Any = None) -> None:
    '''Delete the whole index — Chroma collection, parent docstore, BM25 corpus and manifest.

    Parameters:
    ---------
    embeddings (Embeddings): the embedding function, needed only to open the collection.
    '''

    if Path(CHROMA_PERSIST_PATH).exists():
        try:
            open_vectorstore(embeddings).delete_collection()
        except Exception as exc:
            print(f"  - no Chroma collection to delete ({type(exc).__name__}: {exc})")
        _drop_chroma_client_cache()
        shutil.rmtree(CHROMA_PERSIST_PATH, ignore_errors=True)
        print("  - cleared the Chroma collection")
    if DOCSTORE_DIR.exists():
        shutil.rmtree(DOCSTORE_DIR)
        print("  - cleared the parent docstore")
    for path in (BM25_CORPUS_PATH, MANIFEST_PATH):
        if path.exists():
            path.unlink()
            print(f"  - removed {path.name}")
    ensure_directories()


# Writing and deleting one document
def add_source(
    retriever: ParentDocumentRetriever,
    parents: List[Document],
) -> Tuple[List[str], int]:
    '''Index one document's parents, embedding their children.

    Parameters:
    ---------
    retriever (ParentDocumentRetriever): the writing half of the dense arm.
    parents (list): the parent Documents to index.

    Returns:
    ----------
    parent_ids (list): the ids assigned to those parents.
    children (int): how many child vectors they produced.
    '''

    parent_ids = [str(uuid4()) for _ in parents]
    for parent, parent_id in zip(parents, parent_ids):
        parent.metadata[ID_KEY] = parent_id

    collection = retriever.vectorstore._collection
    before = collection.count()

    for start in range(0, len(parents), PARENT_BATCH):
        batch = parents[start : start + PARENT_BATCH]
        batch_ids = parent_ids[start : start + PARENT_BATCH]
        retriever.add_documents(batch, ids=batch_ids)
        print(f"  - indexed {min(start + PARENT_BATCH, len(parents))}/{len(parents)} sections")

    return parent_ids, collection.count() - before


def delete_source(
    source: str,
    parent_ids: Optional[Iterable[str]] = None,
    vectorstore: Optional[Chroma] = None,
    docstore: Optional[Any] = None,
) -> Tuple[int, int]:
    '''Remove one document's child vectors and parent documents from the index.

    Parameters:
    ---------
    source (str): the PDF filename, as recorded in every child's `source` metadata.
    parent_ids (list): the parents to drop, from the manifest; recovered from Chroma when omitted.
    vectorstore (Chroma): the child-vector collection, opened when omitted.
    docstore (BaseStore): the parent store, opened when omitted.

    Returns:
    ----------
    children (int): how many child vectors were deleted.
    parents (int): how many parent documents were deleted.
    '''

    vectorstore = open_vectorstore() if vectorstore is None else vectorstore
    docstore = open_docstore() if docstore is None else docstore

    found = vectorstore.get(where={SOURCE_KEY: source}, include=["metadatas"])
    child_ids = list(found.get("ids") or [])

    ids_to_drop = [str(pid) for pid in (parent_ids or [])]
    if not ids_to_drop:
        ids_to_drop = sorted(
            {
                str(metadata.get(ID_KEY))
                for metadata in (found.get("metadatas") or [])
                if metadata and metadata.get(ID_KEY)
            }
        )

    if child_ids:
        vectorstore.delete(ids=child_ids)
    if ids_to_drop:
        docstore.mdelete(ids_to_drop)

    return len(child_ids), len(ids_to_drop)


# Main indexer run
def run_indexer(
    directory: Optional[str | Path] = None,
    only: Optional[Iterable[str]] = None,
    rebuild: bool = False,
    describe_images: bool = True,
    strategy: Optional[str] = None,
    dry_run: bool = False,
    embeddings: Any = None,
) -> IndexPlan:
    '''Bring the index into line with the SOP folder.

    Parameters:
    ---------
    directory (str): the folder of SOP PDFs, defaulting to `SOP_DATA_DIR`.
    only (list): restrict the run to these filenames.
    rebuild (boolean): discard the existing index and start again.
    describe_images (boolean): describe figures with an LLM, the indexer's only model cost.
    strategy (str): override the configured `unstructured` strategy.
    dry_run (boolean): report what would change and write nothing.
    embeddings (Embeddings): the embedding function, defaulting to the configured OpenAI model.

    Returns:
    ----------
    plan (IndexPlan): what the run did, or would have done.
    '''

    directory = SOP_DATA_DIR if directory is None else directory
    ensure_directories()
    manifest = empty_manifest() if rebuild else load_manifest()

    conflict = manifest_settings_conflict(manifest)
    if conflict and not rebuild:
        raise RuntimeError(
            "The index on disk was built with different settings: "
            f"{conflict}. Re-run with --rebuild (every document is re-parsed and re-embedded)."
        )

    plan = plan_index_update(directory, manifest=manifest, only=only, rebuild=rebuild)

    print(f"SOP folder: {directory}")
    print(f"Index:      {INDEX_DIR}")
    print(f"Plan:\n{plan.describe()}")

    if dry_run:
        print("\n--dry-run: nothing written.")
        return plan

    if plan.is_empty() and not rebuild and Path(BM25_CORPUS_PATH).exists():
        print("\nIndex is already up to date.")
        return plan

    if rebuild:
        print("\n[rebuild] clearing the existing index")
        clear_index(embeddings)

    manifest["embedding_model"] = EMBEDDING_MODEL
    manifest["collection_name"] = COLLECTION_NAME
    manifest["child_splitter"] = dict(CHILD_SPLITTER_CONFIG)
    manifest["manifest_version"] = MANIFEST_VERSION

    if plan.to_delete and not rebuild:
        print("\n[delete]")
        vectorstore = open_vectorstore(embeddings)
        docstore = open_docstore()
        for source in plan.to_delete:
            recorded = (manifest.get("sources") or {}).get(source) or {}
            children, parents = delete_source(
                source,
                parent_ids=recorded.get("parent_ids"),
                vectorstore=vectorstore,
                docstore=docstore,
            )
            manifest["sources"].pop(source, None)
            save_manifest(manifest)
            print(f"  - {source}: dropped {parents} sections, {children} child vectors")

    if plan.to_index:
        print("\n[index]")
        retriever = build_retriever(embeddings)
        image_describer = _create_image_describer() if describe_images else None

        for path in plan.to_index:
            print(f"\n  {path.name}")
            try:
                parents = build_parent_documents(path, image_describer, strategy=strategy)
            except Exception as exc:
                print(f"  ! failed to parse ({type(exc).__name__}: {exc}); skipped")
                continue
            if not parents:
                print("  ! no text extracted; skipped")
                continue

            delete_source(
                path.name,
                vectorstore=retriever.vectorstore,
                docstore=retriever.docstore,
            )

            parent_ids, children = add_source(retriever, parents)
            manifest["sources"][path.name] = {
                "sha256": plan.digests[path.name],
                "bytes": path.stat().st_size,
                "indexed_at": _now(),
                "strategy": strategy or PDF_PROCESSING_CONFIG["strategy"],
                "described_images": bool(describe_images),
                "parents": len(parent_ids),
                "children": children,
                "parent_ids": parent_ids,
            }
            save_manifest(manifest)
            print(f"  - {len(parent_ids)} sections, {children} child vectors")

    save_manifest(manifest)

    print("\n[bm25]")
    if manifest["sources"]:
        total = write_bm25_corpus()
        print(f"  - wrote {total} parents to {Path(BM25_CORPUS_PATH).name}")
    else:
        Path(BM25_CORPUS_PATH).unlink(missing_ok=True)
        print("  - no documents left in the index; removed the corpus")

    print("\n=== Indexing complete ===")
    print(f"Documents indexed: {len(manifest['sources'])}")
    print(f"Sections:          {sum(entry.get('parents', 0) for entry in manifest['sources'].values())}")
    print(f"Vector store:      {CHROMA_PERSIST_PATH}")
    print(f"Parent docstore:   {DOCSTORE_DIR}")
    print(f"BM25 corpus:       {BM25_CORPUS_PATH}")
    print("\nRestart the app, or call clear_sop_retriever_cache(), to pick this up.")

    return plan


def main(argv: Optional[List[str]] = None) -> int:
    '''Command-line entry point.

    Parameters:
    ---------
    argv (list): the arguments to parse, defaulting to the process's own.

    Returns:
    ----------
    status (int): 0 on success, 1 when the run could not be completed.
    '''

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(
        prog="python -m backend.sop_rag.sop_indexer",
        description=(
            "Index the SOP PDFs for ensemble retrieval. Incremental by default: only "
            "documents that are new, edited or deleted since the last run are touched."
        ),
    )
    parser.add_argument(
        "--directory",
        default=str(SOP_DATA_DIR),
        help="Folder of SOP PDFs (default: $DATA_ROOT/SOP).",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="NAME",
        help="Index only these filenames. Documents missing from the folder are left alone.",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Discard the index and re-parse, re-describe and re-embed every document.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change and write nothing.",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip figure descriptions: no LLM calls, and figures become unsearchable.",
    )
    parser.add_argument(
        "--strategy",
        choices=["hi_res", "fast", "ocr_only", "auto"],
        help="Override the unstructured parsing strategy (default: hi_res, falling back to fast).",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Rebuild only the BM25 corpus, from the parents already in the docstore.",
    )
    args = parser.parse_args(argv)

    try:
        if args.bm25_only:
            print("[bm25-only]")
            total = write_bm25_corpus()
            print(f"  - wrote {total} parents to {BM25_CORPUS_PATH}")
            return 0

        run_indexer(
            directory=args.directory,
            only=args.files,
            rebuild=args.rebuild,
            describe_images=not args.no_images,
            strategy=args.strategy,
            dry_run=args.dry_run,
        )
        return 0
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
