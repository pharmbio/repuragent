from pathlib import Path
import json
import statistics
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from IPython.display import display
from pypdf import PdfReader
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Path to the CSV file containing evaluation questions
QUESTIONS_PATH: Path = Path("questions.csv")  # change as needed

# Directory containing the source SOP PDFs.
# Set to None to auto-detect using SOP_DATA_DIR / project defaults.
PDF_DIR: Path | None = None

# ---- project root on sys.path ----
CURRENT_DIR = Path(".").resolve()
PROJECT_ROOT = CURRENT_DIR.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.sop_rag.config import SOP_DATA_DIR          # noqa: E402
from backend.sop_rag.sop_retriever import SOPRetriever   # noqa: E402

print(f"Project root  : {PROJECT_ROOT}")
print(f"SOP_DATA_DIR  : {SOP_DATA_DIR}")

def normalize_text(text: str) -> str:
    ascii_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    cleaned = []
    for char in ascii_text.lower():
        if char.isalnum():
            cleaned.append(char)
        else:
            cleaned.append(" ")
    return " ".join("".join(cleaned).split())


def _split_pipe_list(value: Any) -> List[str]:
    if value is None or pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split("|") if item.strip()]


def load_questions(path: Path) -> pd.DataFrame:
    questions = pd.read_csv(path).fillna("").copy()
    questions["target_files"] = questions["target_files"].map(_split_pipe_list)
    questions["hard_negative_files"] = questions["hard_negative_files"].map(_split_pipe_list)
    questions["evidence_file"] = questions["evidence_file"].map(
        lambda value: Path(str(value)).name if str(value).strip() else ""
    )
    questions["evidence_page"] = pd.to_numeric(
        questions["evidence_page"],
        errors="coerce",
    ).astype("Int64")
    return questions


def resolve_pdf_dir(explicit_pdf_dir: Optional[Path] = None) -> Path:
    candidates = []
    if explicit_pdf_dir is not None:
        candidates.append(explicit_pdf_dir)
    candidates.extend([
        SOP_DATA_DIR,
        PROJECT_ROOT / "persistence" / "data" / "SOP",
    ])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    checked = ", ".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Could not find SOP PDF directory. Checked: {checked}")


def resolve_pdf_map(pdf_dir: Path) -> Dict[str, Path]:
    return {p.name: p for p in pdf_dir.glob("*.pdf")}


def extract_pdf_pages(pdf_path: Path) -> Dict[int, str]:
    reader = PdfReader(str(pdf_path))
    return {
        idx: (page.extract_text() or "")
        for idx, page in enumerate(reader.pages, start=1)
    }



def set_retrieval_k(retriever: SOPRetriever, k: int) -> None:
    search_kwargs = getattr(retriever.retriever, "search_kwargs", None)
    if isinstance(search_kwargs, dict):
        search_kwargs["k"] = k
    else:
        retriever.retriever.search_kwargs = {"k": k}


def retrieve_documents(
    retriever: SOPRetriever, query: str, k: int
) -> List[Dict[str, Any]]:
    set_retrieval_k(retriever, k)
    docs = retriever.retriever.invoke(query)
    docs = retriever._convert_bytes_to_docs(docs)
    results = []
    for rank, doc in enumerate(docs, start=1):
        metadata = getattr(doc, "metadata", {}) or {}
        page_content = getattr(doc, "page_content", "") or ""
        results.append({
            "rank": rank,
            "filename": Path(metadata.get("filename", "unknown")).name,
            "content_type": metadata.get("content_type"),
            "page_content": page_content,
            "normalized_page_content": normalize_text(page_content),
        })
    return results


def find_evidence_match(
    retrieved_docs: List[Dict[str, Any]],
    evidence_file: str,
    evidence_page: Any,
    evidence_snippet: str,
) -> Optional[Dict[str, Any]]:
    file_name = Path(str(evidence_file)).name if str(evidence_file).strip() else ""
    normalized_snippet = normalize_text(evidence_snippet)
    if not file_name or not normalized_snippet or pd.isna(evidence_page):
        return None
    page_number = int(evidence_page)
    for doc in retrieved_docs:
        if doc["filename"] != file_name:
            continue
        if normalized_snippet in doc["normalized_page_content"]:
            return {
                "rank": doc["rank"],
                "file": file_name,
                "page": page_number,
                "snippet": evidence_snippet,
            }
    return None

def get_first_rank(
    retrieved_docs: List[Dict[str, Any]], predicate
) -> Optional[int]:
    for item in retrieved_docs:
        if predicate(item):
            return item["rank"]
    return None



def evaluate_questions(
    retriever: SOPRetriever,
    questions: pd.DataFrame,
    k: int,
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = []
    for question in questions.itertuples(index=False):
        start = time.perf_counter()
        retrieved_docs = retrieve_documents(retriever, question.question, k)
        latency = time.perf_counter() - start

        target_files = set(question.target_files)
        file_rank = get_first_rank(
            retrieved_docs,
            lambda item: item["filename"] in target_files,
        )
        evidence_match = find_evidence_match(
            retrieved_docs,
            question.evidence_file,
            question.evidence_page,
            question.evidence_snippet,
        )

        results.append({
            "id": question.id,
            "question": question.question,
            "query_type": question.query_type,
            "difficulty": question.difficulty,
            "target_files": question.target_files,
            "hard_negative_files": question.hard_negative_files,
            "evidence_file": question.evidence_file,
            "evidence_page": int(question.evidence_page) if not pd.isna(question.evidence_page) else None,
            "evidence_snippet": question.evidence_snippet,
            "file_match_rank": file_rank,
            "evidence_match_rank": evidence_match["rank"] if evidence_match else None,
            "matched_evidence_file": evidence_match["file"] if evidence_match else None,
            "matched_evidence_page": evidence_match["page"] if evidence_match else None,
            "matched_evidence_snippet": evidence_match["snippet"] if evidence_match else None,
            "latency_sec": latency,
            "retrieved": [
                {
                    "rank": item["rank"],
                    "filename": item["filename"],
                    "content_type": item["content_type"],
                    "preview": item["page_content"][:300],
                }
                for item in retrieved_docs
            ],
        })
    return pd.DataFrame(results)


# Evaulation functions
def recall(ranks: List[Optional[int]]) -> Dict[str, float]:
    total = len(ranks)
    hits = [float(r) for r in ranks if pd.notna(r)]
    return len(hits) / total


def mrr(ranks: List[Optional[int]]) -> Dict[str, float]:
    total = len(ranks)
    hits = [float(r) for r in ranks if pd.notna(r)]
    return sum(1.0 / r for r in hits) / total if total else 0.0