from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, List

import requests
from langchain_core.tools import tool

from app.config import logger
from backend.sop_rag.sop_retriever import SOPRetriever


@dataclass
class LitSenseObject:
    """Dataclass for a single result from the LitSense API."""

    text: str
    score: float
    annotations: List[str]
    pmid: int
    pmcid: str
    section: str


class PyLitSense:
    """Python wrapper for the LitSense API."""

    def __init__(self, base_url: str = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/") -> None:
        self.base_url = base_url.rstrip("/") + "/"

    def query(
        self,
        query_str: str,
        *,
        rerank: bool = False,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        mode: str = "passages",
    ) -> List[LitSenseObject]:
        """Query LitSense API for passages or sentences."""
        path = "passages/" if mode == "passages" else "sentences/"
        url = f"{self.base_url}{path}"

        params = {"query": query_str, "rerank": rerank, "limit": limit}
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()

        results = [LitSenseObject(**result) for result in response.json()]

        if min_score is not None:
            results = [result for result in results if result.score >= min_score]

        return results


def _format_sop_results(documents) -> str:
    """Format SOP documents for display."""
    result_lines: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        filename = "Unknown"
        if hasattr(doc, "metadata"):
            filename = os.path.basename(doc.metadata.get("filename", "Unknown"))

        result_lines.extend(
            [
                f"\n--- Document {idx} ---\n",
                f"Source: {filename}\n",
                f"Content: {getattr(doc, 'page_content', '')}\n",
            ]
        )

    return "".join(result_lines)


@tool
def literature_search_pubmed(query: str, limit: int = 5) -> str:
    """Search scientific literature via the LitSense API."""
    try:
        engine = PyLitSense()
        results = engine.query(query, limit=limit)

        if not results:
            return f"No relevant literature found for '{query}'. Please try a broader query."

        result_sections: List[str] = []
        for idx, result in enumerate(results, start=1):
            section = (
                f"\n--- Passage #{idx} ---\n"
                f"PMID: {result.pmid}\n"
                f"Content: {result.text}\n"
            )
            result_sections.append(section)

        return "".join(result_sections)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error in literature_search_pubmed: %s", exc)
        return f"Error retrieving literature for '{query}': {exc}. Please try again."


@tool
def protocol_search_sop(query: str) -> str:
    """Search SOP documents for protocols and regulatory procedures."""
    retriever = SOPRetriever()

    try:
        documents = retriever.retriever.invoke(query)
        documents = retriever._convert_bytes_to_docs(documents)

        if not documents:
            return f"No SOP content found for query '{query}'."

        return _format_sop_results(documents)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error processing SOP query '%s': %s", query, exc)
        return f"Error retrieving SOP content for '{query}': {exc}. Please try again."


__all__ = [
    "LitSenseObject",
    "PyLitSense",
    "literature_search_pubmed",
    "protocol_search_sop",
]
