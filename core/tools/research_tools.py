'''Literature and protocol search.

Two tools, answering two questions that are not the same one.
`literature_search_litsense` is a single LitSense request — ranked PubMed passages,
back in seconds — for checking whether something has been reported.
`protocol_search_sop` searches the shipped SOP corpus, so a procedural claim can be
grounded in the wording of the source rather than paraphrased.
'''

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

import requests
from langchain_core.tools import tool

from app.config import logger
from backend.sop_rag.config import RETRIEVAL_CONFIG
from backend.sop_rag.sop_retriever import get_sop_retriever

# The tool's default, and the retriever's, are the same number by construction.
DEFAULT_SOP_RESULTS = RETRIEVAL_CONFIG["max_results"]


@dataclass
class LitSenseObject:
    '''Dataclass for a single result from the LitSense API.'''

    text: str
    score: float
    annotations: List[str]
    pmid: int
    pmcid: str
    section: str


class PyLitSense:
    '''Python wrapper for the LitSense API.'''

    def __init__(self, base_url: str = "https://www.ncbi.nlm.nih.gov/research/litsense2-api/api/") -> None:
        self.base_url = base_url.rstrip("/") + "/"

    def query(
        self,
        query_str: str,
        *,
        rerank: bool = True,
        limit: Optional[int] = None,
        min_score: Optional[float] = None,
        mode: str = "passages",
    ) -> List[LitSenseObject]:
        '''Query LitSense API for passages or sentences.

        Parameters:
        ---------
        query_str (str): what to search for, in free text.
        rerank (boolean): whether to let LitSense re-rank by relevance.
        limit (int): how many results to return.
        min_score (float): drop results scoring below this.
        mode (str): `passages` or `sentences`, the granularity LitSense returns.

        Returns:
        ----------
        results (list): the matching `LitSenseObject`s.
        '''

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
    '''Format SOP documents for display.

    Parameters:
    ---------
    documents (list): the SOP chunks the retriever returned.

    Returns:
    ----------
    text (str): the chunks rendered for the agent, each headed by its source and page.
    '''

    result_lines: List[str] = []
    for idx, doc in enumerate(documents, start=1):
        metadata = getattr(doc, "metadata", None) or {}
        source = metadata.get("source") or metadata.get("filename") or "Unknown"
        heading = os.path.basename(str(source))
        page = metadata.get("page")
        if page is not None:
            heading = f"{heading}, page {page}"

        result_lines.extend(
            [
                f"\n--- Document {idx} ---\n",
                f"Source: {heading}\n",
                f"Content: {getattr(doc, 'page_content', '')}\n",
            ]
        )

    return "".join(result_lines)


@tool
def literature_search_litsense(query: str, limit: int = 5) -> str:
    '''Search PubMed passages via the LitSense API. Use it to check whether something has been reported, to confirm a fact
    in passing, or to find the wording a field uses before searching properly.

    Parameters:
    ---------
    query (str): what to search for, in free text.
    limit (int): how many passages to return.

    Returns:
    ----------
    results (str): the matching passages with their citations, or a message when there are none.
    '''

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
        logger.error("Error in literature_search_litsense: %s", exc)
        return f"Error retrieving literature for '{query}': {exc}. Please try again."


@tool
def protocol_search_sop(query: str, max_results: int = DEFAULT_SOP_RESULTS) -> str:
    '''Search the SOP corpus for protocols, standards and regulatory procedures.
    Use this to ground any claim that has to follow a documented procedure —
    assay protocols, reporting standards, regulatory definitions and thresholds.
    Returns the original passages with their source filenames, so the wording can
    be quoted rather than paraphrased.

    Parameters:
    ---------
    query (str): What you need the procedure for, in the terms the document would use.
    max_results (int): How many passages to return. Leave at the default unless you have a reason: a passage is a whole section of a document, so asking for more costs proportionally more of your context. Ask for fewer (2-3) when you need one specific threshold or definition, and for more (8-10) when you are surveying what a procedure covers and do not yet know which section holds it.

    Returns:
    ----------
    results (str): the matching SOP passages with their source filenames, or a message when the index holds nothing relevant.
    '''

    try:
        documents = get_sop_retriever().query(query, max_results=max_results)

        if not documents:
            return f"No SOP content found for query '{query}'."

        return _format_sop_results(documents)

    except Exception as exc:  # pragma: no cover - external service call
        logger.error("Error processing SOP query '%s': %s", query, exc)
        return f"Error retrieving SOP content for '{query}': {exc}. Please try again."


__all__ = [
    "DEFAULT_SOP_RESULTS",
    "LitSenseObject",
    "PyLitSense",
    "literature_search_litsense",
    "protocol_search_sop",
]
