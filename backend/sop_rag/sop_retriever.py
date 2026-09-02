from __future__ import annotations

import math
import re
import threading
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore, create_kv_docstore
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import OPENAI_API_KEY, logger
from backend.sop_rag.bm25_corpus import read_bm25_corpus
from backend.sop_rag.config import (
    BM25_CORPUS_PATH,
    CHILD_SPLITTER_CONFIG,
    CHROMA_PERSIST_PATH,
    COLLECTION_NAME,
    DOCSTORE_DIR,
    EMBEDDING_MODEL,
    ENSEMBLE_CONFIG,
    ID_KEY,
    RETRIEVAL_CONFIG,
    SOURCE_KEY,
)

# Fuse function has two kinds: rank-based and score-based
RANK_FUSIONS = ("rrf", "borda", "log_rank", "condorcet")
SCORE_FUSIONS = ("combsum", "isrc", "log_odds")
FUSE_FUNCS = (*RANK_FUSIONS, *SCORE_FUSIONS)


SEARCH_TYPES = ("similarity", "similarity_score_threshold", "mmr")
# The number of fetch_k for children will be childern_per_parent*fetch_k, then narrow down to fetch_k for parents
CHILDREN_PER_PARENT = 4
LOG_ODDS_EPSILON = 1e-6
_BM25_TOKEN = re.compile(r"[^\W_]+(?:[._\-/][^\W_]+)*")
_BM25_SEPARATORS = re.compile(r"[._\-/]")


def preprocess_bm25_text(text: str) -> List[str]:
    '''Tokenise for the sparse arm: lowercase, punctuation-insensitive, identifier-aware.'''

    tokens: List[str] = []
    for match in _BM25_TOKEN.finditer((text or "").lower()):
        token = match.group(0)
        tokens.append(token)
        parts = _BM25_SEPARATORS.split(token)
        if len(parts) > 1:
            tokens.extend(part for part in parts if part)
    return tokens


def min_max_scaling(scores: Sequence[float]) -> List[float]:
    '''Scale scores into [0, 1] so that two arms' scores are commensurable.'''
    if not scores:
        return []
    lowest = min(scores)
    highest = max(scores)
    if highest == lowest:
        return [1.0] * len(scores)
    span = highest - lowest
    return [(score - lowest) / span for score in scores]


class EnsembleSOPRetriever:
    '''A BM25 arm and a parent-document dense arm over the SOP corpus, fused into one ranking.

    Parameters:
    ---------
    max_results (int): how many fused documents to return.
    fetch_k (int): candidates pulled from each arm before fusion; must be at least `max_results`.
    search_type (str): `similarity`, `similarity_score_threshold` or `mmr`, for the dense arm.
    score_threshold (float): the dense arm's cut-off, used only by `similarity_score_threshold`.
    bm25_weight (float): the sparse arm's weight in the fusion.
    dense_weight (float): the dense arm's weight in the fusion.
    bm25_k1 (float): BM25 term-frequency saturation.
    bm25_b (float): BM25 length normalisation, in [0, 1].
    fuse_func (str): how the two rankings are combined; one of `FUSE_FUNCS`.
    embeddings (Embeddings): the embedding function, defaulting to the configured OpenAI model.
    '''

    def __init__(
        self,
        max_results: Optional[int] = None,
        fetch_k: Optional[int] = None,
        search_type: Optional[str] = None,
        score_threshold: Optional[float] = None,
        bm25_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
        bm25_k1: Optional[float] = None,
        bm25_b: Optional[float] = None,
        fuse_func: Optional[str] = None,
        embeddings: Any = None,
    ) -> None:
        self.max_results = RETRIEVAL_CONFIG["max_results"] if max_results is None else int(max_results)
        self.fetch_k = RETRIEVAL_CONFIG["fetch_k"] if fetch_k is None else int(fetch_k)
        if self.fetch_k < self.max_results:
            raise ValueError("fetch_k must be >= max_results.")
        self.search_type = RETRIEVAL_CONFIG["search_type"] if search_type is None else search_type
        self.score_threshold = (RETRIEVAL_CONFIG["default_score_threshold"] if score_threshold is None else float(score_threshold))
        self.bm25_weight = ENSEMBLE_CONFIG["bm25_weight"] if bm25_weight is None else float(bm25_weight)
        self.dense_weight = ENSEMBLE_CONFIG["dense_weight"] if dense_weight is None else float(dense_weight)
        if self.bm25_weight <= 0 and self.dense_weight <= 0:
            # Every fusion multiplies each arm's contribution by its weight, so two
            # zeroes score every document 0 and the "ranking" is whatever order the
            # accumulator happened to build. Refused at construction rather than
            # returning an arbitrary answer that looks like a real one.
            raise ValueError(
                "bm25_weight and dense_weight cannot both be zero: the ensemble would have no arms."
            )
        self.bm25_k1 = ENSEMBLE_CONFIG["bm25_k1"] if bm25_k1 is None else float(bm25_k1)
        self.bm25_b = ENSEMBLE_CONFIG["bm25_b"] if bm25_b is None else float(bm25_b)
        self.fuse_func = RETRIEVAL_CONFIG["fuse_func"] if fuse_func is None else fuse_func
        if self.fuse_func not in FUSE_FUNCS:
            raise ValueError(f"Unknown fuse_func '{self.fuse_func}'. Choose one of {sorted(FUSE_FUNCS)}.")
        self.rrf_c = ENSEMBLE_CONFIG["rrf_c"]
        self._embeddings = embeddings
        self.bm25_retriever: Optional[BM25Retriever] = None
        self.dense_retriever: Optional[ParentDocumentRetriever] = None
        self.dense_vectorstore: Optional[Chroma] = None
        self.dense_docstore: Any = None
        self._initialize()

    def _build_search_kwargs(self) -> Dict[str, Any]:
        search_kwargs: Dict[str, Any] = {"k": self.fetch_k}
        if self.search_type == "similarity_score_threshold":
            search_kwargs["score_threshold"] = self.score_threshold
        if self.search_type == "mmr":
            search_kwargs["fetch_k"] = self.fetch_k * CHILDREN_PER_PARENT
        return search_kwargs

    def _build_dense_arm(self) -> Tuple[ParentDocumentRetriever, Chroma, Any, int]:
        '''Open the child vector store and the parent docstore.

        Returns:
        ----------
        retriever (ParentDocumentRetriever): the dense arm.
        vectorstore (Chroma): its child vectors, kept for scoring by hand.
        docstore (BaseStore): its parent documents, kept for the same reason.
        count (int): how many child vectors the collection holds.
        '''

        embeddings = self._embeddings
        if embeddings is None:
            embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

        vectorstore = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_PERSIST_PATH),
        )
        count = vectorstore._collection.count()

        docstore = create_kv_docstore(LocalFileStore(str(DOCSTORE_DIR)))
        retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=docstore,
            child_splitter=RecursiveCharacterTextSplitter(**CHILD_SPLITTER_CONFIG),
            id_key=ID_KEY,
            search_type=self.search_type,
            search_kwargs=self._build_search_kwargs(),
        )
        return retriever, vectorstore, docstore, count

    def _build_sparse_arm(self) -> BM25Retriever:
        '''Rebuild the BM25 index from the corpus on disk.'''
        documents = read_bm25_corpus()
        retriever = BM25Retriever.from_documents(
            documents,
            bm25_params={"k1": self.bm25_k1, "b": self.bm25_b},
            preprocess_func=preprocess_bm25_text,
        )
        retriever.k = self.fetch_k
        return retriever

    def _initialize(self) -> None:
        dense_retriever, vectorstore, docstore, dense_count = self._build_dense_arm()
        sparse_retriever = self._build_sparse_arm()

        self.dense_retriever = dense_retriever
        self.dense_vectorstore = vectorstore
        self.dense_docstore = docstore
        self.bm25_retriever = sparse_retriever

    # Fusion functions
    @staticmethod
    def _doc_key(document: Document) -> str:
        '''Identify a document across both arms.

        Parameters:
        ---------
        document (Document): the retrieved document.

        Returns:
        ----------
        key (str): its `doc_id`, or a content hash when the index predates parent ids.
        '''

        metadata = document.metadata or {}
        identifier = metadata.get(ID_KEY)
        if identifier is not None:
            return str(identifier)
        return f"__content__::{hash(document.page_content)}"

    def _accumulate(
        self,
        contributions: Sequence[Sequence[Tuple[Document, float]]],
    ) -> List[Tuple[Document, float]]:
        '''Sum each document's contributions across the arms and order by the total.

        Parameters:
        ---------
        contributions (list): one list per arm of `(document, contribution)` pairs.

        Returns:
        ----------
        fused (list): `(document, score)` pairs, best first.
        '''

        scores: Dict[str, float] = {}
        first_seen: Dict[str, Document] = {}

        for arm in contributions:
            for document, contribution in arm:
                key = self._doc_key(document)
                scores[key] = scores.get(key, 0.0) + contribution
                first_seen.setdefault(key, document)

        ordered = sorted(scores, key=scores.get, reverse=True)
        return [(first_seen[key], scores[key]) for key in ordered]

    def _fuse_by_rank(
        self,
        ranked_lists: Sequence[Sequence[Document]],
        weights: Sequence[float],
        contribution: Callable[[int], float],
    ) -> List[Tuple[Document, float]]:
        '''Fuse on position alone: `contribution(rank)`, weighted per arm.

        Parameters:
        ---------
        ranked_lists (list): one ranked list of Documents per arm.
        weights (list): one weight per arm.
        contribution (callable): what a document at rank `r` (1-based) contributes.

        Returns:
        ----------
        fused (list): `(document, score)` pairs, best first.
        '''

        return self._accumulate(
            [
                [(document, weight * contribution(rank)) for rank, document in enumerate(documents, start=1)]
                for documents, weight in zip(ranked_lists, weights)
            ]
        )

    def _fuse_by_score(
        self,
        scored_lists: Sequence[Sequence[Tuple[Document, float]]],
        weights: Sequence[float],
        contribution: Callable[[float, int], float],
    ) -> List[Tuple[Document, float]]:
        '''Fuse on the arms' own scores: `contribution(normalised_score, rank)`, weighted per arm.

        Parameters:
        ---------
        scored_lists (list): one list of `(document, raw score)` pairs per arm, in rank order.
        weights (list): one weight per arm.
        contribution (callable): what a document with normalised score `s` at rank `r` contributes.

        Returns:
        ----------
        fused (list): `(document, score)` pairs, best first.
        '''

        arms: List[List[Tuple[Document, float]]] = []
        for scored, weight in zip(scored_lists, weights):
            if not scored:
                arms.append([])
                continue
            normalized = min_max_scaling([score for _, score in scored])
            arms.append(
                [
                    (document, weight * contribution(score, rank))
                    for rank, ((document, _), score) in enumerate(zip(scored, normalized), start=1)
                ]
            )
        return self._accumulate(arms)

    def _fuse_condorcet(
        self,
        ranked_lists: Sequence[Sequence[Document]],
        weights: Sequence[float],
    ) -> List[Tuple[Document, float]]:
        '''Fuse by pairwise majority: each arm votes on every pair, weighted.

        Parameters:
        ---------
        ranked_lists (list): one ranked list of Documents per arm.
        weights (list): one weight per arm, used as that arm's voting power.

        Returns:
        ----------
        fused (list): `(document, wins)` pairs, most pairwise wins first.
        '''

        first_seen: Dict[str, Document] = {}
        rank_maps: List[Dict[str, int]] = []
        for documents in ranked_lists:
            ranks: Dict[str, int] = {}
            for rank, document in enumerate(documents, start=1):
                key = self._doc_key(document)
                ranks.setdefault(key, rank)
                first_seen.setdefault(key, document)
            rank_maps.append(ranks)

        keys = list(first_seen)
        wins: Dict[str, int] = {key: 0 for key in keys}
        for index, left in enumerate(keys):
            for right in keys[index + 1 :]:
                votes_left = 0.0
                votes_right = 0.0
                for ranks, weight in zip(rank_maps, weights):
                    rank_left = ranks.get(left, math.inf)
                    rank_right = ranks.get(right, math.inf)
                    if rank_left < rank_right:
                        votes_left += weight
                    elif rank_right < rank_left:
                        votes_right += weight
                if votes_left > votes_right:
                    wins[left] += 1
                elif votes_right > votes_left:
                    wins[right] += 1

        ordered = sorted(keys, key=lambda key: wins[key], reverse=True)
        return [(first_seen[key], float(wins[key])) for key in ordered]

    def fuse(
        self,
        weights: Sequence[float],
        ranked_lists: Optional[Sequence[Sequence[Document]]] = None,
        scored_lists: Optional[Sequence[Sequence[Tuple[Document, float]]]] = None,
    ) -> List[Tuple[Document, float]]:
        '''Combine the arms' rankings according to `fuse_func`.

        Parameters:
        ---------
        weights (list): one weight per arm, in the same order as the lists.
        ranked_lists (list): one ranked list of Documents per arm, for a rank fusion.
        scored_lists (list): one list of `(document, score)` pairs per arm, for a score fusion.

        Returns:
        ----------
        fused (list): `(document, score)` pairs, best first.
        '''

        if self.fuse_func in SCORE_FUSIONS:
            if scored_lists is None:
                raise ValueError(f"fuse_func='{self.fuse_func}' requires scored_lists.")
            if len(scored_lists) != len(weights):
                raise ValueError("scored_lists and weights must have the same length.")
            if self.fuse_func == "combsum":
                return self._fuse_by_score(scored_lists, weights, lambda score, rank: score)
            if self.fuse_func == "isrc":
                return self._fuse_by_score(scored_lists, weights, lambda score, rank: score / (rank * rank))
            return self._fuse_by_score(scored_lists, weights, self._log_odds)

        if ranked_lists is None:
            raise ValueError(f"fuse_func='{self.fuse_func}' requires ranked_lists.")
        if len(ranked_lists) != len(weights):
            raise ValueError("ranked_lists and weights must have the same length.")

        if self.fuse_func == "rrf":
            return self._fuse_by_rank(ranked_lists, weights, lambda rank: 1.0 / (self.rrf_c + rank))
        if self.fuse_func == "borda":
            return self._fuse_by_rank(ranked_lists, weights, lambda rank: float(self.fetch_k - rank + 1))
        if self.fuse_func == "log_rank":
            return self._fuse_by_rank(ranked_lists, weights, lambda rank: -math.log10(rank))
        return self._fuse_condorcet(ranked_lists, weights)

    @staticmethod
    def _log_odds(score: float, rank: int) -> float:
        clipped = min(max(score, LOG_ODDS_EPSILON), 1.0 - LOG_ODDS_EPSILON)
        return math.log(clipped / (1.0 - clipped))

    # The two arms
    def _bm25_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        '''Rank the sparse arm and keep its scores.

        Parameters:
        ---------
        query (str): the search query.

        Returns:
        ----------
        scored (list): the top `fetch_k` `(document, BM25 score)` pairs, best first.
        '''

        retriever = self.bm25_retriever
        if retriever is None:
            raise RuntimeError("BM25 retriever not initialized.")

        tokens = retriever.preprocess_func(query)
        scores = list(retriever.vectorizer.get_scores(tokens))
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [(retriever.docs[index], float(scores[index])) for index in order[: self.fetch_k]]

    def _dense_with_scores(self, query: str) -> List[Tuple[Document, float]]:
        '''Rank the dense arm by parent and keep its scores.

        Parameters:
        ---------
        query (str): the search query.

        Returns:
        ----------
        scored (list): up to `fetch_k` `(parent document, similarity)` pairs, best first.
        '''

        vectorstore = self.dense_vectorstore
        docstore = self.dense_docstore
        if vectorstore is None or docstore is None:
            raise RuntimeError("Dense vectorstore/docstore not initialized.")

        children = vectorstore.similarity_search_with_score(
            query, k=self.fetch_k * CHILDREN_PER_PARENT
        )

        best: Dict[str, float] = {}
        order: List[str] = []
        for child, distance in children:
            parent_id = (child.metadata or {}).get(ID_KEY)
            if parent_id is None:
                continue
            # Chroma reports a distance; the arms are fused on similarity.
            similarity = -float(distance)
            if parent_id not in best:
                best[parent_id] = similarity
                order.append(parent_id)
                if len(order) >= self.fetch_k:
                    break
            elif similarity > best[parent_id]:
                best[parent_id] = similarity

        parent_ids = order[: self.fetch_k]
        parents = docstore.mget(parent_ids)

        scored: List[Tuple[Document, float]] = []
        for parent_id, parent in zip(parent_ids, parents):
            if parent is None:
                continue
            metadata = dict(parent.metadata or {})
            metadata.setdefault(ID_KEY, parent_id)
            scored.append((Document(page_content=parent.page_content, metadata=metadata), best[parent_id]))
        return scored

    def _dense_ranked(self, query: str) -> List[Document]:
        '''The dense arm's ranking alone, for the rank fusions.

        Parameters:
        ---------
        query (str): the search query.

        Returns:
        ----------
        documents (list): up to `fetch_k` parent Documents, best first.
        '''

        if self.dense_retriever is None:
            raise RuntimeError("Dense retriever not initialized.")
        return list(self.dense_retriever.invoke(query))[: self.fetch_k]

    # Retrieval
    def _resolve_max_results(self, max_results: Optional[int]) -> int:
        '''How many results this call should return.

        Parameters:
        ---------
        max_results (int): what the caller asked for, or None for this retriever's default.

        Returns:
        ----------
        count (int): that many, clamped to between 1 and `fetch_k`.
        '''

        if max_results is None:
            return self.max_results
        return max(1, min(int(max_results), self.fetch_k))

    def query_with_scores(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[Tuple[Document, float]]:
        '''Retrieve for `query` and keep the fused score of each hit.

        Parameters:
        ---------
        query (str): what to retrieve SOP material for.
        max_results (int): how many to return, overriding this retriever's default for this call.

        Returns:
        ----------
        results (list): that many `(document, fused score)` pairs, best first.
        '''

        if self.bm25_retriever is None or self.dense_retriever is None:
            raise RuntimeError("SOP retriever not initialized.")

        weights = [self.bm25_weight, self.dense_weight]
        if self.fuse_func in SCORE_FUSIONS:
            fused = self.fuse(
                weights,
                scored_lists=[self._bm25_with_scores(query), self._dense_with_scores(query)],
            )
        else:
            sparse = list(self.bm25_retriever.invoke(query))[: self.fetch_k]
            fused = self.fuse(weights, ranked_lists=[sparse, self._dense_ranked(query)])

        return fused[: self._resolve_max_results(max_results)]

    def query(self, query: str, max_results: Optional[int] = None) -> List[Document]:
        '''Documents relevant to `query`, best first.

        Parameters:
        ---------
        query (str): what to retrieve SOP material for.
        max_results (int): how many to return, overriding this retriever's default for this call.

        Returns:
        ----------
        documents (list): that many parent Documents, best first.
        '''

        return [document for document, _ in self.query_with_scores(query, max_results)]

    search = query

    def get_sources(self, query: str) -> List[str]:
        '''Distinct source filenames behind the hits for `query`.

        Parameters:
        ---------
        query (str): what to retrieve SOP material for.

        Returns:
        ----------
        sources (list): the distinct source filenames behind the hits, for citation.
        '''

        sources = []
        for document in self.query(query):
            metadata = getattr(document, "metadata", None) or {}
            name = metadata.get(SOURCE_KEY) or metadata.get("filename")
            if name:
                sources.append(str(name).rsplit("/", 1)[-1])
        return sorted(set(sources))


_retriever_cache: Dict[Tuple, EnsembleSOPRetriever] = {}
_retriever_cache_lock = threading.Lock()


def get_sop_retriever(**kwargs: Any) -> EnsembleSOPRetriever:
    '''The shared retriever for this configuration, built at most once per process.

    Parameters:
    ---------
    kwargs (dict): any `EnsembleSOPRetriever` argument; each distinct combination is cached separately.

    Returns:
    ----------
    retriever (EnsembleSOPRetriever): the cached instance for those settings.
    '''

    key = tuple(sorted(kwargs.items()))
    retriever = _retriever_cache.get(key)
    if retriever is not None:
        return retriever
    with _retriever_cache_lock:
        retriever = _retriever_cache.get(key)
        if retriever is None:
            retriever = EnsembleSOPRetriever(**kwargs)
            _retriever_cache[key] = retriever
        return retriever

def clear_sop_retriever_cache() -> None:
    '''Forget every cached retriever, e.g. after rebuilding the index on disk.'''

    with _retriever_cache_lock:
        _retriever_cache.clear()


__all__ = [
    "EnsembleSOPRetriever",
    "FUSE_FUNCS",
    "RANK_FUSIONS",
    "SCORE_FUSIONS",
    "clear_sop_retriever_cache",
    "get_sop_retriever",
    "min_max_scaling",
    "preprocess_bm25_text",
]
