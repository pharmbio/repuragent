'''The SOP ensemble: fusion arithmetic, incremental indexing, and one real index.

Two halves. The first is pure arithmetic — the fusion functions decide which
passage an agent gets to quote, and each one is a few lines that are easy to get
subtly wrong, so they are checked against hand-computed expectations rather than
against each other.

The second half builds a **real index**: a real Chroma collection, a real
`ParentDocumentRetriever`, a real BM25 index, real files on disk. Only two things
are faked, and both are the network: `HashingEmbeddings` stands in for OpenAI
embeddings, and `_partition` stands in for `unstructured` reading a PDF. Every
path that decides *what is in the index* — the manifest diff, deleting a
document's vectors by `source`, rebuilding the sparse corpus — is the real one,
because those are the paths where a mistake is silent. An index that has quietly
kept a deleted document's sections still answers every query.
'''

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document

from tests.fakes import HashingEmbeddings
from tests.harness import scoped_sop_index

# --- Fakes for the one part of the pipeline that reads a PDF -------------------


class _Metadata:
    '''The attributes `unstructured` chunk metadata exposes to the indexer.'''

    def __init__(self, page_number: Optional[int] = None, text_as_html: Optional[str] = None) -> None:
        self.page_number = page_number
        self.text_as_html = text_as_html
        self.filename = None
        self.image_base64: Optional[str] = None
        self.orig_elements: List[Any] = []


class Image:
    '''An extracted image block, carrying its payload the way `unstructured` does.'''

    def __init__(self, payload: str) -> None:
        self.metadata = _Metadata()
        self.metadata.image_base64 = payload


class CompositeElement:
    '''A prose section. Named for the class `unstructured` produces, which the indexer type-checks.'''

    def __init__(self, text: str, page: Optional[int] = None, images: Optional[List[str]] = None) -> None:
        self.text = text
        self.metadata = _Metadata(page_number=page)
        self.metadata.orig_elements = [Image(payload) for payload in (images or [])]


class Table:
    '''A table chunk, whose parent text is its HTML rather than its flattened text.'''

    def __init__(self, html: str, page: Optional[int] = None) -> None:
        self.text = "flattened table text"
        self.metadata = _Metadata(page_number=page, text_as_html=html)


ALPHA = [
    CompositeElement(
        "Phospholipidosis is detected by staining lysosomes with LysoTracker Red "
        "and scoring the fluorescent puncta per cell.",
        page=1,
    ),
    CompositeElement(
        "Cationic amphiphilic drugs accumulate in the lysosome and inhibit "
        "phospholipase activity, which is the mechanism this assay reports on.",
        page=2,
    ),
    Table("<table><tr><td>LysoTracker Red</td><td>50 nM</td></tr></table>", page=3),
]

BETA = [
    CompositeElement(
        "SOP-INT-NA-1_3 describes drug combination screening. Synergy is quantified "
        "with the Bliss independence model over a dose matrix.",
        page=1,
    ),
    CompositeElement(
        "Report the Bliss excess for every dose pair, and flag any combination "
        "whose excess exceeds the threshold agreed in the study plan.",
        page=2,
    ),
]

CORPUS = {"alpha.pdf": ALPHA, "beta.pdf": BETA}


def _write_pdfs(folder: Path, names: Optional[List[str]] = None) -> None:
    '''Write placeholder files for the corpus, since only the fake partitioner reads them.

    Parameters:
    ---------
    folder (Path): the SOP folder to write into.
    names (list): which documents to write, defaulting to all of them.
    '''

    for name in names or list(CORPUS):
        (folder / name).write_bytes(f"%PDF-1.4 {name}".encode("utf-8"))


def _fake_partition(chunks_by_name: Optional[Dict[str, List[Any]]] = None):
    '''A stand-in for `unstructured`, returning the chunks this test wrote for that file.

    Parameters:
    ---------
    chunks_by_name (dict): filename -> chunks, defaulting to the module's corpus.

    Returns:
    ----------
    partition (callable): a replacement for `sop_indexer._partition`.
    '''

    table = CORPUS if chunks_by_name is None else chunks_by_name

    def partition(path, strategy=None):
        return table[Path(path).name]

    return partition


def _build_index(folder: Path, indexer, names: Optional[List[str]] = None, **kwargs) -> Any:
    '''Run the indexer over the fake corpus with local embeddings.

    Parameters:
    ---------
    folder (Path): the SOP folder.
    indexer (module): `backend.sop_rag.sop_indexer`, already patched by the fixture.
    names (list): the documents to write first, defaulting to all of them.
    **kwargs: passed through to `run_indexer`.

    Returns:
    ----------
    plan (IndexPlan): what the run did.
    '''

    _write_pdfs(folder, names)
    indexer._partition = _fake_partition()
    return indexer.run_indexer(
        directory=folder,
        describe_images=False,
        embeddings=HashingEmbeddings(),
        **kwargs,
    )


def _retriever(**kwargs):
    from backend.sop_rag.sop_retriever import EnsembleSOPRetriever

    settings = {"embeddings": HashingEmbeddings(), "fetch_k": 10, "max_results": 3}
    settings.update(kwargs)
    return EnsembleSOPRetriever(**settings)


def _fusion(fuse_func: str, fetch_k: int = 10):
    '''An instance configured for fusion only, with no stores opened.

    `fuse` reads three scalars and nothing else, and the arithmetic is worth
    testing without a Chroma collection standing behind it.

    Parameters:
    ---------
    fuse_func (str): the fusion to configure.
    fetch_k (int): the candidate depth, which `borda` scores against.

    Returns:
    ----------
    retriever (EnsembleSOPRetriever): configured for fusion, uninitialised otherwise.
    '''

    from backend.sop_rag.config import ENSEMBLE_CONFIG
    from backend.sop_rag.sop_retriever import EnsembleSOPRetriever

    retriever = object.__new__(EnsembleSOPRetriever)
    retriever.fuse_func = fuse_func
    retriever.fetch_k = fetch_k
    retriever.rrf_c = ENSEMBLE_CONFIG["rrf_c"]
    return retriever


def _doc(identifier: str) -> Document:
    return Document(page_content=f"content of {identifier}", metadata={"doc_id": identifier})


def _ids(results) -> List[str]:
    '''The doc_id of each result, whether they are Documents or `(Document, score)` pairs.

    Parameters:
    ---------
    results (list): what a fusion or a query returned.

    Returns:
    ----------
    identifiers (list): one doc_id per result, in the order given.
    '''

    documents = [result[0] if isinstance(result, tuple) else result for result in results]
    return [(document.metadata or {}).get("doc_id") for document in documents]


# --- Fusion arithmetic --------------------------------------------------------

def test_bm25_tokens_survive_case_and_punctuation() -> None:
    from backend.sop_rag.sop_retriever import preprocess_bm25_text

    assert preprocess_bm25_text("Phospholipidosis, detected.") == ["phospholipidosis", "detected"]
    assert preprocess_bm25_text("LysoTracker") == preprocess_bm25_text("lysotracker")
    assert preprocess_bm25_text("") == []


def test_an_identifier_is_tokenised_whole_and_in_parts() -> None:
    '''Whole, so an exact citation outranks a document that shares only "sop"; in
    parts, so a citation written with different punctuation still finds it.'''

    from backend.sop_rag.sop_retriever import preprocess_bm25_text

    tokens = preprocess_bm25_text("See SOP-INT-NA-1_3 for details.")

    assert "sop-int-na-1_3" in tokens
    assert {"sop", "int", "na", "1", "3"} <= set(tokens)
    # The same identifier written another way shares every part.
    loose = set(preprocess_bm25_text("SOP INT NA 1.3"))
    assert {"sop", "int", "na", "1", "3"} <= loose


def test_min_max_scaling_maps_onto_the_unit_interval() -> None:
    from backend.sop_rag.sop_retriever import min_max_scaling

    assert min_max_scaling([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert min_max_scaling([]) == []


def test_min_max_scaling_treats_a_flat_arm_as_all_ones() -> None:
    '''A BM25 arm that scored every candidate the same must not become all zeros.

    Mapping a flat list onto the bottom of the scale would silently delete that
    arm from the fusion.
    '''

    from backend.sop_rag.sop_retriever import min_max_scaling

    assert min_max_scaling([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_rrf_rewards_the_document_both_arms_found() -> None:
    '''A document ranked second by both arms beats one ranked first by only one.'''

    fusion = _fusion("rrf")
    sparse = [_doc("only-sparse"), _doc("both")]
    dense = [_doc("only-dense"), _doc("both")]

    fused = fusion.fuse([1.0, 1.0], ranked_lists=[sparse, dense])

    scores = {document.metadata["doc_id"]: score for document, score in fused}

    assert _ids(fused)[0] == "both"
    # 2 * 1/62 for `both`, against 1/61 for each single-arm hit.
    assert abs(scores["both"] - 2 / 62) < 1e-12
    assert scores["only-sparse"] < scores["both"]


def test_rrf_respects_the_arm_weights() -> None:
    '''With the dense arm weighted up, its top hit outranks the sparse arm's.'''

    fusion = _fusion("rrf")
    sparse = [_doc("sparse-top")]
    dense = [_doc("dense-top")]

    fused = fusion.fuse([0.4, 0.6], ranked_lists=[sparse, dense])

    assert _ids(fused) == ["dense-top", "sparse-top"]


def test_combsum_adds_the_arms_normalised_scores() -> None:
    '''The score fusions read how well each arm scored a document, not just where it ranked.'''

    fusion = _fusion("combsum")
    shared = _doc("shared")
    sparse = [(shared, 10.0), (_doc("sparse-tail"), 0.0)]
    dense = [(_doc("dense-top"), -0.1), (shared, -0.5)]

    fused = fusion.fuse([0.5, 0.5], scored_lists=[sparse, dense])
    scores = {document.metadata["doc_id"]: score for document, score in fused}

    # `shared` takes 1.0 from the sparse arm and 0.0 from the dense one; the
    # dense arm's own top hit takes 1.0 there and nothing from the sparse arm.
    assert abs(scores["shared"] - 0.5) < 1e-12
    assert abs(scores["dense-top"] - 0.5) < 1e-12
    assert abs(scores["sparse-tail"] - 0.0) < 1e-12
    assert len(fused) == 3


def test_isrc_discounts_a_hit_by_its_rank() -> None:
    fusion = _fusion("isrc")
    sparse = [(_doc("first"), 2.0), (_doc("second"), 2.0)]

    fused = fusion.fuse([1.0], scored_lists=[sparse])
    scores = {document.metadata["doc_id"]: score for document, score in fused}

    # Both scored 2.0, so both normalise to 1.0 and only the rank separates them.
    assert scores["first"] == 1.0
    assert scores["second"] == 0.25


def test_every_fusion_returns_each_document_once() -> None:
    '''Whatever the fusion, a document found by both arms is returned once.'''

    from backend.sop_rag.sop_retriever import RANK_FUSIONS, SCORE_FUSIONS

    sparse = [_doc("a"), _doc("b"), _doc("c")]
    dense = [_doc("c"), _doc("a"), _doc("d")]
    scored_sparse = [(document, 3.0 - index) for index, document in enumerate(sparse)]
    scored_dense = [(document, -float(index)) for index, document in enumerate(dense)]

    for name in (*RANK_FUSIONS, *SCORE_FUSIONS):
        fusion = _fusion(name)
        if name in SCORE_FUSIONS:
            fused = fusion.fuse([0.4, 0.6], scored_lists=[scored_sparse, scored_dense])
        else:
            fused = fusion.fuse([0.4, 0.6], ranked_lists=[sparse, dense])
        identifiers = _ids(fused)
        assert sorted(identifiers) == ["a", "b", "c", "d"], f"{name}: {identifiers}"


def test_a_fusion_that_needs_scores_refuses_a_ranked_list() -> None:
    fusion = _fusion("combsum")
    try:
        fusion.fuse([1.0], ranked_lists=[[_doc("a")]])
    except ValueError as exc:
        assert "scored_lists" in str(exc)
    else:
        raise AssertionError("combsum accepted ranked_lists")


# --- Configuration guards -----------------------------------------------------

def test_an_unknown_fusion_is_refused_at_construction() -> None:
    from backend.sop_rag.sop_retriever import EnsembleSOPRetriever

    try:
        EnsembleSOPRetriever(fuse_func="magic")
    except ValueError as exc:
        assert "magic" in str(exc)
    else:
        raise AssertionError("an unknown fuse_func was accepted")


def test_fetch_k_below_max_results_is_refused() -> None:
    '''Asking for more results than candidates would silently return fewer.'''

    from backend.sop_rag.sop_retriever import EnsembleSOPRetriever

    try:
        EnsembleSOPRetriever(max_results=10, fetch_k=3)
    except ValueError as exc:
        assert "fetch_k" in str(exc)
    else:
        raise AssertionError("fetch_k < max_results was accepted")


def test_two_zero_weights_are_refused() -> None:
    from backend.sop_rag.sop_retriever import EnsembleSOPRetriever

    try:
        EnsembleSOPRetriever(bm25_weight=0, dense_weight=0)
    except ValueError as exc:
        assert "bm25_weight" in str(exc)
    else:
        raise AssertionError("an ensemble with no arms was accepted")


# --- The manifest diff --------------------------------------------------------

def test_the_plan_separates_new_changed_removed_and_unchanged() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _write_pdfs(folder)
        plan = indexer.plan_index_update(folder)
        assert [path.name for path in plan.added] == ["alpha.pdf", "beta.pdf"]
        assert plan.unchanged == [] and plan.removed == []

        manifest = indexer.empty_manifest()
        for name, digest in plan.digests.items():
            manifest["sources"][name] = {"sha256": digest, "parent_ids": []}
        manifest["sources"]["gone.pdf"] = {"sha256": "x", "parent_ids": ["p1"]}

        (folder / "alpha.pdf").write_bytes(b"%PDF-1.4 alpha, edited")
        plan = indexer.plan_index_update(folder, manifest=manifest)

        assert [path.name for path in plan.changed] == ["alpha.pdf"]
        assert plan.unchanged == ["beta.pdf"]
        assert plan.removed == ["gone.pdf"]
        assert plan.added == []
        assert not plan.is_empty()


def test_naming_files_never_deletes_the_others() -> None:
    '''`--files` is for injecting one document, so it must not read absent files as deletions.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _write_pdfs(folder, ["beta.pdf"])
        manifest = indexer.empty_manifest()
        manifest["sources"]["alpha.pdf"] = {"sha256": "whatever", "parent_ids": []}

        plan = indexer.plan_index_update(folder, manifest=manifest, only=["beta.pdf"])

        assert [path.name for path in plan.added] == ["beta.pdf"]
        assert plan.removed == []


def test_an_unknown_filename_is_reported_rather_than_ignored() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _write_pdfs(folder, ["beta.pdf"])
        try:
            indexer.plan_index_update(folder, only=["typo.pdf"])
        except FileNotFoundError as exc:
            assert "typo.pdf" in str(exc)
        else:
            raise AssertionError("a misspelled filename was silently skipped")


def test_changed_index_settings_stop_an_incremental_run() -> None:
    '''Embedding into a collection built by another model must not be possible by accident.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)

        original = indexer.EMBEDDING_MODEL
        indexer.EMBEDDING_MODEL = "text-embedding-3-small"
        try:
            conflict = indexer.manifest_settings_conflict(indexer.load_manifest())
            assert conflict and "embedding model" in conflict

            try:
                indexer.run_indexer(directory=folder, embeddings=HashingEmbeddings())
            except RuntimeError as exc:
                assert "--rebuild" in str(exc)
            else:
                raise AssertionError("an incremental run proceeded across an embedding change")
        finally:
            indexer.EMBEDDING_MODEL = original

        # The format version covers changes the settings cannot see — how parent
        # text is built, rather than where it is stored or by what model.
        stale = indexer.load_manifest()
        stale["manifest_version"] = indexer.MANIFEST_VERSION - 1
        indexer.save_manifest(stale)
        conflict = indexer.manifest_settings_conflict(indexer.load_manifest())
        assert conflict and "index format" in conflict


# --- A real index -------------------------------------------------------------

def test_indexing_writes_all_three_parts_of_the_index() -> None:
    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer
        from backend.sop_rag.config import index_exists

        _build_index(folder, indexer)

        assert index_exists()
        manifest = json.loads((index_dir / "manifest.json").read_text())
        assert sorted(manifest["sources"]) == ["alpha.pdf", "beta.pdf"]

        alpha = manifest["sources"]["alpha.pdf"]
        assert alpha["parents"] == len(ALPHA) == len(alpha["parent_ids"])
        assert alpha["children"] >= alpha["parents"]
        assert len(alpha["sha256"]) == 64

        corpus = json.loads((index_dir / "bm25_corpus.json").read_text())
        assert len(corpus["documents"]) == len(ALPHA) + len(BETA)
        # Both arms have to agree on what a document is, and `doc_id` is how.
        assert all(entry["metadata"].get("doc_id") for entry in corpus["documents"])
        assert {entry["metadata"]["source"] for entry in corpus["documents"]} == {"alpha.pdf", "beta.pdf"}


def test_a_table_is_indexed_as_its_html_and_a_page_is_recorded() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        indexer._partition = _fake_partition()
        _write_pdfs(folder, ["alpha.pdf"])
        parents = indexer.build_parent_documents(folder / "alpha.pdf")

        assert [parent.metadata["page"] for parent in parents] == [1, 2, 3]
        assert all(parent.metadata["source"] == "alpha.pdf" for parent in parents)
        assert "<table>" in parents[-1].page_content
        assert parents[-1].metadata["element_type"] == "Table"


def test_one_section_is_one_parent_however_long_it_is() -> None:
    '''Nothing splits a section: one `unstructured` chunk is exactly one parent.

    The parent splitter that used to bound this was dropped so that parent
    boundaries match the sibling ChemSafeAgent pipeline the retrieval defaults
    were swept on. What that gives up is pinned here rather than left implicit:
    `unstructured`'s `max_characters` is applied *before* figure descriptions are
    written in, so a figure-heavy section can arrive as one very long passage,
    and the whole of it is what an agent gets.
    '''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        long_section = CompositeElement(
            "\n\n".join(f"Step {number}: incubate the plate and read the wells." for number in range(400)),
            page=2,
        )
        indexer._partition = _fake_partition({"alpha.pdf": [long_section]})
        _write_pdfs(folder, ["alpha.pdf"])

        parents = indexer.build_parent_documents(folder / "alpha.pdf")

        header = indexer._source_header("alpha.pdf")
        assert len(parents) == 1
        assert parents[0].page_content == header + long_section.text
        assert parents[0].metadata["page"] == 2
        assert "part" not in parents[0].metadata

        # And a section is never split, so a parent is never labelled as a piece.
        indexer._partition = _fake_partition()
        whole = indexer.build_parent_documents(folder / "alpha.pdf")
        assert len(whole) == len(ALPHA)
        assert all("part" not in parent.metadata for parent in whole)


def test_page_furniture_is_not_sent_to_the_vision_model() -> None:
    '''`unstructured` extracts every image block, logos and header stubs included.

    One SOP annex produced forty-six. Each would cost a vision call and put a
    sentence about a page-number stub into the passage it landed in.
    '''

    import backend.sop_rag.sop_indexer as indexer

    furniture = "x" * (indexer.MIN_IMAGE_PAYLOAD_CHARS - 1)
    figure = "y" * (indexer.MIN_IMAGE_PAYLOAD_CHARS + 1)
    chunk = CompositeElement("Procedure text.", page=1, images=[furniture, figure])

    assert indexer._chunk_images(chunk) == [figure]


def test_a_description_that_describes_nothing_is_dropped() -> None:
    '''Two answers that must not reach the index: a decline, and a filled-in template.'''

    import backend.sop_rag.sop_indexer as indexer

    assert indexer._usable_description("Scatter plot of spot counts per cell, y-axis 0-25.")
    assert indexer._usable_description(indexer.IMAGE_SKIP_TOKEN) is None
    assert indexer._usable_description("  skip  ") is None
    assert indexer._usable_description("") is None
    assert indexer._usable_description(None) is None
    template = "Figure X. [Type: decision tree/workflow/table]. Shows [process or data]."
    assert indexer._usable_description(template) is None


def test_a_figure_is_described_into_the_section_that_contains_it() -> None:
    '''A figure is the one thing retrieval cannot reach on its own, so its
    description belongs in the parent rather than in a document of its own.'''

    import backend.sop_rag.sop_indexer as indexer

    class _Describer:
        def __init__(self) -> None:
            self.seen: List[str] = []

        def batch(self, inputs, config=None):
            self.seen = [item["image"] for item in inputs]
            return ["Decision tree branching on spot count per cell.", indexer.IMAGE_SKIP_TOKEN]

    payloads = ["a" * (indexer.MIN_IMAGE_PAYLOAD_CHARS + 1), "b" * (indexer.MIN_IMAGE_PAYLOAD_CHARS + 1)]
    chunk = CompositeElement("Scoring the assay.", page=1, images=payloads)
    describer = _Describer()

    text = indexer._chunk_to_text(chunk, describer)

    assert describer.seen == payloads
    assert "Scoring the assay." in text
    assert "[Figure] Decision tree branching on spot count per cell." in text
    assert indexer.IMAGE_SKIP_TOKEN not in text


def test_every_parent_names_its_source_document() -> None:
    '''The number a person cites is often only in the filename.

    `SOP-INT-NA-1_3` appears nowhere inside "SOP-INT-NA-1_3 Drug combination
    screening.pdf" — that document calls itself `SOP-R4A-1.1` throughout — so
    without the name in the text, searching for it found nothing.
    '''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        indexer._partition = _fake_partition()
        _write_pdfs(folder, ["beta.pdf"])

        parents = indexer.build_parent_documents(folder / "beta.pdf")

        assert all(parent.page_content.startswith("[Source: beta.pdf]") for parent in parents)
        # The header is a prefix, not a replacement.
        assert "Bliss independence" in parents[0].page_content


def test_a_document_is_findable_by_its_filename() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever()

        scored = retriever._bm25_with_scores("beta.pdf")

        assert scored
        best, score = scored[0]
        assert best.metadata["source"] == "beta.pdf"
        assert score > 0


def test_retrieval_finds_the_document_that_holds_the_terms() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever()

        documents = retriever.query("Bliss independence synergy dose matrix")

        assert documents, "the ensemble returned nothing"
        assert documents[0].metadata["source"] == "beta.pdf"
        assert "Bliss" in documents[0].page_content
        assert len(documents) <= 3

        # Fusion has to dedup: both arms rank the same parents.
        identifiers = [document.metadata["doc_id"] for document in documents]
        assert len(identifiers) == len(set(identifiers))

        # `search` is the name the tool layer used before the ensemble existed.
        assert retriever.search("Bliss independence synergy dose matrix") == documents


def test_the_sparse_arm_finds_an_exact_identifier() -> None:
    '''The reason for having a keyword arm at all: an SOP number is not a meaning.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever()

        scored = retriever._bm25_with_scores("SOP-INT-NA-1_3")

        assert scored, "the sparse arm returned nothing"
        best, score = scored[0]
        assert "SOP-INT-NA-1_3" in best.page_content
        assert score > 0


def test_the_dense_arm_returns_parents_not_child_chunks() -> None:
    '''A child is what matches; the parent is what gets quoted.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever()

        scored = retriever._dense_with_scores("lysosome phospholipidosis staining")

        assert scored
        # Whole sections come back, not the ~400-character children that matched.
        sections = [chunk.text for chunk in ALPHA if isinstance(chunk, CompositeElement)]
        returned = [document.page_content for document, _ in scored]
        assert any(
            section in content for section in sections for content in returned
        ), "no returned document contains a whole parent section"
        assert all((document.metadata or {}).get("doc_id") for document, _ in scored)


def test_sources_are_reported_for_citation() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)

        sources = _retriever().get_sources("phospholipidosis lysosome")

        assert sources
        assert set(sources) <= {"alpha.pdf", "beta.pdf"}


# --- Incremental injection ----------------------------------------------------

def test_a_second_run_with_nothing_changed_does_no_work() -> None:
    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        first = json.loads((index_dir / "manifest.json").read_text())

        plan = indexer.run_indexer(directory=folder, embeddings=HashingEmbeddings())

        assert plan.is_empty()
        assert plan.unchanged == ["alpha.pdf", "beta.pdf"]
        second = json.loads((index_dir / "manifest.json").read_text())
        # Same parent ids: nothing was re-parsed and nothing was re-embedded.
        assert first["sources"] == second["sources"]


def test_injecting_a_new_document_leaves_the_others_alone() -> None:
    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer, names=["alpha.pdf"])
        alpha_before = json.loads((index_dir / "manifest.json").read_text())["sources"]["alpha.pdf"]

        plan = _build_index(folder, indexer)

        assert [path.name for path in plan.added] == ["beta.pdf"]
        manifest = json.loads((index_dir / "manifest.json").read_text())
        assert manifest["sources"]["alpha.pdf"] == alpha_before
        assert manifest["sources"]["beta.pdf"]["parents"] == len(BETA)

        corpus = json.loads((index_dir / "bm25_corpus.json").read_text())
        assert len(corpus["documents"]) == len(ALPHA) + len(BETA)

        # And the new document is reachable, which the corpus alone would not prove.
        documents = _retriever().query("Bliss independence synergy")
        assert any(document.metadata["source"] == "beta.pdf" for document in documents)


def test_reindexing_a_changed_document_replaces_its_sections() -> None:
    '''The failure this guards is silent: old sections surviving beside the new ones.'''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        before = json.loads((index_dir / "manifest.json").read_text())["sources"]["beta.pdf"]

        replacement = CompositeElement(
            "Revised: synergy is now quantified with the Loewe additivity model.", page=1
        )
        (folder / "beta.pdf").write_bytes(b"%PDF-1.4 beta, revised")
        indexer._partition = _fake_partition({"alpha.pdf": ALPHA, "beta.pdf": [replacement]})
        plan = indexer.run_indexer(directory=folder, describe_images=False, embeddings=HashingEmbeddings())

        assert [path.name for path in plan.changed] == ["beta.pdf"]
        after = json.loads((index_dir / "manifest.json").read_text())["sources"]["beta.pdf"]
        assert after["parents"] == 1
        assert not set(after["parent_ids"]) & set(before["parent_ids"])

        corpus = json.loads((index_dir / "bm25_corpus.json").read_text())
        contents = [entry["page_content"] for entry in corpus["documents"]]
        assert len(contents) == len(ALPHA) + 1
        assert any("Loewe" in content for content in contents)
        assert not any("Bliss" in content for content in contents)

        # The stale child vectors have to be gone from Chroma too, not just from
        # the docstore: a hit on one would return None and vanish from the results.
        collection = indexer.open_vectorstore(HashingEmbeddings())._collection
        stale = collection.get(where={"doc_id": {"$in": list(before["parent_ids"])}}, include=[])
        assert not stale["ids"]


def test_deleting_a_pdf_removes_it_from_the_index() -> None:
    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        beta = json.loads((index_dir / "manifest.json").read_text())["sources"]["beta.pdf"]

        (folder / "beta.pdf").unlink()
        plan = indexer.run_indexer(directory=folder, describe_images=False, embeddings=HashingEmbeddings())

        assert plan.removed == ["beta.pdf"]
        manifest = json.loads((index_dir / "manifest.json").read_text())
        assert sorted(manifest["sources"]) == ["alpha.pdf"]

        collection = indexer.open_vectorstore(HashingEmbeddings())._collection
        assert not collection.get(where={"source": "beta.pdf"}, include=[])["ids"]

        docstore = indexer.open_docstore()
        assert docstore.mget(list(beta["parent_ids"])) == [None] * len(beta["parent_ids"])

        # Nothing from the deleted document can still be retrieved.
        documents = _retriever().query("Bliss independence synergy dose matrix")
        assert all(document.metadata["source"] == "alpha.pdf" for document in documents)


def test_a_rebuild_leaves_no_orphan_vector_segments() -> None:
    '''`delete_collection()` alone leaks the collection's HNSW segment directory.

    That is 12 MB of preallocated vector file per rebuild, in a directory this
    repository commits — four rebuilds had put 48 MB of dead weight in it before
    anyone looked.
    '''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        _build_index(folder, indexer, rebuild=True)
        _build_index(folder, indexer, rebuild=True)

        segments = [path for path in (index_dir / "chroma_db").iterdir() if path.is_dir()]
        assert len(segments) == 1, f"orphan segment directories left behind: {segments}"

        # And the index still works, which is the half a plain rmtree breaks.
        assert _retriever().query("Bliss independence synergy")


def test_indexing_a_document_twice_does_not_duplicate_it() -> None:
    '''What a run killed between embedding a document and saving the manifest leaves behind.

    The stores hold its sections and the manifest does not, so the next run sees
    a new document. Indexing has to be idempotent per `source` or the corpus
    grows a second copy of everything that was interrupted.
    '''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)

        manifest = indexer.load_manifest()
        orphaned = manifest["sources"].pop("beta.pdf")
        indexer.save_manifest(manifest)

        plan = indexer.run_indexer(directory=folder, describe_images=False, embeddings=HashingEmbeddings())

        assert [path.name for path in plan.added] == ["beta.pdf"]
        rewritten = indexer.load_manifest()["sources"]["beta.pdf"]
        assert rewritten["parents"] == len(BETA)
        assert not set(rewritten["parent_ids"]) & set(orphaned["parent_ids"])

        collection = indexer.open_vectorstore(HashingEmbeddings())._collection
        assert collection.count() == sum(
            entry["children"] for entry in indexer.load_manifest()["sources"].values()
        )
        corpus = json.loads((index_dir / "bm25_corpus.json").read_text())
        assert len(corpus["documents"]) == len(ALPHA) + len(BETA)


def test_rebuild_starts_from_an_empty_index() -> None:
    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        first = json.loads((index_dir / "manifest.json").read_text())["sources"]

        _build_index(folder, indexer, rebuild=True)
        second = json.loads((index_dir / "manifest.json").read_text())["sources"]

        assert sorted(second) == sorted(first)
        assert not set(second["alpha.pdf"]["parent_ids"]) & set(first["alpha.pdf"]["parent_ids"])

        collection = indexer.open_vectorstore(HashingEmbeddings())._collection
        assert collection.count() == sum(entry["children"] for entry in second.values())


def test_the_bm25_corpus_can_be_rebuilt_on_its_own() -> None:
    '''`--bm25-only` has to be able to repair the sparse arm without re-embedding anything.'''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer
        from backend.sop_rag.bm25_corpus import write_bm25_corpus

        _build_index(folder, indexer)
        (index_dir / "bm25_corpus.json").unlink()

        total = write_bm25_corpus()

        assert total == len(ALPHA) + len(BETA)
        assert (index_dir / "bm25_corpus.json").exists()


def test_removing_every_document_leaves_no_corpus_behind() -> None:
    '''An index with nothing in it must not be left with a corpus claiming otherwise.'''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        for name in CORPUS:
            (folder / name).unlink()

        plan = indexer.run_indexer(directory=folder, describe_images=False, embeddings=HashingEmbeddings())

        assert sorted(plan.removed) == sorted(CORPUS)
        assert json.loads((index_dir / "manifest.json").read_text())["sources"] == {}
        assert not (index_dir / "bm25_corpus.json").exists()


def test_a_lost_corpus_is_rebuilt_by_an_ordinary_run() -> None:
    '''The sparse arm can go missing on its own, and "up to date" would not fix it.'''

    with scoped_sop_index() as (folder, index_dir):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        (index_dir / "bm25_corpus.json").unlink()

        plan = indexer.run_indexer(directory=folder, describe_images=False, embeddings=HashingEmbeddings())

        assert plan.is_empty()
        corpus = json.loads((index_dir / "bm25_corpus.json").read_text())
        assert len(corpus["documents"]) == len(ALPHA) + len(BETA)


def test_a_missing_index_names_the_command_that_builds_it() -> None:
    with scoped_sop_index() as (_folder, _index):
        try:
            _retriever()
        except FileNotFoundError as exc:
            assert "sop_indexer" in str(exc)
        else:
            raise AssertionError("a retriever was built over an index that does not exist")


def test_get_sop_retriever_caches_per_configuration() -> None:
    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer
        from backend.sop_rag.sop_retriever import clear_sop_retriever_cache, get_sop_retriever

        _build_index(folder, indexer)
        embeddings = HashingEmbeddings()

        first = get_sop_retriever(embeddings=embeddings, max_results=3, fetch_k=10)
        again = get_sop_retriever(embeddings=embeddings, max_results=3, fetch_k=10)
        other = get_sop_retriever(embeddings=embeddings, max_results=2, fetch_k=10)

        assert first is again, "the retriever was rebuilt for an identical configuration"
        assert first is not other, "two configurations shared one retriever"

        clear_sop_retriever_cache()
        assert get_sop_retriever(embeddings=embeddings, max_results=3, fetch_k=10) is not first


# --- The tool layer -----------------------------------------------------------

def test_max_results_is_a_per_call_override() -> None:
    '''Per call, not per retriever: `get_sop_retriever` caches per configuration, so
    varying this through the constructor would build one whole retriever per value.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever(max_results=3, fetch_k=10)
        question = "phospholipidosis lysosome staining"

        assert len(retriever.query(question)) == 3
        assert len(retriever.query(question, max_results=1)) == 1
        assert len(retriever.query(question, max_results=5)) == 5
        assert len(retriever.query_with_scores(question, max_results=2)) == 2

        # And the shared retriever is still one object, whatever was asked for.
        from backend.sop_rag.sop_retriever import get_sop_retriever

        embeddings = HashingEmbeddings()
        first = get_sop_retriever(embeddings=embeddings, fetch_k=10)
        first.query(question, max_results=2)
        assert get_sop_retriever(embeddings=embeddings, fetch_k=10) is first


def test_an_out_of_range_result_count_is_clamped_not_refused() -> None:
    '''The caller is usually a model filling in a tool argument, and there is a
    sensible answer to "give me 500 passages" that is not an error message.'''

    with scoped_sop_index() as (folder, _index):
        import backend.sop_rag.sop_indexer as indexer

        _build_index(folder, indexer)
        retriever = _retriever(max_results=3, fetch_k=10)

        assert retriever._resolve_max_results(None) == 3
        assert retriever._resolve_max_results(500) == 10
        assert retriever._resolve_max_results(0) == 1
        assert retriever._resolve_max_results(-4) == 1
        assert len(retriever.query("phospholipidosis", max_results=0)) == 1


def test_the_tool_offers_max_results_and_passes_it_through() -> None:
    '''A tool argument the model can set and the retriever then ignores is worse
    than not offering it, so both halves are checked.'''

    import core.tools.research_tools as research_tools
    from backend.sop_rag.config import RETRIEVAL_CONFIG

    # The number itself is a tuning decision that lives in the config; what must
    # hold is that the tool offers the retriever's own default rather than a second
    # one that drifted away from it.
    argument = research_tools.protocol_search_sop.args["max_results"]
    assert argument["default"] == research_tools.DEFAULT_SOP_RESULTS == RETRIEVAL_CONFIG["max_results"]

    asked: Dict[str, Any] = {}

    class _Stub:
        def query(self, query, max_results=None):
            asked["query"] = query
            asked["max_results"] = max_results
            return [Document(page_content="body", metadata={"source": "alpha.pdf"})]

    original = research_tools.get_sop_retriever
    research_tools.get_sop_retriever = lambda **kwargs: _Stub()
    try:
        rendered = research_tools.protocol_search_sop.invoke(
            {"query": "how is synergy scored", "max_results": 9}
        )
        assert asked == {"query": "how is synergy scored", "max_results": 9}
        assert "alpha.pdf" in rendered

        research_tools.protocol_search_sop.invoke({"query": "again"})
        assert asked["max_results"] == research_tools.DEFAULT_SOP_RESULTS
    finally:
        research_tools.get_sop_retriever = original


def test_the_tool_names_the_source_and_page_of_every_passage() -> None:
    from core.tools.research_tools import _format_sop_results

    rendered = _format_sop_results(
        [
            Document(page_content="body one", metadata={"source": "alpha.pdf", "page": 4}),
            # What the retired index wrote, which a rebuild has not yet replaced.
            Document(page_content="body two", metadata={"filename": "/data/SOP/beta.pdf"}),
            Document(page_content="body three", metadata={}),
        ]
    )

    assert "Source: alpha.pdf, page 4" in rendered
    assert "Source: beta.pdf" in rendered
    assert "Source: Unknown" in rendered
    assert "body three" in rendered


if __name__ == "__main__":
    from tests.harness import run_module_tests

    passed, failed = run_module_tests(globals(), label="test_sop_rag.")
    raise SystemExit(1 if failed else 0)
