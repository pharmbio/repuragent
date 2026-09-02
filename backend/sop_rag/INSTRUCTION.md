# SOP retrieval

`protocol_search_sop` searches the SOP corpus — assay protocols, reporting standards,
regulatory definitions and thresholds — and hands the agent the original passages, so
wording can be quoted rather than paraphrased.

## Why two arms

An SOP is asked two kinds of question, and one retriever answers only one of them.

| Query | What finds it |
| --- | --- |
| `SOP-INT-NA-1_3`, "section 5.2", "LysoTracker Red" | **BM25**. An embedding of an identifier lands nowhere near the clause that defines it. |
| "how do I tell phospholipidosis from a staining artefact" | **Embeddings**. The passage that answers it shares almost no words with the question. |

So both run, over the same documents, and their two rankings are fused.

```
                            ┌── BM25 over parent text ────────┐
query ──────────────────────┤                                 ├── fuse ── top k
                            └── Chroma over child chunks ──────┘
                                 (a hit returns its parent)
```

## The pieces

| File | What it is |
| --- | --- |
| `config.py` | every path and parameter; the only place they are named |
| `sop_indexer.py` | PDFs → parent sections → child vectors + BM25 corpus + manifest |
| `bm25_corpus.py` | the sparse arm's on-disk format, written by the indexer and read by the retriever |
| `sop_retriever.py` | `EnsembleSOPRetriever`, the two arms and the seven fusions |

On disk, under `MEMORY_ROOT/sop_documents/ensemble/`:

```
chroma_db/          child chunks (~400 chars), embedded
docstore/           the parent sections, one file per section
bm25_corpus.json    the same parents again, as text for BM25 to index at startup
manifest.json       which PDF produced which parents, and its SHA-256
```

## What a parent contains

One section of one document, as `unstructured` chunked it by title, with three things done
to it:

- **Its source name is prefixed** as `[Source: <filename>]`. An SOP is cited by its number,
  and the number is frequently only in the filename — `SOP-INT-NA-1_3` appears nowhere
  inside `SOP-INT-NA-1_3 Drug combination screening.pdf`, which calls itself `SOP-R4A-1.1`
  throughout.
- **Its figures are described into it.** A figure is the one thing retrieval cannot reach on
  its own, so a vision model writes ~80 words about each one *inside* the section that
  contains it, rather than storing it as a document of its own where the passage referring
  to it and the figure itself could be retrieved apart. Page furniture — logos, header
  rules, a "Page:" stub — is skipped by payload size, and a model that answers `SKIP` or
  hands back a filled-in template is ignored.
- **Nothing splits it further, so it is bounded only by `max_characters=10000`** — and
  loosely, because those figure descriptions are appended *after* `unstructured` has applied
  that cap. One figure-heavy section of the phospholipidosis annex reached 110 000 characters,
  and a parent is a whole tool result, so that whole passage is what an agent would get.
  A 6 000-character parent splitter used to bound this and was **deliberately removed**: a
  parent is now exactly one `unstructured` section, which is the boundary the sibling
  ChemSafeAgent pipeline uses and the one its retrieval sweep was measured on.

## Indexing

Source PDFs live in `DATA_ROOT/SOP` (`persistence/data/SOP`). **Adding one is a one-liner:**

```bash
cd persistence/data/SOP
cp ~/Downloads/new_sop.pdf .
python reindex.py
```

Runs are **incremental**. `manifest.json` holds a SHA-256 per indexed PDF, so a run compares
the folder against it and touches only the difference: a new file is parsed and embedded, an
edited file has its old sections deleted first, a deleted file has its sections dropped, and
everything unchanged is skipped without a single model call.

```bash
python reindex.py                  # inject whatever changed
python reindex.py --dry-run        # say what would change, write nothing
python reindex.py --files new.pdf  # one document, by name; never deletes anything
python reindex.py --rebuild        # discard the index and start again
python reindex.py --no-images      # skip figure descriptions: no LLM calls at all
python reindex.py --bm25-only      # rebuild just the keyword arm, from the parents on disk
```

Equivalently, from the repository root, `python -m backend.sop_rag.sop_indexer <same flags>`.

A rebuild is the expensive path: `unstructured` hi-res parsing takes a couple of minutes per
document, every figure costs one vision call, and every child chunk costs an embedding. That
is the reason the default is incremental and the reason the built index is committed.

A run interrupted part-way is safe to repeat. The manifest is written after each document,
so a resumed run skips what finished; and a document is deleted by `source` before it is
written, so one that had been embedded but not yet recorded is replaced rather than
duplicated.

## Querying

```python
from backend.sop_rag.sop_retriever import get_sop_retriever

documents = get_sop_retriever().query("how is phospholipidosis scored")
scored = get_sop_retriever().query_with_scores("Bliss independence")   # with fused scores
fewer = get_sop_retriever().query("LysoTracker concentration", max_results=2)
```

## Tuning

`RETRIEVAL_CONFIG` and `ENSEMBLE_CONFIG` in `config.py` hold the defaults; every one of them
is also a keyword argument, so an experiment does not need an edit:

```python
get_sop_retriever(fuse_func="rrf", bm25_weight=0.5, dense_weight=0.5, fetch_k=20)
```

| Knob | Default | What it does |
| --- | --- | --- |
| `fuse_func` | `combsum` | `rrf`, `borda`, `log_rank`, `condorcet` use rank only; `combsum`, `isrc`, `log_odds` use the arms' scores |
| `fetch_k` | 50 | candidates per arm before fusion |
| `max_results` | 5 | passages returned |
| `bm25_weight` / `dense_weight` | 0.4 / 0.6 | each arm's say in the fusion |
| `bm25_k1` / `bm25_b` | 0.8 / 0.3 | term-frequency saturation and length normalisation |
