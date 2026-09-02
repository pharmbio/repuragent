# SOP corpus

The PDFs the agents search with `protocol_search_sop`. Drop a new one in here and run:

```bash
python reindex.py
```

That parses, describes the figures in, and embeds **only** what is new or edited since the
last run, deletes anything you removed from this folder, and rebuilds the keyword arm's
corpus. Restart the app afterwards to pick it up.

```bash
python reindex.py --dry-run        # say what would change, write nothing
python reindex.py --files new.pdf  # just this one document
python reindex.py --rebuild        # re-parse and re-embed everything (slow, costs tokens)
python reindex.py --no-images      # skip figure descriptions, so no LLM calls at all
```

It needs `OPENAI_API_KEY` in the repository's `.env` — it is read for you — and the built
index is committed, so run it deliberately and commit what it writes to
`persistence/memory/sop_documents/ensemble/`.

`backend/sop_rag/INSTRUCTION.md` explains what the index is and how retrieval over it works.
