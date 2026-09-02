# Repuragent — An AI Scientist for Drug Repurposing

## Overview

Drug repurposing offers an efficient strategy to accelerate therapeutic discovery by identifying new indications for existing drugs. However, the process remains hindered by the heterogeneity of biological and chemical data and the difficulty of forming early, evidence-based hypotheses about candidate drugs, targets, and clinical endpoints. We introduce Repuragent (Drug Repurposing Agentic System), a proof-of-concept multi-agent framework designed to autonomously plan, execute, and refine data-driven repurposing workflows under human-in-the-loop supervision. The system integrates autonomous research, data extraction, knowledge graph (KG) construction, and analytical reasoning with an adaptive long-term memory mechanism that improves the system over time.

<div align="center">
  <img src="images/agent_architecture.png" width="500">
</div>

This repository is the **local, single-user application**: clone it, run it on your own
machine, and every file, database row and conversation stays on your device. There is no
account to create and no database server to run. We also offer:

- [Repuragent Web](https://repuragent.serve.scilifelab.se) — the hosted multi-user version, usable without installation.
- [Repuragent Web on GitHub](https://github.com/pharmbio/repuragent-web) — its source.
- [Documentation](https://repuragent.readthedocs.io/) — user guides and technical details.

## Version announcement

Version 2 keeps the same science as version 1 and rebuilds how it is planned, executed
and shown.

- **Approve before it runs.** The planner proposes a breakdown and the run waits at an
  approval gate until you accept it — including conditional approvals ("go ahead, but
  only phase 3 drugs"), which are carried into the execution as constraints.
- **The plan is a file, not prose.** The supervisor writes and updates `plan.md` in the
  conversation's output folder, so progress cannot drift from what actually happened,
  a task can be resumed, and a follow-up can read the work log.
- **A persistent task monitor** shows which step you are on while the run is still going.
- **Context isolation.** The supervisor holds the whole conversation; each specialist gets
  a written brief and nothing else, so its work is not shaped by another agent's.
- **A rebuilt SOP search.** An ensemble retriever fuses a BM25 arm and a parent-document
  dense arm over the same passages, with an incremental indexer beside it. An identifier
  you would cite (`SOP-INT-NA-1_3`) and a description of what you need now both find the
  right document.

### The agents

- **Planning agent** — decomposes the request into an executable breakdown, using standard
  operating procedures, literature, and precedent from earlier successful runs.
- **Supervisor** — writes a task brief for each step, delegates it to the specialist that
  can do it, checks what comes back, and keeps `plan.md` current.
- **Research agent** — literature search, SOP retrieval, disease-identifier resolution,
  knowledge-graph construction from OpenTargets, ChEMBL, UniProt, Reactome, KEGG and GWAS
  data, and mining that graph for drugs, proteins, pathways and mechanisms.
- **Prediction agent** — pre-trained CPSign conformal ADMET models (CYP3A4, CYP2C19,
  CYP2D6, CYP1A2, CYP2C9, hERG, Ames, P-gp, PAMPA, BBB, solubility, lipophilicity) and
  new-indication prediction.
- **Data agent** — Python for inspecting uploads, combining tables, scoring and ranking
  candidates, statistics and figures.
- **Report agent** — the final answer, tied to the evidence the run produced.

### Memory

- **Episodic memory** — how successful tasks were decomposed, retrieved as precedent when
  planning a similar one. Recording an episode is deliberate: press *Remember this plan*.
- **Conversation state** — checkpointed in SQLite, so a conversation survives a restart
  and a plan can wait at its approval gate indefinitely.
- **SOP retrieval** — a prebuilt index over regulatory guidance and protocol documents, so
  procedural claims are grounded in the wording of the source. Add a document by dropping
  the PDF into `persistence/data/SOP` and running `python reindex.py` there; only what
  changed is re-indexed.

## Quick start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), or Python 3.12+ and a JRE.
- An OpenAI API key from [platform.openai.com](https://platform.openai.com/).
- (Optional) a LangSmith account for tracing, from [smith.langchain.com](https://smith.langchain.com/).

### With Docker

```bash
git clone https://github.com/pharmbio/repuragent.git
cd repuragent
cp .env.example .env        # then put your OPENAI_API_KEY in it
docker compose up --build
```

Open [http://localhost:7860](http://localhost:7860).

Everything the app persists — the database, your conversations, uploads, results, the
SOP corpus and its index — lives under `persistence/`, which is the single directory
`docker-compose.yml` mounts. `docker compose down` and `up` again picks up exactly where
you left off. (Upgrading from v1: that one mount replaced the three older ones for
`./data`, `./results` and `./backend/memory`.)

To put that data somewhere else, set `PERSIST_ROOT` in `.env` — or `DATA_ROOT`,
`RESULTS_ROOT` and `MEMORY_ROOT` individually.

### Without Docker

```bash
pip install -r requirements.txt   # a JRE is also needed, for the CPSign ADMET models
cp .env.example .env              # then put your OPENAI_API_KEY in it
python main.py
```

### Daily use

```bash
docker compose up          # start
docker compose down        # stop
docker compose logs -f     # follow the logs
python -m tests.run_all    # the test suite: offline, no API key needed
```

## Project structure

```
repuragent/
├── app/           Gradio UI, the run loop, conversations, downloads
├── core/          the agent graph (core/agents/), prompts, and every agent-facing tool (core/tools/)
├── backend/       the database, the SOP retrieval system, the sandbox, domain API clients
├── persistence/   everything that survives a restart: your conversations, the database,
│                  uploads, results, the SOP corpus and its prebuilt index
├── models/        pre-trained CPSign ADMET models
├── analysis/      the evaluations behind the paper
├── tests/         163 offline tests
└── main.py        entry point
```

Configuration is in one file, `app/config.py`, and every value in it is overridable from
the environment; `.env.example` lists the ones worth knowing about.
