# Technical Details

This chapter presents technical details under the hood of Repuragent. It provides developers with the understanding needed to customise the system. Use this section when you need to extend the system (e.g., add a new tool), audit data flows, or explain to others exactly how Repuragent handles their information. Refer back to the [Usage Guidelines](shared_usage.md) for operator-facing workflows.

## 5.1 System Architecture Overview

### 5.1.1 Core Services

| Component | Role | Local app | Web app |
| --- | --- | --- | --- |
| UI/API | Gradio + FastAPI (`app/gradio_app.py`) | Serves straight to `localhost`; no sign-in | Same widgets, plus auth routes and retention hooks |
| Agent graph | `core/agents/agentic_system.py` — classify → plan → human approval → a nested execution subgraph | Identical | Identical |
| Short-term memory | LangGraph checkpoints (conversations, tool outputs) | SQLite, `persistence/memory/shortterm_memory/langgraph_checkpoints.db` | PostgreSQL database |
| Conversation list | Titles and rendered timelines | The `conversations` table in the same SQLite file | The `user_threads` table, scoped by account |
| Long-term memory | Chroma episodic store (`persistence/memory/episodic_memory`) | On disk, inside the mounted volume | Same path |
| SOP retrieval | `backend/sop_rag` — an ensemble retriever (BM25 + parent-document dense) with an incremental indexer | `persistence/memory/sop_documents/ensemble` | Same layout |


### 5.1.2 Data Flow

1. **User input** enters through Gradio, is checkpointed by LangGraph, and becomes the “messages” consumed by the graph.
2. **Routing** (`task_classifier`) decides between a full plan, a direct execution, a follow-up, and a question about the conversation itself. It routes on *dependence*, not size, so a follow-up like “now rank those by hERG risk” is recognised as one.
3. **Planning agent** produces the breakdown and the run stops at a human-approval interrupt. Approval may carry conditions, which become constraints on the execution.
4. **`plan_init`** parses the approved plan into `plan.md` in the conversation's output folder. That file is the contract the supervisor executes and the progress the task-monitor panel renders.
5. **Supervisor** writes a brief per step and delegates it to a specialist, recording each outcome with `plan_update`. Specialists are context-isolated: each sees its brief and its own work, not the conversation.
6. **Specialist agents** (research, prediction, data) are LangGraph nodes bound to tool suites. Tool calls persist artifacts via `backend/utils/output_paths.py` inside the active conversation's directory.
7. **Report agent** writes the final answer against the evidence the run produced.
8. **Downloads** are gated by HMAC-signed tokens (`FILE_DOWNLOAD_SECRET`) that embed the path, the conversation and an expiry, and the route re-checks that the path is inside that conversation's own directories before serving it.

## 5.2 Agents & Tools

The graph is assembled in `core/agents/agentic_system.py`; each agent is built with
`langchain.agents.create_agent`, and the tool suite each one receives is the list of the
same name in `core/agents/agents.py`. Agent-facing tools all live in `core/tools/`, which
is therefore the complete inventory of what the system can do.

### Planning agent (`PLANNING_TOOLS`)

- **Main role:** decomposes user requirements into a multi-step plan before execution begins.
- **Tools:**
  - `literature_search_litsense` – ranked PubMed passages via the LitSense API.
  - `protocol_search_sop` – RAG tools for REMEDi4ALL's SOPs.
- **Notes:** Uses `PLANNING_SYSTEM_PROMPT`, whose Examples placeholder is filled from episodic memory by the context middleware. It also has `read_files`, so it can look at an upload before planning around it. Its output format is parsed by `parse_plan_steps` — change one and the other must follow.

---

### Research agent (`RESEARCH_TOOLS`)

- **Main role:** gathers biomedical context, builds or inspects knowledge graphs, and surfaces citations plus KG-derived candidates.
- **Tools:**
  - `literature_search_litsense` – ranked PubMed passages via the LitSense API.
  - `protocol_search_sop` – RAG tools for REMEDi4ALL's SOPs.
  - `annotate_chemicals` – Collect drug annotations from public chemical databases (including, ChEMBL, UniChem, PubChem, and KEGG) based on exact match with query pattern.
  - `search_disease_id` – resolves disease names to the identifiers required by KGG.
  - `create_knowledge_graph` – kicks off KGG graph generation and stores it as a pickle file.
  - `extract_drugs_from_kg` – pulls drug nodes plus metadata from an existing KG.
  - `extract_proteins_from_kg` – pulls protein targets from the KG snapshot.
  - `extract_pathways_from_kg` – pulls pathway associations captured in the KG.
  - `extract_mechanism_of_actions_from_kg` – pulls MoA relationships from the KG.
  - `getDrugsforProteins` – pulls Open Targets `knownDrugs` rows given proteins.
  - `getDrugsforMechanisms` – queries ChEMBL’s mechanism/molecule endpoints for the supplied MoA strings, filters by phase/type, and returns the matching drug set with SMILES.
  - `getDrugsforPathways` – resolves pathway names to Reactome IDs, maps associated proteins to Ensembl IDs, and reuses Open Targets `knownDrugs` + ChEMBL SMILES to list pathway-linked drugs.

---

### Data agent (`DATA_TOOLS`)

- **Main role:** performs data analysis and visualization inside a sandboxed workspace.
- **Tools:**
  - `python_executor` – sandboxed Python REPL (pandas, NumPy, RDKit, scikit-learn, etc.) that preserves state between calls. It is an AST-walking interpreter, not `exec`, and every write is clamped inside the conversation's output folder.
  - `reset_python_state` – clears the Python namespace to recover from errors or keep memory low.
  - `read_files` – bounded reads of uploads and outputs, with a preview envelope for large files.

---

### Prediction agent (`PREDICTION_TOOLS`)

- **Main role:** standardizes SMILES inputs and executes CPSign/RDKit models to score ADME/Tox liabilities plus physicochemical properties.
- **Tools:**
  - `predict_repurposedrugs` – new-indication prediction for a set of candidates.
  - `CYP3A4_classifier` – CPSign classification model for CYP3A4 inhibition.
  - `CYP2C19_classifier` – CPSign classification model for CYP2C19 inhibition.
  - `CYP2D6_classifier` – CPSign classification model for CYP2D6 inhibition.
  - `CYP1A2_classifier` – CPSign classification model for CYP1A2 inhibition.
  - `CYP2C9_classifier` – CPSign classification model for CYP2C9 inhibition.
  - `hERG_classifier` – CPSign classification model for hERG cardiotoxicity risk.
  - `AMES_classifier` – CPSign classification model for Ames mutagenicity.
  - `PGP_classifier` – CPSign classification model for P-gp substrate.
  - `PAMPA_classifier` – CPSign classification model for PAMPA permeability.
  - `BBB_classifier` – CPSign classification model for Blood–brain Barrier Penetration.
  - `Solubility_regressor` – CPSign regression model for solubility (output logS).
  - `Lipophilicity_regressor` – RDKit-backed logP estimator.
- **Notes:** Every tool writes its output to a CSV under `persistence/results/<thread_id>/`, and the sidebar lists it for download. The classifiers are **conformal**: a prediction of `0.5` means both labels are plausible at the configured confidence — an abstention, not a probability, and not something to average. CPSign also silently omits structures it cannot featurize, so results are joined on `smiles`, never on row order.

---

### Report agent (`REPORT_TOOLS`)

- **Main role:** assembles the final narrative brief once execution completes.
- **Tools:** `read_files`, `python_executor` — enough to open an artifact it needs to cite.
- **Notes:** Guided by the report prompts, which emit `# Response Summary` → `## Answer` / `## Evidence` / `## Open Issues`. Those headings are what the report stylesheet is built around. Unlike the specialists, the report agent keeps the full conversation view, because it is the one that has to cite evidence.

---

### Supervisor (`core/agents/handoff.py`)

- **Main role:** executes the approved plan by delegating one step at a time, and keeps `plan.md` current.
- **Tools:**
  - `transfer_to_research_agent` / `transfer_to_prediction_agent` / `transfer_to_data_agent` – hand one step to a specialist. The brief is asked for in parts (`objective`, `inputs`, `artifacts`, `constraints`, `expected_output`, `context`) rather than as free text, because the specialist is context-isolated and the brief is its entire input. A brief with a dangling reference, an unexplained artifact or no expected output is refused and comes back to be rewritten.
  - `plan_status` / `plan_update` – read and advance the on-disk plan. `plan_update` validates the step number and status against the file, timestamps it, and returns the refreshed ledger; that ledger *is* the progress display.
- **Notes:** The report agent is a graph node rather than a delegation target, so there is no `transfer_to_report_agent`. 

