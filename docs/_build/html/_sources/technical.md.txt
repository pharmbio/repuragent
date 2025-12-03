# Technical Details

This chapter is aimed at power users who want to understand how Repuragent is wired under the hood—what services run where, how data flows through the stack, and which tools each agent can call.

## 5.1 System Architecture Overview

### 5.1.1 Core Services

| Component | Role | Local app | Web app |
| --- | --- | --- | --- |
| UI/API | Gradio + FastAPI (`app/gradio_app.py`) | Runs inside the local Docker container | Same code, but wrapped with auth routes & retention hooks |
| Supervisor graph | LangGraph supervisor (`core/supervisor/supervisor.py`) orchestrating planning → human review → execution agents | Identical | Identical |
| Short-term memory | LangGraph checkpoints (user conversations, tool outputs) | SQLite file in `backend/memory/shortterm_memory` | PostgreSQL database |
| Long-term memory | Chroma + LangMem episodic store (`backend/memory/episodic_memory`) | Stored on disk, warmed via Docker volume | Stored in `persistence/memory/episodic_memory` |
| SOP Retrieval | `backend/sop_rag` (indexer + retriever) backed by Chroma + docstore | Local files under `backend/memory/sop_documents` | Operators mount their own `persistence/memory/sop_documents` |


### 5.1.2 Data Flow

1. **User input** enters through Gradio, is logged via LangGraph checkpoints, and becomes the “messages” channel consumed by the supervisor.
2. **Supervisor routing** (`route_from_start`) decides whether to invoke the planning agent or proceed directly to execution. Planning outputs can trigger a human-review interrupt before execution resumes.
3. **Execution agents** (Research, Data, Prediction, Report) are LangGraph nodes bound to tool suites. Tool calls persist artifacts via `backend/utils/output_paths.py` inside the active thread directory.
4. **Downloads** are gated by signed tokens (`FILE_DOWNLOAD_SECRET`) that embed the path and expiry (10 min by default).

## 5.2 Agents & Tools

Each agent is defined in `core/agents/` and built with LangGraph’s `create_react_agent` (or a custom StateGraph for the data agent).

### Planning Agent (`core/agents/planning_agent.py`)

- **Main role:** decomposes user requirements into a multi-step plan before execution begins.
- **Tools:**
  - `literature_search_pubmed` – RAG tools for all availale publication on PubMed.
  - `protocol_search_sop` – RAG tools for REMEDi4ALL's SOPs.
- **Notes:** Uses `PLANNING_SYSTEM_PROMPT_ver3` (episodically enhanced when enabled) via
  `EpisodicLearningSystem.create_enhanced_planning_prompt` so previous successful
  decompositions can seed new plans.

---

### Research Agent (`core/agents/research_agent.py`)

- **Main role:** gathers biomedical context, builds or inspects knowledge graphs, and surfaces citations plus KG-derived candidates.
- **Tools:**
  - `literature_search_pubmed` – RAG tools for all availale publication on PubMed.
  - `protocol_search_sop` – RAG tools for REMEDi4ALL's SOPs.
  - `search_disease_id` – resolves disease names to the identifiers required by KGG.
  - `create_knowledge_graph` – kicks off KGG graph generation and persists the pickle.
  - `extract_drugs_from_kg` – pulls drug nodes plus metadata from an existing KG.
  - `extract_proteins_from_kg` – pulls protein targets from the KG snapshot.
  - `extract_pathways_from_kg` – pulls pathway associations captured in the KG.
  - `extract_mechanism_of_actions_from_kg` – pulls MoA relationships from the KG.
  - `getDrugsforProteins` – pulls Open Targets `knownDrugs` rows given proteins.
  - `getDrugsforMechanisms` – queries ChEMBL’s mechanism/molecule endpoints for the supplied MoA strings, filters by phase/type, and returns the matching drug set with SMILES.
  - `getDrugsforPathways` – resolves pathway names to Reactome IDs, maps associated proteins to Ensembl IDs, and reuses Open Targets `knownDrugs` + ChEMBL SMILES to list pathway-linked drugs.
  - `prompt_with_file_path` – resolves natural-language file references into concrete repo paths.
- **Notes:** Outputs citations, graph summaries, and ranked tables as JSON/CSV artifacts stored under each thread directory.

---

### Data Agent (`core/agents/data_agent.py`)

- **Main role:** performs data analysis and visualization inside a sandboxed workspace.
- **Tools:**
  - `python_executor` – sandboxed Python REPL (pandas, NumPy, RDKit, scikit-learn, etc.) that preserves state between calls.
  - `reset_python_state` – nukes the Python namespace to recover from errors or keep memory low.
  - `prompt_with_file_path` – turns human-friendly file descriptions into absolute paths inside the thread sandbox.
- **Notes:** Runs on a custom LangGraph StateGraph (`ToolNode`) that injects the writable directory via `ensure_task_dir()`, enabling clean CSV/Parquet exports and Matplotlib/Plotly/Seaborn visualizations.

---

### Prediction Agent (`core/agents/prediction_agent.py`)

- **Main role:** standardizes SMILES inputs and executes CPSign/RDKit models to score ADME/Tox liabilities plus physicochemical properties.
- **Tools:**
  - `smiles_csv` – canonicalizes raw SMILES inputs into `data/modelling_data.csv`.
  - `CYP3A4_classifier` – CPSign classification model for CYP3A4 inhibition.
  - `CYP2C19_classifier` – CPSign classification model for CYP2C19 inhibition.
  - `CYP2D6_classifier` – CPSign classification model for CYP2D6 inhibition.
  - `CYP1A2_classifier` – CPSign classification model for CYP1A2 inhibition.
  - `CYP2C9_classifier` – CPSign classification model for CYP2C9 inhibition.
  - `hERG_classifier` – CPSign classification model for hERG cardiotoxicity risk.
  - `AMES_classifier` – CPSign Ames mutagenicity classifier.
  - `PGP_classifier` – CPSign classifier for P-gp substrate likelihood.
  - `PAMPA_classifier` – CPSign classifier for PAMPA permeability.
  - `BBB_classifier` – CPSign classifier for blood–brain barrier penetration.
  - `Solubility_regressor` – CPSign regression model returning logS with confidence.
  - `Lipophilicity_regressor` – RDKit-backed logP estimator.
- **Notes:** Every tool writes its CSV output (probabilities, `p_value_0/1`, binary calls) via `task_file_path()` so downloads remain sandboxed per thread.

---

### Report Agent (`core/agents/report_agent.py`)

- **Main role:** assembles the final narrative brief once execution completes.
- **Tools:** *(none – relies on the conversation history provided by the supervisor).*
- **Notes:** Guided by `REPORT_SYSTEM_PROMPT`, it summarizes findings, caveats, and recommended next steps using.

---

### Supervisor & Human-in-the-Loop

- **Main role:** orchestrates routing between planning, agents, and optional `human_chat` interrupts while maintaining shared memory.
- **Tools:**
  - `create_supervisor` – wires together the Research, Data, Prediction, and Report agents.
  - `route_from_start` – initial router deciding whether to plan first or execute directly.
  - `human_chat` – pause point for manual approval or edits before execution resumes.
- **Notes:** `initialize_memory()` selects `AsyncSqliteSaver` (local) or the web checkpoint backend, and `backend/utils/output_paths.py` keeps every tool locked to its thread directory for both safety and per-user isolation.

Use this section when you need to extend the system (e.g., add a new tools), audit data flows for compliance reviews, or explain to others exactly how Repuragent handles their information. Refer back to the [Shared Usage Guidelines](shared_usage.md) for operator-facing workflows and the [Troubleshooting](troubleshooting.md) chapter for common failure modes.
