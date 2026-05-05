# Technical Details

This chapter presents technical details under the hood of Repuragent. It provides developers with the understanding needed to customise the system. Use this section when you need to extend the system (e.g., add a new tool), audit data flows, or explain to others exactly how Repuragent handles their information. Refer back to the [Usage Guidelines](shared_usage.md) for operator-facing workflows.

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

1. **User input** enters through Gradio, is logged via LangGraph checkpoints, and becomes the “messages” consumed by the supervisor.
2. **Initial routing** (`route_from_start`) decides whether to invoke the planning agent or proceed directly to execution. 
3. **Planning agent** generates intial plan and can trigger a human-review interrupt before execution resumes.
4. **Supervisor agent** recieves final plan from planning agent and delegate sub-task to specialised agent and keep trach on the execution status. 
3. **Specilise agents** (Research, Data, Prediction, Report) are LangGraph nodes bound to tool suites. Tool calls persist artifacts via `backend/utils/output_paths.py` inside the active thread directory.
4. **Downloads** are gated by signed tokens (`FILE_DOWNLOAD_SECRET`) that embed the path and expiry (10 min by default).

## 5.2 Agents & Tools

Each agent is defined in `core/agents/` and built with LangGraph’s `create_react_agent` (or a custom StateGraph for the data agent).

### Planning Agent (`core/agents/planning_agent.py`)

- **Main role:** decomposes user requirements into a multi-step plan before execution begins.
- **Tools:**
  - `literature_search_pubmed` – RAG tools for all available publications on PubMed.
  - `protocol_search_sop` – RAG tools for REMEDi4ALL's SOPs.
- **Notes:** Uses `PLANNING_SYSTEM_PROMPT_ver3` with Examples placeholders, which enables pasting the episodic memory to enhance planning capability. 

---

### Research Agent (`core/agents/research_agent.py`)

- **Main role:** gathers biomedical context, builds or inspects knowledge graphs, and surfaces citations plus KG-derived candidates.
- **Tools:**
  - `literature_search_pubmed` – RAG tools for all available publications on PubMed.
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

### Data Agent (`core/agents/data_agent.py`)

- **Main role:** performs data analysis and visualization inside a sandboxed workspace.
- **Tools:**
  - `python_executor` – sandboxed Python REPL (pandas, NumPy, RDKit, scikit-learn, etc.) that preserves state between calls.
  - `reset_python_state` – nukes the Python namespace to recover from errors or keep memory low.

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
  - `AMES_classifier` – CPSign classification model for Ames mutagenicity.
  - `PGP_classifier` – CPSign classification model for P-gp substrate.
  - `PAMPA_classifier` – CPSign classification model for PAMPA permeability.
  - `BBB_classifier` – CPSign classification model for Blood–brain Barrier Penetration.
  - `Solubility_regressor` – CPSign regression model for solubility (output logS).
  - `Lipophilicity_regressor` – RDKit-backed logP estimator.
- **Notes:** Every tool writes its output to a CSV file (stored in `results/<task_id>/output.csv` for the local version). In both Local and Web apps, the output files are displayed in the UI so users can easily download them to their own devices.

---

### Report Agent (`core/agents/report_agent.py`)

- **Main role:** assembles the final narrative brief once execution completes.
- **Tools:** *(none – relies on the conversation history provided by the supervisor).*
- **Notes:** Guided by `REPORT_SYSTEM_PROMPT`, it summarizes findings, caveats, and recommended next steps.

---

### Supervisor Agent (`core/supervisor/supervisor.py`)

- **Main role:** orchestrates routing between sub-agents.
- **Tools:**
  - `transfer_to_data_agent` – delegate tasks and contexts to the Data Agent. 
  - `transfer_to_prediction_agent` – delegate tasks and contexts to the Prediction Agent. 
  - `transfer_to_research_agent` – delegate tasks and context to the Research Agent. 
  - `transfer_to_report_agent` – delegate tasks and context to the Report Agent. 

