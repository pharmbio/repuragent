# Overview

Repuragent is an Agentic AI system for Drug Repurposing. It uses a LangGraph supervisor
architecture at its core, where the supervisor agent drives a team of specialized agents.
These agents mine biomedical literature, traverse curated knowledge graphs, run ADMET
models, generate and execute Python code, and write human-readable reports. This chapter
orients you before you dive into edition-specific guides.

## 1.1 What the Application Delivers

- **Autonomous planning**  
  The planning agent decomposes a user's questions into a sequence of actions, enriched by
  Standard Operating Procedures and episodic memory of prior successful workflows.
- **Evidence gathering**  
  The research agent queries PubMed via LitSense, searches your SOP vault, and builds
  knowledge graphs (KGG) to surface drugs, proteins, and pathways.
- **Data/analysis automation**  
  The data agent executes safety-controlled Python code snippets (pandas, RDKit,
  scikit-learn, etc) directly inside the task workspace.
- **Predictive modeling**  
  The prediction agent calls pretrained CPSign classifiers/regressors plus RDKit-derived
  features to produce ADMET dashboards.
- **Reporting and oversight**  
  The supervisor agent ensures every step is auditable, pauses for human approval when
  needed, and the report agent synthesizes the whole task run at the end.

## 1.2 Two Versions

- **Local app:** the [GitHub Repo](https://github.com/pharmbio/repuragent) that you clone,
  run in Docker, and keep entirely on your workstation. Nothing leaves your machine beyond
  the API calls authorized by your own `OPENAI_API_KEY`.
- **Web app:** hosted at [repuragent.serve.scilifelab.se](https://repuragent.serve.scilifelab.se).
  It uses the same LangGraph architecture in the core, but is deployed on the SciLifeLab
  Serve infrastructure for multiple users. No installation is required for this version.
- **Local App Guide:** use the [Local App Guide](local.md) when you want full control and
  private processing (requires your own OpenAI API key).
- **Web App Guide:** use the [Web App Guide](web.md) when you need instant usage without
  managing hardware.

## 1.3 Feature Comparison

| Capability | Local app | Web app |
| --- | --- | --- |
| Distribution | GitHub source; you run it with Docker | Hosted service |
| Authentication | None (runs as the launching user) | Email + password |
| Short-term memory | Local SQLite | Supabase PostgreSQL |
| Long-term/episodic memory | Chroma DB | Chroma DB |
| Extra dependencies | Docker Desktop, `OPENAI_API_KEY`, optional LangSmith keys | NO |


### Typical Workflow

1. **Frame the problem** – describe the indication, constraints, or hypotheses in chat.
2. **Attach relevant data** – upload CSV/JSON/PDF files.
3. **Approve and monitor** – planning may pause for human review, modification, and approval.
4. **Review results** – download artifacts from the Conversation panel.
5. **Iterate or branch** – continue the same thread to preserve context, or start a new one when exploring a different hypothesis.
