# Overview

Repuragent is an Agentic AI system for Drug Repurposing. It uses Agentic Supervisor
architecture at its core, where the supervisor agent drives a team of five specialised agents, including a planning agent, a prediction agent, a research agent, a data agent, and a report agent. These agents dynamically plan the execution, run ADMET models, mine biomedical literature, traverse curated knowledge graphs, generate and execute Python code, and write human-readable reports. This chapter orients you before you dive into edition-specific guides.

## 1.1 What the Repuragent delivers

- **Autonomous planning**  
  The planning agent decomposes the user's question into a sequence of actions, enriched by
  Standard Operating Procedures (SOPs) and episodic memory of prior successful workflows.
- **Evidence gathering**  
  The research agent queries literature on PubMed, searches the SOPs vault, and builds
  knowledge graphs (KG) to surface drugs, proteins, and pathways.
- **Predictive modeling**  
  The prediction agent calls pretrained CPSign classifiers/regressors to produce ADMET properties predictions.
- **Data/analysis automation**  
  The data agent executes safety-controlled Python code snippets (pandas, RDKit,
  scikit-learn, etc) directly inside the task workspace.
- **Reporting and oversight**  
  The UI and supervisor agent design ensures every executed step is traceable, can be paused for human approval when needed, and that the report agent synthesises the entire task run at the end.

## 1.2 Two Versions

- **Local app:** the [GitHub Repo](https://github.com/pharmbio/repuragent) that you clone,
  build and run in Docker Container, and keep input/ouput, files, conversation history entirely on your device. Nothing leaves your machine beyond the API calls authorised by your own `OPENAI_API_KEY`. Use the [Local App Guide](local.md) when you want full control and private processing (requires your own OpenAI API key).
- **Web app:** hosted at [repuragent.serve.scilifelab.se](https://repuragent.serve.scilifelab.se).
  It uses the same core with Local, but is deployed on the SciLifeLab Serve infrastructure for multiple users. No installation is required for this version. Use the [Web App Guide](web.md) for instant use without managing hardware.

## 1.3 Feature Comparison

| Capability | Local app | Web app |
| --- | --- | --- |
| Distribution | GitHub source; run it with Docker | Hosted service |
| Authentication | None (single-user) | Email + password |
| Short-term memory | Local SQLite | Supabase PostgreSQL |
| Long-term/episodic memory | Chroma DB | Chroma DB |
| Extra dependencies | Docker Desktop, `OPENAI_API_KEY`, optional LangSmith keys | NO |