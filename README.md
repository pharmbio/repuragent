# Repuragent - An AI Scientist for Drug Repurposing

## Overview

Drug repurposing offers an efficient strategy to accelerate therapeutic discovery by identifying new indications for existing drugs. However, the process remains hindered by the heterogeneity of biological and chemical data and the difficulty of forming early, evidence-based hypotheses about candidate drugs, targets, and clinical endpoints. We introduce Repuragent (Drug Repurposing Agentic System), a proof-of-concept multi-agent framework designed to autonomously plan, execute, and refine data-driven repurposing workflows under human-in-the-loop supervision. The system integrates autonomous research, data extraction, knowledge graph (KG) construction, and analytical reasoning with an adaptive long-term memory mechanism that improves the system over time.

<div align="center">
  <img src="images/agent_architecture.png" width="500">
</div>



### Demo Page with example output:
[RepurAgent Demo Page](https://repuragent.streamlit.app) with example output for a COVID-19 repurposing.

### Core Agent Architecture

- **Planning Agent**: Decomposes complex tasks using episodic memory learning, Standard Operating Procedures (SOPs), and academic publications.
- **Supervisor Agent**: Delegate tasks to specialized agents and track the complete status of the whole task sequence.
- **Research Agent**: Performs literature mining via PubMed, accesses knowledge graphs, and integrates biomedical databases.
- **Prediction Agent**: Executes molecular property predictions using pre-trained ML models for ADMET properties
- **Data Agent**: Manages multi-format data processing, SMILES standardization, and visualization
- **Report Agent**: Generates comprehensive analytical reports and visualizations

### Advanced Memory Systems

- **Episodic Memory**: Pattern extraction from successful executions to improve future planning
- **Short-term Memory**: SQLite-based conversation persistence with thread management
- **SOP RAG System**: Retrieval-augmented generation using professional Standard Operating Procedures

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- OpenAI API key from [platform.openai.com](https://platform.openai.com/)
- (Optional) LangSmith account for tracing from [smith.langchain.com](https://smith.langchain.com/)

### Initial Setup
```bash
# 1. Clone repository
git clone https://github.com/your-username/repuragent.git
cd repuragent

# 2. Create .env file with required API keys
## Mandatory API key
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env   

## Optional set-up
echo "LANGCHAIN_TRACING_V2=true" >> .env
echo "LANGCHAIN_ENDPOINT=https://api.smith.langchain.com" >> .env
echo "LANGCHAIN_API_KEY=your-langsmith-api-key-here" >> .env
echo "LANGCHAIN_PROJECT=repuragent" >> .env

# 3. Build and run Docker containers
docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501) to access the application.

### Daily Usage
```bash
# Start the application (after initial setup)
docker-compose up

# Stop the application
docker-compose down

# View logs
docker-compose logs -f
```

### LangSmith Setup (Optional)
1. Create account at [smith.langchain.com](https://smith.langchain.com/)
2. Get your API key from the settings page
3. Add the LangSmith variables to your `.env` file as shown above
4. Restart Docker containers to apply changes


## Project Structure
```
repuragent/
├── app/           # Streamlit UI interface
├── core/          # AI agents and logic
├── backend/       # Memory and RAG systems
├── models/        # Machine learning models
├── data/          # Input data will be stored here
├── results/       # Output files will be stored here
└── main.py        # Entry point
```

