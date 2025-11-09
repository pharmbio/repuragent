# Repuragent

## Overview

Repuragent is a multi-agentic AI system designed for drug repurposing sector. The system orchestrates five specialized AI agents through a sophisticated supervisor architecture to tackle complex data gathering and integration in pharmaceutical sector.

<figure style="text-align: center;">
  <img src="images/agent_architecture.png" width="500">
  <figcaption> Core architecture of the Repuragent multi-agent AI system.</figcaption>
</figure>

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
echo "OPENAI_API_KEY=your-openai-api-key-here" > .env
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

## Features
- Multi-agent AI system for drug discovery
- Conformal prediction with CPSign
- Knowledge graph integration
- Molecular property prediction (ADMET, toxicity, bioactivity)
- Interactive web interface

## Project Structure
```
repuragent/
├── app/           # Streamlit interface
├── core/          # AI agents and logic
├── backend/       # Memory and RAG systems
├── models/        # Machine learning models
│   └── CPSign/    # Molecular prediction models
├── data/          # Input data
├── results/       # Output files
└── main.py        # Entry point
```

## Manual Installation
For advanced users who prefer not to use Docker:
```bash
# Requires Python 3.11+ and Java 11+
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements.txt
streamlit run main.py
```

## Common Issues
- **Port 8501 in use**: Run `docker-compose down` first
- **API key errors**: Verify your OpenAI key is valid
- **Module errors**: Rebuild with `docker-compose up --build`

View logs: `docker-compose logs -f`

