# Repuragent

## Overview

Repuragent is a multi-agentic AI system designed to tackle drug repurposing challenges. The system features a supervisor agent that coordinates four specialized sub-agents for targeted tasks:

- **Research Agent**: Conducts literature reviews and gathers scientific evidence from biomedical databases and publications
- **Prediction Agent**: Performs molecular property predictions using pre-trained machine learning models
- **Data Agent**: Manages data processing, visualising and integration.
- **Report Agent**: Generates comprehensive reports

Additionally, the system includes a dedicated planning agent responsible for decomposing complex high-level tasks into manageable subtasks. This agent leverages professional Standard Operating Procedures (SOPs) and academic publications to inform strategic planning decisions.

The system incorporates episodic memory capabilities that extract patterns from previous successful executions and apply these learned patterns to enhance planning and decision-making for future actions.

## Quick Start

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- OpenAI API key from [platform.openai.com](https://platform.openai.com/)

### Setup
```bash
# 1. Clone repository
git clone <repository-url>
cd repuragent

# 2. Create .env file
echo "OPENAI_API_KEY=your-api-key-here" > .env

# 3. Run application
docker-compose up --build
```

Open [http://localhost:8501](http://localhost:8501)

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

