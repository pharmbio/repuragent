# Repuragent

AI-powered drug repurposing system using multi-agent architecture.

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
├── CPSign/        # Molecular prediction models
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

