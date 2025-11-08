# Repuragent - AI-Powered Drug Repurposing Platform

## Prerequisites

Before getting started, ensure you have:

1. **Docker Desktop** installed on your system
   - [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Includes Docker Compose automatically
   - **No Java or Python installation required** - Docker handles all dependencies

2. **OpenAI API Key**
   - Sign up at [OpenAI](https://platform.openai.com/)
   - Create an API key from your dashboard
   - Required for AI functionality

3. **Git** (for cloning the repository)
   - Most systems have this pre-installed
   - [Download Git](https://git-scm.com/downloads) if needed

**That's it!** Docker will automatically install and configure:
- Python 3.11 environment
- Java 11+ (for CPSign molecular predictions)
- All Python dependencies (langchain, streamlit, pandas, etc.)
- CPSign and machine learning models

---

Repuragent is an advanced AI-powered platform for drug repurposing that combines machine learning, conformal prediction, and knowledge graphs to accelerate the discovery of new therapeutic applications for existing drugs.

## Features

- **Multi-Agent AI System**: Leverages specialized AI agents for planning, research, prediction, and reporting
- **Conformal Prediction**: Uses CPSign for reliable molecular property predictions
- **Knowledge Graph Integration**: Incorporates biological and chemical knowledge networks
- **Interactive Web Interface**: Streamlit-based user interface for easy interaction
- **Molecular Property Prediction**: Predicts ADMET properties, toxicity, and bioactivity
- **Literature Integration**: Automated research and protocol retrieval

## System Requirements

**For Docker Installation (Recommended):**
- Docker Desktop (includes Docker Compose)
- OpenAI API key
- **No Java or Python installation required** - everything is handled by Docker

**For Manual Installation (Advanced Users):**
- Python 3.11+ 
- Java 11+ (for CPSign)
- Git

## Quick Start with Docker (Recommended)

### Prerequisites
- Docker Desktop installed on your system
- OpenAI API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Repuragent
```

### 2. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
# OpenAI API key (mandatory)
OPENAI_API_KEY=your-actual-api-key-here

# LANGSMITH TRACING API KEY (Optional)
#LANGSMITH_TRACING=true
#LANGSMITH_ENDPOINT="https://api.smith.langchain.com"
#LANGSMITH_API_KEY=tracking-api-key
#LANGSMITH_PROJECT=project-name
```

### 3. Build and Run with Docker Compose

**First Time Setup (builds everything - takes a few minutes):**
```bash
# Build and start the application
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

**Subsequent Runs (uses cached image - starts in seconds):**
```bash
# Just start the application (no rebuild needed)
docker-compose up

# Or run in detached mode
docker-compose up -d
```

> **Note**: You only need to use `--build` the first time or when you update the code. Docker caches the built image with all dependencies (Java, Python packages) so subsequent startups are very fast.

### 4. Access the Application
Open your web browser and navigate to:
```
http://localhost:8501
```

### 5. Stop the Application
```bash
# Stop the containers (your data in ./data and ./results is preserved)
docker-compose down

# Stop and remove volumes (if you want to reset data)
docker-compose down -v
```

## Alternative: Docker Run Command

If you prefer using docker run instead of docker-compose:

```bash
# Build the image
docker build -t repuragent .

# Run the container
docker run -p 8501:8501 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/results:/app/results \
  -v $(pwd)/.env:/app/.env \
  --name repuragent-app \
  repuragent
```

## Manual Installation (Advanced Users)

If you prefer to run without Docker:

### Prerequisites
- Python 3.11+
- Java 11+ (for CPSign)
- Git

### Steps
1. Clone the repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt` (if available) or install packages manually
5. Set up your `.env` file as described above
6. Run: `streamlit run main.py`

## Project Structure

```
Repuragent/
├── app/                    # Streamlit web application
│   ├── streamlit_app.py   # Main application entry point
│   ├── config.py          # Configuration settings
│   └── ui/                # UI components
├── backend/               # Backend services
│   ├── memory/            # Conversation and episodic memory
│   ├── sop_rag/          # Standard Operating Procedures RAG
│   └── utils/             # Utility functions and tools
├── core/                  # Core AI agents and logic
│   ├── agents/            # Specialized AI agents
│   ├── prompts/           # System prompts
│   └── supervisor/        # Agent coordination
├── CPSign/                # CPSign JAR and precomputed models
├── models/                # Trained machine learning models
├── data/                  # Input data directory
├── results/               # Output results directory
├── main.py               # Application entry point
├── Dockerfile            # Docker container definition
├── docker-compose.yml    # Docker Compose configuration
└── .env                  # Environment variables (create this)
```

## Usage

1. **Start the Application**: Access the web interface at http://localhost:8501
2. **Set Your API Key**: Ensure your OpenAI API key is properly configured in the `.env` file
3. **Upload Molecular Data**: Provide SMILES strings or molecular files
4. **Configure Analysis**: Select prediction models and analysis parameters
5. **Run Analysis**: Execute the AI-powered drug repurposing workflow
6. **Review Results**: Examine predictions, reports, and recommendations

## Key Components

- **CPSign Integration**: Uses CPSign 2.0.0 for conformal prediction of molecular properties
- **Multi-Modal AI**: Combines text analysis, molecular modeling, and knowledge graphs
- **Persistent Memory**: Maintains conversation history and learning across sessions
- **Extensible Architecture**: Modular design allows for easy addition of new agents and tools

## Data Persistence and Performance

### What Gets Saved Between Sessions
- **Your Data**: Files in `./data/` and `./results/` persist on your local machine
- **Docker Image**: All installed software (Java, Python packages) cached for fast restarts
- **Settings**: Your `.env` file with API keys

### Performance Notes
- **First Time**: `docker-compose up --build` takes 5-10 minutes (installs everything)
- **Subsequent Runs**: `docker-compose up` starts in seconds (uses cached image)
- **No Re-installation**: Java and Python packages stay installed in the cached Docker image

## Troubleshooting

### Common Issues

1. **Port Already in Use Error**
   ```
   Bind for 0.0.0.0:8501 failed: port is already allocated
   ```
   **Solutions:**
   ```bash
   # Option 1: Stop any existing containers and clean up
   docker-compose down --remove-orphans
   docker system prune -f
   
   # Option 2: Kill any process using port 8501
   # On Mac/Linux:
   lsof -ti:8501 | xargs kill -9
   # On Windows:
   netstat -ano | findstr :8501
   taskkill /PID <PID_NUMBER> /F
   
   # Option 3: Use a different port (edit docker-compose.yml)
   # Change "8501:8501" to "8502:8501" then access via http://localhost:8502
   ```

2. **Orphan Containers Warning**
   ```
   Found orphan containers ([repuragent-app]) for this project
   ```
   **Solution:**
   ```bash
   docker-compose up --remove-orphans
   ```

3. **Memory Issues**: Increase Docker memory allocation for large datasets
4. **API Key Issues**: Verify your OpenAI API key is valid and properly set
5. **Java Issues**: Ensure Java 11+ is available in the container (handled automatically)

6. **Module Import Errors**
   ```
   ModuleNotFoundError: No module named 'langgraph.checkpoint.sqlite'
   ```
   **Solution:**
   ```bash
   # The requirements.txt has been updated with missing dependencies
   # Rebuild the Docker image
   docker-compose down
   docker-compose up --build
   ```

### Logs
View application logs:
```bash
# Docker Compose logs
docker-compose logs -f

# Docker run logs
docker logs repuragent-app -f
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

[Add your license information here]

## Support

For issues and questions:
- Check the troubleshooting section
- Review the logs for error messages
- Create an issue on the repository

---

**Note**: This application requires an active OpenAI API key for full functionality. The system integrates multiple AI agents that rely on language models for analysis and decision-making.

