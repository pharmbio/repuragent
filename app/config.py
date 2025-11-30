import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables as early as possible
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API keys and external services
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")
LANGCHAIN_ENDPOINT = os.environ.get("LANGCHAIN_ENDPOINT")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "repuragent")
LANGCHAIN_TRACING_V2 = os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"

# Memory directory setup
MEMORY_DIR = Path("backend/memory")
MEMORY_DIR.mkdir(exist_ok=True)
SHORTTERM_MEMORY_DIR = MEMORY_DIR / "shortterm_memory"
SHORTTERM_MEMORY_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = SHORTTERM_MEMORY_DIR / "langgraph_checkpoints.db"
THREAD_IDS_FILE = SHORTTERM_MEMORY_DIR / "thread_ids.json"

# File retention and download security
RESULT_RETENTION_DAYS = int(os.environ.get("RESULT_RETENTION_DAYS", "2"))
FILE_DOWNLOAD_SECRET = os.environ.get("FILE_DOWNLOAD_SECRET", "repuragent-download")
FILE_DOWNLOAD_TOKEN_TTL_SECONDS = int(os.environ.get("FILE_DOWNLOAD_TOKEN_TTL_SECONDS", "600"))

# UI/server controls
UI_QUEUE_MAX_SIZE = int(os.environ.get("UI_QUEUE_MAX_SIZE", "32"))
UI_CONCURRENCY_LIMIT = int(os.environ.get("UI_CONCURRENCY_LIMIT", "8"))
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

# Application settings
APP_TITLE = "Repuragent"
LOGO_PATH = "images/logo.png"
RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "100"))
