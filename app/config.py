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
FILE_DOWNLOAD_SECRET = os.environ.get("FILE_DOWNLOAD_SECRET", "repuragent-download")
FILE_DOWNLOAD_TOKEN_TTL_SECONDS = int(os.environ.get("FILE_DOWNLOAD_TOKEN_TTL_SECONDS", "600"))

# UI/server controls
UI_QUEUE_MAX_SIZE = int(os.environ.get("UI_QUEUE_MAX_SIZE", "32"))
UI_CONCURRENCY_LIMIT = int(os.environ.get("UI_CONCURRENCY_LIMIT", "8"))
GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
SUPERVISOR_OUTPUT_MODE = os.environ.get("SUPERVISOR_OUTPUT_MODE", "full_history").strip().lower()
if SUPERVISOR_OUTPUT_MODE not in {"full_history", "last_message"}:
    logger.warning(
        "Invalid SUPERVISOR_OUTPUT_MODE=%s; falling back to full_history",
        SUPERVISOR_OUTPUT_MODE,
    )
    SUPERVISOR_OUTPUT_MODE = "full_history"
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/pharmbio/repuragent")
USER_GUIDE_URL = os.environ.get("USER_GUIDE_URL", "https://repuragent.readthedocs.io/")

# Application settings
APP_TITLE = "Repuragent"
LOGO_PATH = "images/logo.png"
RECURSION_LIMIT = int(os.environ.get("RECURSION_LIMIT", "100"))
