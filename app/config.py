import os
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Memory directory setup
MEMORY_DIR = Path("backend/memory")
MEMORY_DIR.mkdir(exist_ok=True)
SHORTTERM_MEMORY_DIR = MEMORY_DIR / "shortterm_memory"
SHORTTERM_MEMORY_DIR.mkdir(exist_ok=True)
SQLITE_DB_PATH = SHORTTERM_MEMORY_DIR / "langgraph_checkpoints.db"
THREAD_IDS_FILE = SHORTTERM_MEMORY_DIR / "thread_ids.json"

# Application settings
APP_TITLE = "Repuragent"
LOGO_PATH = "images/logo.png"
RECURSION_LIMIT = 100