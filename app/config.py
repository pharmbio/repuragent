'''Single source of truth for configuration.

Everything tunable lives here and is overridable from the environment, so behaviour
can be changed for an installation without editing code. Import this module for
`logger` too — it configures logging exactly once.

This is the **local, single-user** build: there are no accounts, no sessions and no
PostgreSQL. One SQLite file holds both the LangGraph checkpoints and the
conversation list, and nothing on disk is partitioned by who is using it — a
conversation id is the whole scope hierarchy.
'''

from __future__ import annotations

import hashlib
import logging
import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

load_dotenv(ENV_PATH)
load_dotenv()


def _bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Gradio pings its analytics endpoint on startup. This app handles unpublished
# research data on someone's own machine, so telemetry is off unless opted in.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
# Silences the tokenizers fork warning that HuggingFace emits under uvicorn.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("repuragent")

# --- Identity -----------------------------------------------------------------

APP_TITLE = os.environ.get("APP_TITLE", "Repuragent")
APP_DESCRIPTION = os.environ.get(
    "APP_DESCRIPTION",
    "An AI scientist for drug repurposing: plans a workflow, waits for your "
    "approval, then executes it across literature, knowledge graphs and ADMET models.",
)
GITHUB_URL = os.environ.get("GITHUB_URL", "https://github.com/pharmbio/repuragent")
DOCS_URL = os.environ.get("DOCS_URL", "https://repuragent.readthedocs.io/")

# --- Credentials --------------------------------------------------------------

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    # Better here than as an authentication error from OpenAI several layers deep.
    logger.warning("OPENAI_API_KEY is not set; model calls will fail.")

# --- Filesystem ---------------------------------------------------------------

PERSISTENCE_ROOT = Path(
    os.environ.get("PERSIST_ROOT") or os.environ.get("PERSISTENCE_ROOT") or REPO_ROOT / "persistence"
).resolve()
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PERSISTENCE_ROOT / "data")).resolve()
RESULTS_ROOT = Path(os.environ.get("RESULTS_ROOT", PERSISTENCE_ROOT / "results")).resolve()
MEMORY_ROOT = Path(os.environ.get("MEMORY_ROOT", PERSISTENCE_ROOT / "memory")).resolve()

for _directory in (PERSISTENCE_ROOT, DATA_ROOT, RESULTS_ROOT, MEMORY_ROOT):
    _directory.mkdir(parents=True, exist_ok=True)

# Backwards-compatible alias: several modules and the analysis notebooks refer to
# the memory directory by this name.
MEMORY_DIR = MEMORY_ROOT

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", REPO_ROOT / "models")).resolve()

# --- Database -----------------------------------------------------------------
# One SQLite file holds the LangGraph checkpoints *and* the conversation list, the
# way the web build's single PostgreSQL database does. Keeping them together is what
# lets deleting a conversation remove its checkpoints and its row in one place.
#
# The default is the path this app has always used, so an existing installation is
# picked up rather than started from scratch. `SQLITE_DB_PATH` is the name earlier
# versions used and is still honoured.

DATABASE_PATH = Path(
    os.environ.get("DATABASE_PATH")
    or os.environ.get("SQLITE_DB_PATH")
    or MEMORY_ROOT / "shortterm_memory" / "langgraph_checkpoints.db"
).resolve()
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

# How long a query may wait for the write lock. SQLite serialises writers, and the
# retention worker and a running conversation can want it at the same moment.
DATABASE_TIMEOUT_SECONDS = _float_env("DATABASE_TIMEOUT_SECONDS", 30.0)

# --- Models -------------------------------------------------------------------
# One knob per role. The planner and the data agent do the hardest reasoning; the
# routers and judges are single-decision calls and run on a smaller model.

PLANNING_MODEL = os.environ.get("PLANNING_MODEL", "gpt-5.2-2025-12-11")
SUPERVISOR_MODEL = os.environ.get("SUPERVISOR_MODEL", "gpt-5.2-2025-12-11")
DATA_MODEL = os.environ.get("DATA_MODEL", "gpt-5.2-2025-12-11")
RESEARCH_MODEL = os.environ.get("RESEARCH_MODEL", "gpt-5-mini-2025-08-07")
PREDICTION_MODEL = os.environ.get("PREDICTION_MODEL", "gpt-5-mini-2025-08-07")
REPORT_MODEL = os.environ.get("REPORT_MODEL", "gpt-5-mini-2025-08-07")
TASK_CLASSIFIER_MODEL = os.environ.get("TASK_CLASSIFIER_MODEL", "gpt-5-mini-2025-08-07")
APPROVAL_JUDGE_MODEL = os.environ.get("APPROVAL_JUDGE_MODEL", "gpt-5-nano-2025-08-07")
CONTEXT_SUMMARY_MODEL = os.environ.get("CONTEXT_SUMMARY_MODEL", "gpt-5.2-2025-12-11")
EPISODE_EXTRACTION_MODEL = os.environ.get("EPISODE_EXTRACTION_MODEL", "gpt-4o-mini-2024-07-18")

# SOP indexing
SOP_IMAGE_DESCRIPTION_MODEL = os.environ.get("SOP_IMAGE_DESCRIPTION_MODEL", "gpt-5-mini-2025-08-07")
SOP_EMBEDDING_MODEL = os.environ.get("SOP_EMBEDDING_MODEL", "text-embedding-3-large")

# Counts every superstep across the graph and its sub-agents, so roughly two per
# model+tool exchange. A plan-driven repurposing run delegates to three specialists
# many times over; 100 truncated those mid-execution.
RECURSION_LIMIT = _int_env("RECURSION_LIMIT", 250)

# --- Server -------------------------------------------------------------------

GRADIO_SERVER_NAME = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
GRADIO_SERVER_PORT = _int_env("GRADIO_SERVER_PORT", 7860)
UI_QUEUE_MAX_SIZE = _int_env("UI_QUEUE_MAX_SIZE", 128)
UI_CONCURRENCY_LIMIT = _int_env("UI_CONCURRENCY_LIMIT", 8)
FILE_LIST_REFRESH_INTERVAL_SECONDS = _float_env("FILE_LIST_REFRESH_INTERVAL_SECONDS", 1.0)

# --- Files --------------------------------------------------------------------

# Off by default. The shared web deployment expires uploads and results after two
# days so they do not accumulate; deleting a local user's own work on a timer is not
# a favour. Set a positive number of days to turn the sweep back on.
RESULT_RETENTION_DAYS = _int_env("RESULT_RETENTION_DAYS", 0)
RETENTION_INTERVAL_SECONDS = _int_env("RETENTION_INTERVAL_SECONDS", 1800)

DEFAULT_CONVERSATION_TITLE = os.environ.get("DEFAULT_CONVERSATION_TITLE", "New conversation")


def _download_token_secret() -> bytes:
    '''Key that signs download links.

    With no accounts a download link proves only that it was minted by this process
    and has not expired, which is what stops a crafted `?token=` from reading a file
    outside the two managed roots. An explicit `FILE_DOWNLOAD_SECRET` makes links
    survive a restart; without one the key is random per process, which is harmless
    because links expire in minutes.

    Returns:
    ----------
    secret (bytes): the key download links are signed with, from the environment or generated per process.
    '''

    configured = os.environ.get("FILE_DOWNLOAD_SECRET")
    if configured:
        return configured.encode("utf-8")
    logger.debug(
        "No FILE_DOWNLOAD_SECRET set; signing download links with a per-process key. "
        "Existing links stop working on restart."
    )
    return secrets.token_bytes(32)


DOWNLOAD_TOKEN_SECRET = _download_token_secret()
# A link is minted inside a time bucket rather than at the exact second, so an
# unchanged file list re-renders to byte-identical markup and the sidebar can be
# skipped instead of replaced. The TTL is sized so quantisation cannot shorten a
# link below its former fixed lifetime (TTL - BUCKET = 600 s).
DOWNLOAD_TOKEN_BUCKET_SECONDS = _int_env("DOWNLOAD_TOKEN_BUCKET_SECONDS", 300)
DOWNLOAD_TTL_SECONDS = _int_env("FILE_DOWNLOAD_TOKEN_TTL_SECONDS", 900)

# --- Streaming ----------------------------------------------------------------
# Tokens are coalesced before reaching Gradio: one UI update per token would
# re-render the whole timeline hundreds of times a second.

STREAM_TOKENS = _bool_env("STREAM_TOKENS", True)
STREAM_FLUSH_SECONDS = _float_env("STREAM_FLUSH_SECONDS", 0.12)
STREAM_FLUSH_CHARS = _int_env("STREAM_FLUSH_CHARS", 180)

# --- Context handling ---------------------------------------------------------

CONTEXT_COMPRESSION = _bool_env("CONTEXT_COMPRESSION", True)
SUMMARY_MAX_MESSAGES = _int_env("SUMMARY_MAX_MESSAGES", 200)
SUMMARY_TRIGGER_MIN_MESSAGES_FIRST = _int_env("SUMMARY_TRIGGER_MIN_MESSAGES_FIRST", 8)
SUMMARY_TRIGGER_MIN_MESSAGES = _int_env("SUMMARY_TRIGGER_MIN_MESSAGES", 20)
SUMMARY_TRIGGER_CHAR_LIMIT = _int_env("SUMMARY_TRIGGER_CHAR_LIMIT", 12000)
SUMMARY_SOURCE_MAX_CHARS = _int_env("SUMMARY_SOURCE_MAX_CHARS", 120000)
SUMMARY_SOURCE_MESSAGE_MAX_CHARS = _int_env("SUMMARY_SOURCE_MESSAGE_MAX_CHARS", 8000)
MEMORY_MAX_ITEMS = _int_env("MEMORY_MAX_ITEMS", 20)
MEMORY_OUTPUTS_MAX_ITEMS = _int_env("MEMORY_OUTPUTS_MAX_ITEMS", 20)

# How many completed exchanges survive compression verbatim. These are what a
# follow-up actually refers to ("now rank them by hERG risk"), and they are small,
# so they are never handed to the summarizer.
CONTEXT_KEEP_TURNS = _int_env("CONTEXT_KEEP_TURNS", 3)
CONTEXT_ANCHOR_REQUEST_MAX_CHARS = _int_env("CONTEXT_ANCHOR_REQUEST_MAX_CHARS", 4000)
CONTEXT_ANCHOR_REPORT_MAX_CHARS = _int_env("CONTEXT_ANCHOR_REPORT_MAX_CHARS", 6000)
CONTEXT_GOAL_MAX_CHARS = _int_env("CONTEXT_GOAL_MAX_CHARS", 2000)
CONTEXT_ARTIFACT_MAX_ITEMS = _int_env("CONTEXT_ARTIFACT_MAX_ITEMS", 25)

# Tool traffic inside the live run: the most recent results stay intact, older ones
# collapse to a stub. Nothing is dropped, so tool_call pairing stays valid.
TOOL_RESULT_MAX_CHARS = _int_env("TOOL_RESULT_MAX_CHARS", 50000)
# The hard ceiling on what a tool may write **into the conversation**, as opposed to
# what the model is shown. The two are different in a way that costs real disk: the
# limit above is a *view*, rebuilt per model call, while a ToolMessage is *state* —
# and LangGraph's SQLite checkpointer rewrites the whole state as one blob on every
# superstep. One 143 MB result from a run that ended a cell on a DataFrame therefore
# cost 20.6 GB across 126 checkpoints. Generously above the view limit, so nothing a
# model actually reads is ever affected by it.
TOOL_RESULT_PERSIST_MAX_CHARS = _int_env("TOOL_RESULT_PERSIST_MAX_CHARS", 200000)
TOOL_RESULT_RECENT_FULL = _int_env("TOOL_RESULT_RECENT_FULL", 6)
TOOL_RESULT_ELIDED_CHARS = _int_env("TOOL_RESULT_ELIDED_CHARS", 800)

# --- Episodic memory ----------------------------------------------------------

EPISODIC_MAX_EXAMPLES = _int_env("EPISODIC_MAX_EXAMPLES", 2)
EPISODIC_CACHE_TTL_SECONDS = _float_env("EPISODIC_CACHE_TTL_SECONDS", 300.0)

# --- Tools --------------------------------------------------------------------

# Hard cap on a single python_executor result before it reaches the transcript,
# which is replayed to the model on every later call in the run.
PYTHON_OUTPUT_MAX_CHARS = _int_env("PYTHON_OUTPUT_MAX_CHARS", 20000)
# Interpreter sessions kept resident, keyed by conversation.
PYTHON_SESSION_CACHE_SIZE = _int_env("PYTHON_SESSION_CACHE_SIZE", 32)
# Wall-clock ceiling for one python_executor call. LangChain runs sync tools in a
# thread pool, so cancelling a run does not kill the thread: without a ceiling a
# runaway loop holds the conversation's interpreter lock forever. Generous, because
# a genuine knowledge-graph traversal is slow (0 disables).
PYTHON_EXEC_TIMEOUT_SECONDS = _int_env("PYTHON_EXEC_TIMEOUT_SECONDS", 900)
# Save matplotlib figures a run leaves unsaved into the conversation output scope.
# The server is headless, so plt.show() is a no-op and an unsaved figure — often the
# actual deliverable — is lost silently.
FIGURE_AUTOSAVE = _bool_env("FIGURE_AUTOSAVE", True)

# Above this size read_files returns a preview envelope (metadata, head, tail and how
# to get the rest) instead of the whole file.
READ_FILES_PREVIEW_THRESHOLD_CHARS = _int_env("READ_FILES_PREVIEW_THRESHOLD_CHARS", 60000)
READ_FILES_PREVIEW_HEAD_LINES = _int_env("READ_FILES_PREVIEW_HEAD_LINES", 40)
READ_FILES_PREVIEW_TAIL_LINES = _int_env("READ_FILES_PREVIEW_TAIL_LINES", 10)

# CPSign (Java) ADMET models.
CPSIGN_JAR = Path(os.environ.get("CPSIGN_JAR", MODELS_ROOT / "CPSign" / "cpsign-2.0.0-fatjar.jar"))
CPSIGN_CONFIDENCE = _float_env("CPSIGN_CONFIDENCE", 0.80)
CPSIGN_TIMEOUT_SECONDS = _int_env("CPSIGN_TIMEOUT_SECONDS", 900)

__all__ = [name for name in dir() if name.isupper() or name in {"logger", "REPO_ROOT"}]
