from __future__ import annotations

import asyncio
from contextlib import suppress
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import shutil
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
from gradio.themes.utils import colors
from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
import uvicorn

from app.app_config import AppRunConfig
from app.config import (
    APP_TITLE,
    FILE_DOWNLOAD_SECRET,
    FILE_DOWNLOAD_TOKEN_TTL_SECONDS,
    GRADIO_SERVER_NAME,
    GRADIO_SERVER_PORT,
    LOGO_PATH,
    UI_CONCURRENCY_LIMIT,
    UI_QUEUE_MAX_SIZE,
)
from app.langgraph_runner import build_stream_input, stream_langgraph_events, app_session
from app.state import FileRecord, UIState
from app.ui.chat_timeline import (
    append_user_message,
    process_chunk,
    rebuild_from_plain_messages,
    rebuild_from_raw_messages,
)
from backend.memory.episodic_memory.conversation import (
    create_new_conversation,
    load_conversation,
)
from backend.memory.episodic_memory.episodic_learning import get_orchestrator
from backend.memory.episodic_memory.thread_manager import (
    load_thread_ids,
    remove_thread_id,
    update_thread_title,
)
from backend.utils.output_paths import (
    get_results_root,
    list_task_files,
    remove_task_dir,
    set_current_task_id,
    reset_current_task_id,
)


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR = get_results_root()
INPUT_ROOT = DATA_DIR
LEGACY_INPUT_TASKS_ROOT = DATA_DIR / "tasks"
ALLOWED_DOWNLOAD_ROOTS = tuple(
    path.resolve() for path in (INPUT_ROOT, LEGACY_INPUT_TASKS_ROOT, RESULTS_DIR)
)
DOWNLOAD_ROUTE = "/api/files/download"
_DOWNLOAD_SECRET = (FILE_DOWNLOAD_SECRET or "repuragent-download").encode("utf-8")

EPISODIC_ORCHESTRATOR = None
FILES_ROUTER = APIRouter(prefix="/api/files")
FILE_LIST_REFRESH_INTERVAL_SECONDS = 1.0

INTRO_MARKDOWN = (
    """Hello! I'm **Repuragent** - your AI Agent for Drug Repurposing. My team includes:

    - **Planning Agent:** Decomposes given task into sub-tasks using knowledge from Standard Operating Procedures (SOPs) and biomedical literatures. 
    - **Supervisor Agent:** Keeps track and coordinates agent's plan. 
    - **Prediction Agent:** Makes ADMET predictions using pre-trained models.
    - **Research Agent:** Retrieves relevant Standard Operating Procedures (SOPs), biomedical data from multiple database, and knowledge graph analysis.
    - **Data Agent:** Performs data manipulation, preprocessing, and analysis.
    - **Report Agent:** Summarizes agent workflow and wrtie final report. 

    How can I assist you today?"""
)

INTRO_SKIP_TEXTS = {INTRO_MARKDOWN.strip()}

PRIMARY_FERN = colors.Color(
    c50="#dbeee5",
    c100="#cfe3d9",
    c200="#bad4c7",
    c300="#9fc3b2",
    c400="#78a78f",
    c500="#3f7f6e",
    c600="#1f5c55",
    c700="#184842",
    c800="#10322d",
    c900="#0a211f",
    c950="#05110f",
    name="repuragent_primary_green",
)

SECONDARY_SAGE = colors.Color(
    c50="#edf6f2",
    c100="#dfeee6",
    c200="#c8dfd3",
    c300="#b2d1c0",
    c400="#95bfa9",
    c500="#79ad92",
    c600="#5e967c",
    c700="#4b7761",
    c800="#365646",
    c900="#233a2f",
    c950="#142019",
    name="repuragent_secondary_green",
)

REPURAGENT_THEME = (
    gr.themes.Default(
        primary_hue=PRIMARY_FERN,
        secondary_hue=SECONDARY_SAGE,
        neutral_hue=colors.gray,
    ).set(
        color_accent="*primary_600",
        color_accent_soft="#dbeee5",
        color_accent_soft_dark="*primary_700",
        button_primary_background_fill="*primary_600",
        button_primary_background_fill_hover="*primary_500",
        button_primary_text_color="#f6fbf8",
        button_primary_text_color_hover="#f6fbf8",
    )
)


def _logo_html() -> str:
    """Embed the logo as inline HTML to avoid Gradio's image toolbar."""
    logo_path = Path(LOGO_PATH)
    if not logo_path.exists():
        return ""
    data = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    mime, _ = mimetypes.guess_type(str(logo_path))
    mime = mime or "image/png"
    return f'<img src="data:{mime};base64,{data}" alt="{APP_TITLE} logo" class="app-logo-img" />'


def _get_orchestrator():
    global EPISODIC_ORCHESTRATOR
    if EPISODIC_ORCHESTRATOR is None:
        EPISODIC_ORCHESTRATOR = get_orchestrator()
    return EPISODIC_ORCHESTRATOR


def _sanitize_filename(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in (" ", ".", "_", "-")).strip() or "file"


def _hash_file(path: Path) -> str:
    hasher = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _urlsafe_b64encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _urlsafe_b64decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode(data + padding)


def _safe_resolve(path_value: str) -> Path:
    return Path(path_value).expanduser().resolve()


def _is_allowed_download_path(path: Path) -> bool:
    for root in ALLOWED_DOWNLOAD_ROOTS:
        try:
            path.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _encode_download_token(payload: Dict[str, Any]) -> str:
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(_DOWNLOAD_SECRET, body, hashlib.sha256).digest()
    return f"{_urlsafe_b64encode(body)}.{_urlsafe_b64encode(signature)}"


def _decode_download_token(token: str) -> Dict[str, Any]:
    try:
        body_part, sig_part = token.split(".", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Malformed download token") from exc
    body = _urlsafe_b64decode(body_part)
    provided_sig = _urlsafe_b64decode(sig_part)
    expected_sig = hmac.new(_DOWNLOAD_SECRET, body, hashlib.sha256).digest()
    if not hmac.compare_digest(provided_sig, expected_sig):
        raise HTTPException(status_code=403, detail="Invalid download token")
    try:
        payload = json.loads(body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="Corrupted download token") from exc
    expires_at = int(payload.get("exp", 0))
    if not expires_at or expires_at < int(time.time()):
        raise HTTPException(status_code=401, detail="Download link expired")
    return payload


def _build_download_payload(record: FileRecord) -> Optional[Dict[str, Any]]:
    if not record.path:
        return None
    resolved_path = _safe_resolve(record.path)
    if not _is_allowed_download_path(resolved_path):
        return None
    return {
        "path": str(resolved_path),
        "name": record.name,
        "exp": int(time.time()) + FILE_DOWNLOAD_TOKEN_TTL_SECONDS,
    }


def _input_task_root(thread_id: str) -> Path:
    root = INPUT_ROOT / thread_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _legacy_input_task_root(thread_id: str) -> Path:
    return LEGACY_INPUT_TASKS_ROOT / thread_id


def _input_files_dir(thread_id: str) -> Path:
    directory = _input_task_root(thread_id) / "files"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _legacy_input_files_dir(thread_id: str) -> Path:
    return _legacy_input_task_root(thread_id) / "files"


def _list_input_files(thread_id: str) -> List[Path]:
    directories = []
    current_dir = _input_files_dir(thread_id)
    directories.append(current_dir)
    legacy_dir = _legacy_input_files_dir(thread_id)
    if legacy_dir != current_dir and legacy_dir.exists():
        directories.append(legacy_dir)
    files: List[Path] = []
    for directory in directories:
        if not directory.exists():
            continue
        files.extend(path for path in directory.rglob("*") if path.is_file())
    files.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return files


def _hydrate_input_files(state: UIState, thread_id: str) -> None:
    records: List[FileRecord] = []
    for path in _list_input_files(thread_id):
        try:
            file_hash = _hash_file(path)
        except OSError:
            continue
        records.append(
            FileRecord(
                path=str(path),
                hash=file_hash,
                name=path.name,
            )
        )
    state.thread_files[thread_id] = records
    if state.current_thread_id == thread_id:
        state.uploaded_files = list(records)


def _clear_input_files(thread_id: str) -> None:
    current_dir = _input_files_dir(thread_id)
    if current_dir.exists():
        shutil.rmtree(current_dir, ignore_errors=True)
    current_dir.mkdir(parents=True, exist_ok=True)
    legacy_dir = _legacy_input_files_dir(thread_id)
    if legacy_dir.exists():
        shutil.rmtree(legacy_dir, ignore_errors=True)


def _remove_input_task_dir(thread_id: str) -> None:
    current_root = INPUT_ROOT / thread_id
    if current_root.exists():
        shutil.rmtree(current_root, ignore_errors=True)
    legacy_root = _legacy_input_task_root(thread_id)
    if legacy_root.exists():
        shutil.rmtree(legacy_root, ignore_errors=True)


def _save_uploaded_file(uploaded_file, thread_id: str) -> Tuple[Path, str]:
    """Persist uploaded file to the data directory."""
    target_dir = _input_files_dir(thread_id)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_name = getattr(uploaded_file, "orig_name", None) or os.path.basename(uploaded_file.name)
    filename, ext = os.path.splitext(orig_name)
    safe_name = _sanitize_filename(filename)
    final_name = f"{safe_name}_{timestamp}{ext}"
    destination = target_dir / final_name
    shutil.copy(uploaded_file.name, destination)
    return destination, _hash_file(destination)


MAX_VISIBLE_FILES = 50


def _render_thread_files(state: UIState, thread_id: str) -> str:
    files = state.thread_output_files.get(thread_id, [])
    if not files:
        return "<p class='conversation-card__empty'>No output files yet.</p>"
    items: List[str] = []
    limited_files = files[:MAX_VISIBLE_FILES]
    for record in limited_files:
        path_obj = Path(record.path) if record.path else None
        if path_obj:
            try:
                rel_display = str(path_obj.relative_to(RESULTS_DIR))
            except ValueError:
                rel_display = str(path_obj)
        else:
            rel_display = record.name
        payload = _build_download_payload(record)
        if payload:
            token = _encode_download_token(payload)
            name_markup = (
                "<a class='conversation-card__file-link' href='{href}' "
                "target='_blank' rel='noopener' data-download-link='{token}' "
                "data-file-name='{download_name}' download='{download_name}'>"
                "{label}</a>"
            ).format(
                href=escape(f"{DOWNLOAD_ROUTE}?token={token}", quote=True),
                token=escape(token, quote=True),
                label=escape(record.name),
                download_name=escape(record.name, quote=True),
            )
        else:
            name_markup = "<span class='conversation-card__file-name'>{}</span>".format(
                escape(record.name)
            )
        items.append(
            "<li class='conversation-card__file-item' title='{rel}'>"
            "{name}"
            "</li>".format(
                rel=escape(rel_display),
                name=name_markup,
            )
        )
    remaining = len(files) - MAX_VISIBLE_FILES
    more_indicator = ""
    if remaining > 0:
        more_indicator = f"<li class='conversation-card__file-more'>+{remaining} more…</li>"
    return (
        "<div class='conversation-card__files-container'>"
        "<ul class='conversation-card__files'>{}</ul>"
        "{}"
        "</div>"
    ).format("".join(items), more_indicator)


def _snapshot_output_files(thread_id: str) -> List[FileRecord]:
    try:
        disk_files = list_task_files(thread_id)
    except Exception:
        disk_files = []
    records: List[FileRecord] = []
    for path in disk_files:
        try:
            resolved = path.resolve()
        except OSError:
            resolved = path
        try:
            stamp = str(path.stat().st_mtime_ns)
        except OSError:
            stamp = None
        records.append(
            FileRecord(
                path=str(resolved),
                hash=stamp,
                name=path.name,
            )
        )
    return records


def _refresh_output_files_for(state: UIState, thread_id: Optional[str]) -> bool:
    if not thread_id:
        return False
    new_records = _snapshot_output_files(thread_id)
    previous = state.thread_output_files.get(thread_id, [])
    if new_records != previous:
        state.thread_output_files[thread_id] = new_records
        return True
    return False


def _refresh_all_output_files(state: UIState) -> None:
    for thread in state.thread_ids:
        tid = thread.get("thread_id")
        if not tid:
            continue
        state.thread_output_files.setdefault(tid, [])
        _refresh_output_files_for(state, tid)


def _conversation_panel_markup(state: UIState) -> str:
    cards: List[str] = [
        "<div class='conversation-list__container' id='conversation-list-root'>",
        "<div class='conversation-list__header'>Conversation</div>",
    ]
    if not state.thread_ids:
        cards.append("<p class='conversation-card__empty'>No conversations yet.</p></div>")
        return "\n".join(cards)
    for thread in state.thread_ids:
        thread_id = thread["thread_id"]
        is_active = thread_id == state.current_thread_id
        title = escape(thread["title"])
        state.thread_output_files.setdefault(thread_id, [])
        file_block = _render_thread_files(state, thread_id)
        cards.append(
            "<details class='conversation-card {active}' data-thread-id='{tid}' {open_attr}>"
            "<summary>"
            "<div class='conversation-card__title-row'>"
            "<span class='conversation-card__chevron' aria-hidden='true'></span>"
            "<span class='conversation-card__title'>{title}</span>"
            "<button type='button' class='conversation-card__delete' "
            "data-delete-thread='{tid}' data-confirm-message='Delete this conversation?'>🗑️</button>"
            "</div>"
            "</summary>"
            "<div class='conversation-card__body'>{files}</div>"
            "</details>".format(
                active="is-active" if is_active else "",
                tid=escape(thread_id),
                open_attr="open" if is_active else "",
                title=title,
                files=file_block,
            )
        )
    cards.append("</div>")
    return "\n".join(cards)


def _conversation_panel_update(state: UIState):
    return gr.update(value=_conversation_panel_markup(state))


@FILES_ROUTER.get("/download")
async def download_file(token: str):
    payload = _decode_download_token(token)
    path_value = payload.get("path")
    if not path_value:
        raise HTTPException(status_code=400, detail="Missing file path")
    resolved_path = _safe_resolve(path_value)
    if not _is_allowed_download_path(resolved_path):
        raise HTTPException(status_code=403, detail="Access denied")
    if not resolved_path.exists() or not resolved_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    filename = payload.get("name") or resolved_path.name
    mime, _ = mimetypes.guess_type(filename)
    return FileResponse(resolved_path, filename=filename, media_type=mime or "application/octet-stream")


_CONVERSATION_SCRIPT = """
<script>
(function() {
    function findBus() {
        const el = document.getElementById("conversation-action-bus");
        if (!el) {
            return null;
        }
        if (el.matches && el.matches("textarea, input")) {
            return el;
        }
        return el.querySelector ? el.querySelector("textarea, input") : null;
    }

    function sendAction(payload) {
        const bus = findBus();
        if (!bus) {
            return;
        }
        const enriched = Object.assign({ ts: Date.now() }, payload || {});
        bus.value = JSON.stringify(enriched);
        bus.dispatchEvent(new Event("input", { bubbles: true }));
        bus.dispatchEvent(new Event("change", { bubbles: true }));
    }
    function filenameFromDisposition(disposition) {
        if (!disposition) {
            return "";
        }
        const utf8Match = /filename\\*=UTF-8''([^;]+)/i.exec(disposition);
        if (utf8Match && utf8Match[1]) {
            try {
                return decodeURIComponent(utf8Match[1]);
            } catch (error) {
                return utf8Match[1];
            }
        }
        const basicMatch = /filename="?([^";]+)"?/i.exec(disposition);
        if (basicMatch && basicMatch[1]) {
            return basicMatch[1];
        }
        return "";
    }

    async function triggerDownload(anchor) {
        const url = anchor.getAttribute("href");
        if (!url) {
            return;
        }
        anchor.dataset.downloading = "1";
        try {
            const response = await fetch(url, { credentials: "same-origin" });
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            const blob = await response.blob();
            const headerName = response.headers.get("content-disposition");
            const inferred = filenameFromDisposition(headerName);
            const preferred = anchor.getAttribute("data-file-name") || "";
            const filename = preferred || inferred || anchor.textContent.trim() || "download";
            const blobUrl = window.URL.createObjectURL(blob);
            const temp = document.createElement("a");
            temp.href = blobUrl;
            temp.download = filename;
            document.body.appendChild(temp);
            temp.click();
            window.setTimeout(() => {
                document.body.removeChild(temp);
                window.URL.revokeObjectURL(blobUrl);
            }, 0);
        } catch (error) {
            console.error("Download failed", error);
            window.alert("Unable to download file. Please try again.");
            window.open(url, "_blank", "noopener");
        } finally {
            delete anchor.dataset.downloading;
        }
    }

    function bindHandlers() {
        const root = document.getElementById("conversation-list-root");
        const bus = findBus();
        if (!root || !bus) {
            return;
        }

        root.querySelectorAll("summary").forEach((summary) => {
            if (summary.dataset.repBound === "1") {
                return;
            }
            summary.dataset.repBound = "1";
            summary.addEventListener("click", (event) => {
                if (event.target && event.target.closest("[data-delete-thread]")) {
                    return;
                }
                const parent = summary.closest("details");
                if (!parent) {
                    return;
                }
                const threadId = parent.getAttribute("data-thread-id");
                if (!threadId) {
                    return;
                }
                sendAction({ type: "activate", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-delete-thread]").forEach((button) => {
            if (button.dataset.repBound === "1") {
                return;
            }
            button.dataset.repBound = "1";
            button.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                const threadId = button.getAttribute("data-delete-thread");
                if (!threadId) {
                    return;
                }
                const confirmMessage = button.getAttribute("data-confirm-message");
                if (confirmMessage && !window.confirm(confirmMessage)) {
                    return;
                }
                sendAction({ type: "delete", thread_id: threadId });
            });
        });

        root.querySelectorAll("[data-download-link]").forEach((link) => {
            if (link.dataset.repDownloadBound === "1") {
                return;
            }
            link.dataset.repDownloadBound = "1";
            link.addEventListener("click", (event) => {
                event.preventDefault();
                event.stopPropagation();
                if (link.dataset.downloading === "1") {
                    return;
                }
                triggerDownload(link);
            });
        });
    }

    function ensureReady() {
        if (!document.getElementById("conversation-list-root") || !findBus()) {
            window.requestAnimationFrame(ensureReady);
            return;
        }
        bindHandlers();
    }

    ensureReady();
    if (window.__repConversationObserver) {
        window.__repConversationObserver.disconnect();
    }
    const observer = new MutationObserver(() => {
        window.requestAnimationFrame(bindHandlers);
    });
    observer.observe(document.body, { childList: true, subtree: true });
    window.__repConversationObserver = observer;
})();
</script>
"""


async def _refresh_conversation(state: UIState, thread_id: str) -> None:
    app_config = AppRunConfig(user_request=None, use_episodic_learning=False)
    async with app_session(app_config) as app:
        convo = await load_conversation(thread_id, app)
    state.current_thread_id = thread_id
    state.processed_message_ids = set()
    raw_messages = convo.get("raw_messages") or []
    if raw_messages:
        rebuild_from_raw_messages(state, raw_messages)
    else:
        rebuild_from_plain_messages(state, convo.get("messages", []))
    state.processed_message_ids = convo.get("processed_message_ids", set())
    state.processed_content_hashes = set()
    state.ensure_thread_storage(thread_id)
    state.thread_output_files.setdefault(thread_id, [])
    _hydrate_input_files(state, thread_id)
    _refresh_output_files_for(state, thread_id)
    state.uploaded_files = list(state.thread_files.get(thread_id, []))


def _initialize_state() -> UIState:
    state = UIState()
    threads = load_thread_ids()
    if not threads:
        new_conv = create_new_conversation()
        threads = load_thread_ids()
        state.current_thread_id = new_conv["thread_id"]
        rebuild_from_plain_messages(state, new_conv["messages"])
    state.thread_ids = threads
    if state.thread_ids and not state.current_thread_id:
        state.current_thread_id = state.thread_ids[-1]["thread_id"]
    for thread in state.thread_ids:
        tid = thread["thread_id"]
        state.ensure_thread_storage(tid)
        state.thread_output_files.setdefault(tid, [])
        _hydrate_input_files(state, tid)
    _refresh_all_output_files(state)
    return state


async def on_app_load():
    state = _initialize_state()
    if state.current_thread_id:
        await _refresh_conversation(state, state.current_thread_id)
    approve_update = gr.update(visible=state.waiting_for_approval)
    return (
        state,
        _conversation_panel_markup(state),
        list(state.messages),
        state.use_episodic_learning,
        gr.update(value=""),
        gr.update(value=""),
    )


def on_toggle_learning(use_learning: bool, state: UIState):
    state.use_episodic_learning = bool(use_learning)
    return state


async def _activate_thread(thread_id: Optional[str], state: UIState):
    if not thread_id:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    await _refresh_conversation(state, thread_id)
    state.waiting_for_approval = False
    state.current_app_config = None
    state.approval_interrupted = False
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


def on_new_task(state: UIState):
    new_conv = create_new_conversation()
    state.thread_ids = load_thread_ids()
    state.current_thread_id = new_conv["thread_id"]
    rebuild_from_plain_messages(state, new_conv["messages"])
    state.processed_content_hashes = set()
    state.waiting_for_approval = False
    state.approval_interrupted = False
    state.current_app_config = None
    state.thread_files[new_conv["thread_id"]] = []
    state.thread_output_files[new_conv["thread_id"]] = []
    _clear_input_files(state.current_thread_id)
    _hydrate_input_files(state, state.current_thread_id)
    state.uploaded_files = list(state.thread_files.get(state.current_thread_id, []))
    state.processed_message_ids = set()
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


async def _delete_thread(thread_id: Optional[str], state: UIState):
    if not thread_id or len(state.thread_ids) <= 1:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    remove_thread_id(thread_id)
    _remove_input_task_dir(thread_id)
    remove_task_dir(thread_id)
    state.thread_ids = load_thread_ids()
    state.thread_files.pop(thread_id, None)
    state.thread_output_files.pop(thread_id, None)
    if state.current_thread_id == thread_id and state.thread_ids:
        state.current_thread_id = state.thread_ids[-1]["thread_id"]
        await _refresh_conversation(state, state.current_thread_id)
    state.waiting_for_approval = False
    state.approval_interrupted = False
    state.current_app_config = None
    return (
        state,
        _conversation_panel_update(state),
        list(state.messages),
        gr.update(value=""),
    )


async def on_conversation_action(action_payload: str, state: UIState):
    payload = (action_payload or "").strip()
    if not payload:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
            gr.update(value=""),
        )
    try:
        action = json.loads(payload)
    except json.JSONDecodeError:
        return (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
            gr.update(value=""),
        )
    action_type = action.get("type")
    thread_id = action.get("thread_id")
    if action_type == "delete":
        result = await _delete_thread(thread_id, state)
    elif action_type == "activate":
        result = await _activate_thread(thread_id, state)
    else:
        result = (
            state,
            _conversation_panel_update(state),
            list(state.messages),
            gr.update(value=""),
        )
    return (*result, gr.update(value=""))


def on_files_uploaded(files, state: UIState):
    if not files:
        return state, _conversation_panel_update(state)
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _conversation_panel_update(state)
    state.ensure_thread_storage(current_thread)
    existing_hashes = {
        record.hash for record in state.thread_files.get(current_thread, []) if record.hash
    }
    for file_obj in files:
        destination, file_hash = _save_uploaded_file(file_obj, current_thread)
        if file_hash and file_hash in existing_hashes:
            destination.unlink(missing_ok=True)
            continue
        existing_hashes.add(file_hash)
    _hydrate_input_files(state, current_thread)
    return state, _conversation_panel_update(state)


def on_clear_files(state: UIState):
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _conversation_panel_update(state)
    _clear_input_files(current_thread)
    _hydrate_input_files(state, current_thread)
    return state, _conversation_panel_update(state)


def on_periodic_file_refresh(state: UIState):
    if not state or not state.current_thread_id:
        return state, gr.update()
    updated = _refresh_output_files_for(state, state.current_thread_id)
    if updated:
        return state, _conversation_panel_update(state)
    return state, gr.update()


def _append_file_paths(prompt: str, state: UIState) -> str:
    files = state.uploaded_files
    if not files:
        return prompt
    if len(files) == 1:
        return f"{prompt}\n\nUploaded file: {files[0].path}"
    addition = "\n\nUploaded files:\n" + "\n".join(f"- {file.path}" for file in files)
    return prompt + addition


async def _run_user_message(prompt: str, state: UIState, *, approve_signal: Optional[str] = None):
    prompt = (prompt or "").strip()
    if not prompt and not approve_signal:
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
        return

    if approve_signal:
        state.waiting_for_approval = False
        state.approval_interrupted = False
        app_config = state.current_app_config or AppRunConfig(
            user_request=None,
            use_episodic_learning=state.use_episodic_learning,
        )
        stream_input = build_stream_input(approve_signal, resume=True)
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )
    else:
        final_prompt = _append_file_paths(prompt, state)
        append_user_message(state, prompt)
        user_messages = [m for m in state.messages if m.role == "user"]
        if len(user_messages) == 1 and state.current_thread_id:
            title = prompt[:30] + "..." if len(prompt) > 30 else prompt
            update_thread_title(state.current_thread_id, title)
            state.thread_ids = load_thread_ids()
        app_config = AppRunConfig(
            user_request=prompt if state.use_episodic_learning else None,
            use_episodic_learning=state.use_episodic_learning,
        )
        state.current_app_config = app_config
        resume = state.waiting_for_approval
        state.waiting_for_approval = False
        state.approval_interrupted = False
        stream_input = build_stream_input(prompt if resume else final_prompt, resume=resume)
        yield (
            state,
            list(state.messages),
            gr.update(value=""),
            _conversation_panel_update(state),
        )

    state.current_app_config = app_config

    task_token = set_current_task_id(state.current_thread_id)
    try:
        stream_iter = stream_langgraph_events(
            app_config,
            stream_input,
            state.current_thread_id,
            check_for_interrupts=True,
        )
        stream_task = asyncio.create_task(stream_iter.__anext__())
        watch_thread_id = state.current_thread_id
        poll_task = (
            asyncio.create_task(asyncio.sleep(FILE_LIST_REFRESH_INTERVAL_SECONDS))
            if watch_thread_id
            else None
        )
        try:
            while stream_task:
                wait_tasks = [stream_task]
                if poll_task:
                    wait_tasks.append(poll_task)
                done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)
                if poll_task and poll_task in done:
                    poll_task = asyncio.create_task(asyncio.sleep(FILE_LIST_REFRESH_INTERVAL_SECONDS))
                    if _refresh_output_files_for(state, watch_thread_id):
                        yield (
                            state,
                            list(state.messages),
                            gr.update(value=""),
                            _conversation_panel_update(state),
                        )
                if stream_task in done:
                    try:
                        event_type, payload = stream_task.result()
                    except StopAsyncIteration:
                        stream_task = None
                        break
                    if event_type == "chunk":
                        additions = process_chunk(state, payload)
                        if additions:
                            yield (
                                state,
                                list(state.messages),
                                gr.update(value=""),
                                _conversation_panel_update(state),
                            )
                    elif event_type == "complete":
                        state.waiting_for_approval = bool(payload)
                        state.approval_interrupted = bool(payload)
                        yield (
                            state,
                            list(state.messages),
                            gr.update(value=""),
                            _conversation_panel_update(state),
                        )
                    stream_task = asyncio.create_task(stream_iter.__anext__())
        finally:
            if poll_task:
                poll_task.cancel()
                with suppress(asyncio.CancelledError):
                    await poll_task

        if _refresh_output_files_for(state, state.current_thread_id):
            yield (
                state,
                list(state.messages),
                gr.update(value=""),
                _conversation_panel_update(state),
            )
    finally:
        reset_current_task_id(task_token)


async def on_send_message(prompt: str, state: UIState):
    async for update in _run_user_message(prompt, state):
        yield update


def on_extract_learning(state: UIState):
    orchestrator = _get_orchestrator()
    if not state.current_thread_id:
        return "⚠️ No active thread."
    result = orchestrator.extract_current_conversation(state.current_thread_id)
    if result.get("success") and result.get("episodes_extracted", 0):
        return result.get("message", "✅ Pattern extracted!")
    return result.get("message", "No patterns extracted.")


def build_demo():
    extra_css = """
    :root {
        --app-font: "Inter", "Helvetica Neue", Arial, sans-serif;
    }
    body,
    .gradio-container,
    .gradio-container * {
        font-family: var(--app-font) !important;
    }
    .gradio-container {
        max-width: 1280px;
        width: 95vw;
        margin: 0 auto !important;
        padding-top: 0.05rem;
    }
    #app-header {
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 0.05rem;
    }
    #app-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0 !important;
    }
    #app-logo .app-logo-img {
        width: 90px;
        height: 90px;
        object-fit: contain;
        display: block;
    }
    #app-title {
        margin: 0 !important;
        padding: 0 !important;
        display: flex;
        align-items: center;
    }
    #app-title .app-title-text {
        font-size: 3.8rem;
        font-weight: 900;
        line-height: 1;
        margin: 0;
    }
    #intro-text,
    #intro-text p {
        margin-top: 0 !important;
        margin-bottom: 0.2rem !important;
    }
    #layout-row {
        gap: 1rem;
        align-items: flex-start;
        margin-top: 0.05rem;
    }
    #conversation-column {
        display: flex;
        flex-direction: column;
        gap: 0.85rem;
    }
    #sidebar-column {
        display: flex;
        flex-direction: column;
        gap: 0.65rem;
    }
    #chatbot-panel {
        font-size: 1rem;
        line-height: 1.5;
    }
    #chatbot-panel .prose,
    #chatbot-panel .prose p {
        font-size: inherit !important;
        line-height: inherit !important;
    }
    #chatbot-panel .bot-message *,
    #chatbot-panel .message.bot *,
    #chatbot-panel [data-testid*="assistant"],
    #chatbot-panel [data-testid*="assistant"] * {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    #chatbot-panel .user-message *,
    #chatbot-panel .message.user *,
    #chatbot-panel [data-testid*="user"],
    #chatbot-panel [data-testid*="user"] * {
        font-size: 1rem !important;
    }
    #conversation-action-bus {
        display: none !important;
    }
    """
    with gr.Blocks(
        title=APP_TITLE,
        theme=REPURAGENT_THEME,
        css=extra_css,
        head=_CONVERSATION_SCRIPT,
    ) as demo:
        state = gr.State()

        with gr.Row(elem_id="app-header"):
            logo_markup = _logo_html()
            if logo_markup:
                with gr.Column(scale=0, min_width=96):
                    gr.HTML(logo_markup, elem_id="app-logo")
            with gr.Column(scale=1):
                gr.HTML(f"<div class='app-title-text'>{APP_TITLE}</div>", elem_id="app-title")
        gr.HTML(
            """
            <style>
            details.tool-block {
                border: 1px solid #e5e7eb;
                border-radius: 8px;
                padding: 8px 12px;
                background: #f9fafb;
                margin: 10px 0;
            }
            details.tool-block summary {
                font-weight: 600;
                color: #374151;
                cursor: pointer;
            }
            details.tool-block pre {
                margin: 8px 0 0 0;
                font-size: 0.95rem;
                background: #f4f6fb;
                padding: 12px;
                border-radius: 8px;
                overflow-x: auto;
                white-space: pre-wrap;
                font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
            }
            .tool-code-block {
                background: #f8fafc;
                border: 1px solid #e5e7eb;
                border-radius: 10px;
                padding: 14px 16px;
                margin-top: 10px;
                overflow-x: auto;
            }
            .tool-code-label {
                font-size: 0.75rem;
                letter-spacing: 0.08em;
                font-weight: 600;
                color: #6b7280;
                margin-bottom: 6px;
            }
            .tool-code-block pre {
                margin: 0;
                font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace;
                font-size: 0.95rem;
                line-height: 1.5;
                color: #111827;
                background: transparent;
                white-space: pre;
            }
    #conversation-list {
        margin-top: 0.5rem;
        font-family: inherit;
        width: 100%;
        display: block;
    }
    #conversation-list, #conversation-list > div, #conversation-list-root {
        width: 100%;
        box-sizing: border-box;
    }
    #conversation-list-root {
        border: 1px solid #d1d5db;
        border-radius: 12px;
        background: #fff;
        overflow: hidden;
        box-shadow: 0 1px 2px rgb(15 23 42 / 0.04);
    }
    .conversation-list__header {
        font-weight: 600;
        padding: 0.75rem 0.85rem;
        border-bottom: 1px solid #e5e7eb;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        font-size: 0.85rem;
        color: #4b5563;
        background: #f9fafb;
    }
    details.conversation-card {
        border-bottom: 1px solid #f3f4f6;
    }
    details.conversation-card:last-child {
        border-bottom: none;
    }
    details.conversation-card summary {
        list-style: none;
        padding: 0.6rem 0.85rem;
        cursor: pointer;
        background: transparent;
        transition: background 0.2s ease, color 0.2s ease;
    }
    details.conversation-card summary::-webkit-details-marker {
        display: none;
    }
    details.conversation-card.is-active summary {
        background: #eef2ff;
        color: #111827;
    }
    .conversation-card__title-row {
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .conversation-card__title {
        font-size: 0.9rem;
        font-weight: 600;
        color: inherit;
        flex: 1;
    }
    .conversation-card__chevron {
        width: 12px;
        height: 12px;
        border-right: 2px solid currentColor;
        border-bottom: 2px solid currentColor;
        transform: rotate(45deg);
        transition: transform 0.2s ease;
    }
    details.conversation-card[open] .conversation-card__chevron {
        transform: rotate(-135deg);
    }
    .conversation-card__delete {
        border: 1px solid #d1d5db;
        border-radius: 4px;
        padding: 0.1rem 0.35rem;
        font-size: 0.8rem;
        background: #fff;
        cursor: pointer;
        color: #4b5563;
        transition: background 0.2s ease, color 0.2s ease;
    }
    .conversation-card__delete:hover {
        background: #f3f4f6;
        color: #111827;
    }
    .conversation-card__body {
        background: #f9fafb;
        padding: 0.5rem 0.85rem 0.85rem;
    }
    .conversation-card__files-container {
        max-height: 180px;
        overflow-y: auto;
        padding-right: 0.25rem;
    }
    .conversation-card__files {
        list-style: none;
        margin: 0;
        padding: 0;
    }
    .conversation-card__file-item {
        font-size: 0.82rem;
        padding: 0 0;
        color: #374151;
    }
    .conversation-card__file-name {
        font-weight: 500;
    }
    .conversation-card__file-link {
        font-weight: 600;
        color: #1d4ed8;
        text-decoration: none;
    }
    .conversation-card__file-link:hover,
    .conversation-card__file-link:focus {
        text-decoration: underline;
    }
    .conversation-card__file-more {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.3rem;
    }
    .conversation-card__empty {
        font-size: 0.82rem;
        color: #6b7280;
        margin: 0;
    }
    </style>
    """
)

        with gr.Row(elem_id="layout-row"):
            with gr.Column(scale=1, min_width=280, elem_id="sidebar-column"):
                use_learning = gr.Checkbox(label="Use Episodic Learning", value=True)
                extract_btn = gr.Button("📚 Extract Learning")
                learning_status = gr.Markdown()
                new_task_btn = gr.Button("New Task")
                conversation_list = gr.HTML(
                    value="",
                    elem_id="conversation-list",
                    min_height=10,
                    container=False,
                )
                conversation_action_bus = gr.Textbox(
                    value="",
                    show_label=False,
                    elem_id="conversation-action-bus",
                )
                file_refresh_timer = gr.Timer(
                    value=FILE_LIST_REFRESH_INTERVAL_SECONDS,
                    active=True,
                    render=False,
                )
                file_upload = gr.File(label="Upload files", file_count="multiple", file_types=["file"])
                clear_files_btn = gr.Button("Clear Files")

            with gr.Column(scale=4, elem_id="conversation-column"):
                gr.Markdown(INTRO_MARKDOWN, elem_id="intro-text")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=600,
                    type="messages",
                    elem_id="chatbot-panel",
                )
                user_input = gr.Textbox(label="Your message", lines=3)
                send_btn = gr.Button("Send", variant="primary")

        demo.load(
            on_app_load,
            inputs=None,
            outputs=[
                state,
                conversation_list,
                chatbot,
                use_learning,
                user_input,
                conversation_action_bus,
            ],
        )

        use_learning.change(
            on_toggle_learning,
            inputs=[use_learning, state],
            outputs=state,
        )

        conversation_action_bus.change(
            on_conversation_action,
            inputs=[conversation_action_bus, state],
            outputs=[state, conversation_list, chatbot, user_input, conversation_action_bus],
        )

        file_refresh_timer.tick(
            on_periodic_file_refresh,
            inputs=[state],
            outputs=[state, conversation_list],
            trigger_mode="always_last",
        )

        new_task_btn.click(
            on_new_task,
            inputs=state,
            outputs=[state, conversation_list, chatbot, user_input],
        )

        file_upload.upload(
            on_files_uploaded,
            inputs=[file_upload, state],
            outputs=[state, conversation_list],
        )

        clear_files_btn.click(
            on_clear_files,
            inputs=state,
            outputs=[state, conversation_list],
        )

        send_btn.click(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input, conversation_list],
        )
        user_input.submit(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input, conversation_list],
        )

        extract_btn.click(
            on_extract_learning,
            inputs=state,
            outputs=learning_status,
        )

    return demo


def launch():
    demo = build_demo().queue(
        max_size=UI_QUEUE_MAX_SIZE,
        default_concurrency_limit=UI_CONCURRENCY_LIMIT,
    )
    fastapi_app = FastAPI()
    fastapi_app.include_router(FILES_ROUTER)
    application = gr.mount_gradio_app(fastapi_app, demo, path="/")
    uvicorn.run(application, host=GRADIO_SERVER_NAME, port=GRADIO_SERVER_PORT, log_level="info")
