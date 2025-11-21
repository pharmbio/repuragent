from __future__ import annotations

import base64
import hashlib
import mimetypes
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr

from app.app_config import AppRunConfig
from app.config import APP_TITLE, LOGO_PATH
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


DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EPISODIC_ORCHESTRATOR = None

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


def _save_uploaded_file(uploaded_file) -> Tuple[Path, str]:
    """Persist uploaded file to the data directory."""
    DATA_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    orig_name = getattr(uploaded_file, "orig_name", None) or os.path.basename(uploaded_file.name)
    filename, ext = os.path.splitext(orig_name)
    safe_name = _sanitize_filename(filename)
    final_name = f"{safe_name}_{timestamp}{ext}"
    destination = DATA_DIR / final_name
    shutil.copy(uploaded_file.name, destination)
    return destination, _hash_file(destination)


def _format_file_list(state: UIState) -> str:
    files = state.thread_files.get(state.current_thread_id or "", [])
    if not files:
        return "_No files uploaded for this task._"
    lines = [f"- {record.name} (`{record.path}`)" for record in files]
    return "\n".join(lines)


def _resolve_thread_id(state: UIState, selection: Optional[str]) -> Optional[str]:
    if not selection:
        return None
    if selection in state.thread_choice_map:
        return state.thread_choice_map[selection]
    if "|||" in selection:
        return selection.split("|||", 1)[0]
    return selection


def _conversation_rows(state: UIState) -> List[List[str]]:
    rows: List[List[str]] = []
    name_counts: Dict[str, int] = {}
    for thread in state.thread_ids:
        title = thread["title"]
        count = name_counts.get(title, 0) + 1
        name_counts[title] = count
        label = title if count == 1 else f"{title} ({count})"
        prefix = "●" if thread["thread_id"] == state.current_thread_id else "○"
        rows.append([f"{prefix} {label}", "🗑️"])
    return rows


def _conversation_table_update(state: UIState):
    rows = _conversation_rows(state)
    return gr.update(value=rows, row_count=(len(rows), "dynamic"))


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
    state.uploaded_files = list(state.thread_files.get(thread_id, []))


def _initialize_state() -> UIState:
    state = UIState()
    threads = load_thread_ids()
    if not threads:
        new_conv = create_new_conversation()
        state.thread_ids = load_thread_ids()
        state.current_thread_id = new_conv["thread_id"]
        rebuild_from_plain_messages(state, new_conv["messages"])
        state.ensure_thread_storage(state.current_thread_id)
    else:
        state.thread_ids = threads
        state.current_thread_id = threads[-1]["thread_id"]
        state.ensure_thread_storage(state.current_thread_id)
    return state


async def on_app_load():
    state = _initialize_state()
    if state.current_thread_id:
        await _refresh_conversation(state, state.current_thread_id)
    attachments = _format_file_list(state)
    approve_update = gr.update(visible=state.waiting_for_approval)
    return (
        state,
        _conversation_table_update(state),
        list(state.messages),
        attachments,
        state.use_episodic_learning,
        gr.update(value=""),
    )


def on_toggle_learning(use_learning: bool, state: UIState):
    state.use_episodic_learning = bool(use_learning)
    return state


async def _activate_thread(thread_id: Optional[str], state: UIState):
    if not thread_id:
        return (
            state,
            _conversation_table_update(state),
            list(state.messages),
            _format_file_list(state),
            gr.update(value=""),
        )
    await _refresh_conversation(state, thread_id)
    state.waiting_for_approval = False
    state.current_app_config = None
    state.approval_interrupted = False
    attachments = _format_file_list(state)
    return (
        state,
        _conversation_table_update(state),
        list(state.messages),
        attachments,
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
    state.uploaded_files = []
    state.processed_message_ids = set()
    return (
        state,
        _conversation_table_update(state),
        list(state.messages),
        _format_file_list(state),
        gr.update(value=""),
    )


async def _delete_thread(thread_id: Optional[str], state: UIState):
    if not thread_id or len(state.thread_ids) <= 1:
        return (
            state,
            _conversation_table_update(state),
            list(state.messages),
            _format_file_list(state),
            gr.update(value=""),
        )
    remove_thread_id(thread_id)
    state.thread_ids = load_thread_ids()
    state.thread_files.pop(thread_id, None)
    if state.current_thread_id == thread_id and state.thread_ids:
        state.current_thread_id = state.thread_ids[-1]["thread_id"]
        await _refresh_conversation(state, state.current_thread_id)
    state.waiting_for_approval = False
    state.approval_interrupted = False
    state.current_app_config = None
    return (
        state,
        _conversation_table_update(state),
        list(state.messages),
        _format_file_list(state),
        gr.update(value=""),
    )


async def on_conversation_table_select(evt: gr.SelectData, state: UIState):
    if not state.thread_ids or evt.index is None:
        return (
            state,
            _conversation_table_update(state),
            list(state.messages),
            _format_file_list(state),
            gr.update(value=""),
        )
    row, col = evt.index
    if row is None or row >= len(state.thread_ids):
        return (
            state,
            _conversation_table_update(state),
            list(state.messages),
            _format_file_list(state),
            gr.update(value=""),
        )
    thread_id = state.thread_ids[row]["thread_id"]
    if col == 1:
        return await _delete_thread(thread_id, state)
    return await _activate_thread(thread_id, state)


def on_files_uploaded(files, state: UIState):
    if not files:
        return state, _format_file_list(state)
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _format_file_list(state)
    state.ensure_thread_storage(current_thread)
    existing_hashes = {record.hash for record in state.thread_files[current_thread]}
    for file_obj in files:
        destination, file_hash = _save_uploaded_file(file_obj)
        if file_hash in existing_hashes:
            destination.unlink(missing_ok=True)
            continue
        record = FileRecord(path=str(destination), hash=file_hash, name=os.path.basename(destination))
        state.thread_files[current_thread].append(record)
        existing_hashes.add(file_hash)
    state.uploaded_files = list(state.thread_files[current_thread])
    return state, _format_file_list(state)


def on_clear_files(state: UIState):
    current_thread = state.current_thread_id
    if not current_thread:
        return state, _format_file_list(state)
    state.thread_files[current_thread] = []
    state.uploaded_files = []
    return state, _format_file_list(state)


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
        )

    state.current_app_config = app_config

    async for event_type, payload in stream_langgraph_events(
        app_config,
        stream_input,
        state.current_thread_id,
        check_for_interrupts=True,
    ):
        if event_type == "chunk":
            additions = process_chunk(state, payload)
            if additions:
                yield (
                    state,
                    list(state.messages),
                    gr.update(value=""),
                )
        elif event_type == "complete":
            state.waiting_for_approval = bool(payload)
            state.approval_interrupted = bool(payload)
            yield (
                state,
                list(state.messages),
                gr.update(value=""),
            )


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
        padding-top: 1.25rem;
    }
    #app-header {
        align-items: center;
        gap: 0.85rem;
        margin-bottom: 0.85rem;
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
    #intro-text {
        margin-top: 0.25rem !important;
    }
    #layout-row {
        gap: 1rem;
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
    """
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Default(), css=extra_css) as demo:
        state = gr.State()

        with gr.Row(elem_id="app-header"):
            logo_markup = _logo_html()
            if logo_markup:
                with gr.Column(scale=0, min_width=96):
                    gr.HTML(logo_markup, elem_id="app-logo")
            with gr.Column(scale=1):
                gr.HTML(f"<div class='app-title-text'>{APP_TITLE}</div>", elem_id="app-title")
        gr.Markdown(INTRO_MARKDOWN, elem_id="intro-text")
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
            #conversation-table table,
            #conversation-table table td,
            #conversation-table table th {
                font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
                font-size: 0.9rem;
            }
            #conversation-table table td:last-child {
                text-align: center;
                width: 48px;
            }
            </style>
            """
        )

        with gr.Row(elem_id="layout-row"):
            with gr.Column(scale=1, min_width=240):
                use_learning = gr.Checkbox(label="Use Episodic Learning", value=True)
                extract_btn = gr.Button("📚 Extract Learning")
                learning_status = gr.Markdown()

                conversation_table = gr.Dataframe(
                    headers=["Conversation", ""],
                    datatype=["str", "str"],
                    label="Conversations",
                    interactive=False,
                    row_count=(0, "dynamic"),
                    col_count=(2, "fixed"),
                    elem_id="conversation-table",
                )
                new_task_btn = gr.Button("New Task")

                file_upload = gr.File(label="Upload files", file_count="multiple", file_types=["file"])
                clear_files_btn = gr.Button("Clear Files")
                files_md = gr.Markdown()

            with gr.Column(scale=4):
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
                conversation_table,
                chatbot,
                files_md,
                use_learning,
                user_input,
            ],
        )

        use_learning.change(
            on_toggle_learning,
            inputs=[use_learning, state],
            outputs=state,
        )

        conversation_table.select(
            on_conversation_table_select,
            inputs=state,
            outputs=[state, conversation_table, chatbot, files_md, user_input],
        )

        new_task_btn.click(
            on_new_task,
            inputs=state,
            outputs=[state, conversation_table, chatbot, files_md, user_input],
        )

        file_upload.upload(
            on_files_uploaded,
            inputs=[file_upload, state],
            outputs=[state, files_md],
        )

        clear_files_btn.click(
            on_clear_files,
            inputs=state,
            outputs=[state, files_md],
        )

        send_btn.click(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input],
        )
        user_input.submit(
            on_send_message,
            inputs=[user_input, state],
            outputs=[state, chatbot, user_input],
        )

        extract_btn.click(
            on_extract_learning,
            inputs=state,
            outputs=learning_status,
        )

    return demo


def launch():
    app = build_demo()
    server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    app.queue(max_size=32).launch(server_name=server_name, server_port=server_port, share=False)
