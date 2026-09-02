'''Conversations, uploads, and episodic-memory extraction.

Everything here is a Gradio handler or something a handler calls directly. There is
no sign-in: the app opens straight into the user's conversations, and a first run
creates one if the store is empty.
'''

from __future__ import annotations

import json
from typing import Optional

import gradio as gr

from app import timeline_store
from app.config import DEFAULT_CONVERSATION_TITLE, logger
from app.conversation_store import create_thread, delete_thread, load_threads
from app.downloads import is_data_path
from app.files import (
    clear_thread_uploads,
    delete_thread_data,
    hash_file,
    list_upload_files,
    refresh_thread_files,
    save_uploaded_file,
)
from app.langgraph_runner import read_pending_approval
from app.state import UIState
from app.ui.chat_timeline import reset_chat_messages
from app.ui.conversation_panel import (
    conversation_panel_update,
    invalidate_panel_cache,
    thread_to_dict,
)
from app.ui.progress_panel import progress_update
from app.ui.projection import render


def initialize_state() -> UIState:
    return UIState()


# --- Conversations ------------------------------------------------------------


async def refresh_conversation(state: UIState, thread_id: str) -> None:
    '''Load one conversation into view: timeline, files, approval state.

    Parameters:
    ---------
    state (UIState): the state to load into.
    thread_id (str): the conversation to bring into view.
    '''

    state.current_thread_id = thread_id
    state.stale_threads.discard(thread_id)
    state.processed_message_ids = set()
    state.processed_tools_ids = set()

    await timeline_store.restore(state, thread_id)

    state.ensure_thread_storage(thread_id)
    refresh_thread_files(state, thread_id)
    state.uploaded_files = [
        record for record in state.thread_files.get(thread_id, []) if is_data_path(record.path)
    ]

    # Restore the approval gate from the graph rather than clearing it, so a
    # conversation paused for plan review is still resumable — and still says so —
    # after a thread switch or a page reload.
    state.pending_approval = await read_pending_approval(thread_id)


async def sync_threads(state: UIState, ensure_one: bool = True) -> None:
    '''Reload the conversation list and bring the current one into view.

    Parameters:
    ---------
    state (UIState): the state to reload into.
    ensure_one (boolean): create an empty conversation when the store holds none.
    '''

    threads = await load_threads()
    if not threads and ensure_one:
        await create_thread(DEFAULT_CONVERSATION_TITLE)
        threads = await load_threads()

    state.thread_ids = [thread_to_dict(meta) for meta in threads]
    valid_ids = {thread["thread_id"] for thread in state.thread_ids}
    for thread_id in valid_ids:
        state.ensure_thread_storage(thread_id)
    state.stale_threads = {tid for tid in state.stale_threads if tid in valid_ids}

    if state.current_thread_id not in valid_ids:
        state.current_thread_id = state.thread_ids[0]["thread_id"] if state.thread_ids else None
    if state.selected_thread_id not in valid_ids:
        state.selected_thread_id = state.current_thread_id

    # Only the conversation actually on screen is scanned: walking every thread's
    # directories here put an O(conversations) filesystem crawl on startup.
    if state.current_thread_id:
        await refresh_conversation(state, state.current_thread_id)
    else:
        reset_chat_messages(state)
        state.uploaded_files = []


async def activate_thread(thread_id: Optional[str], state: UIState):
    if not thread_id or thread_id not in {thread["thread_id"] for thread in state.thread_ids}:
        return render(state)
    state.selected_thread_id = thread_id
    await refresh_conversation(state, thread_id)
    state.current_app_config = None
    # Switching conversations clears the composer: text drafted for one thread reads
    # as a mistake once another is on screen.
    return render(state, clear_input=True)


async def new_task(state: UIState):
    meta = await create_thread(DEFAULT_CONVERSATION_TITLE)
    state.current_thread_id = meta.thread_id
    state.selected_thread_id = meta.thread_id
    state.thread_ids = [thread_to_dict(meta)] + list(state.thread_ids)
    state.thread_files[meta.thread_id] = []
    state.stale_threads.discard(meta.thread_id)
    state.uploaded_files = []
    reset_chat_messages(state)
    state.pending_approval = None
    invalidate_panel_cache(state)
    return render(state, clear_input=True)


async def _delete_thread_action(thread_id: Optional[str], state: UIState):
    if not thread_id:
        return render(state)
    if len(state.thread_ids) <= 1:
        gr.Info("Keep at least one conversation — start a new task first.")
        return render(state)
    await delete_thread(thread_id)
    delete_thread_data(thread_id)
    invalidate_panel_cache(state)
    await sync_threads(state, ensure_one=True)
    return render(state)


# --- Handlers -----------------------------------------------------------------


async def on_app_load():
    state = initialize_state()
    await sync_threads(state, ensure_one=True)
    return (*render(state, clear_input=True), gr.update(value=""))


async def on_new_task(state: UIState):
    return await new_task(state or initialize_state())


async def on_conversation_action(action_payload: str, state: UIState):
    '''Handle a click routed through the sidebar's hidden action bus.

    Parameters:
    ---------
    action_payload (str): the encoded click, routed through the sidebar's hidden action bus.
    state (UIState): the state to act on.

    Returns:
    ----------
    updates (tuple): the render after handling the click.
    '''

    state = state or initialize_state()
    payload = (action_payload or "").strip()
    if not payload:
        return (*render(state), gr.update(value=""))
    try:
        action = json.loads(payload)
    except json.JSONDecodeError:
        return (*render(state), gr.update(value=""))

    action_type = action.get("type")
    thread_id = action.get("thread_id")
    if action_type == "delete":
        result = await _delete_thread_action(thread_id, state)
    elif action_type == "activate":
        result = await activate_thread(thread_id, state)
    else:
        result = render(state)
    # Clear the bus so the same click is not replayed on the next change event.
    return (*result, gr.update(value=""))


# --- Files --------------------------------------------------------------------


async def on_files_uploaded(files, state: UIState):
    state = state or initialize_state()
    if not files or not state.current_thread_id:
        return state, conversation_panel_update(state)

    existing = {hash_file(path) for path in list_upload_files(state.current_thread_id)}
    for file_object in files:
        destination, digest = save_uploaded_file(file_object, thread_id=state.current_thread_id)
        if digest in existing:
            # Uploading the same file twice would give the agents two paths to the
            # same data and no way to know they match.
            destination.unlink(missing_ok=True)
            continue
        existing.add(digest)
    refresh_thread_files(state, state.current_thread_id)
    return state, conversation_panel_update(state)


async def on_clear_files(state: UIState):
    state = state or initialize_state()
    if not state.current_thread_id:
        return state, conversation_panel_update(state)
    clear_thread_uploads(state.current_thread_id)
    refresh_thread_files(state, state.current_thread_id)
    return state, conversation_panel_update(state)


async def on_periodic_file_refresh(state: UIState):
    '''Keep the sidebar and the plan panel current while a run is in flight.

    Gated on there being a run: this ticks once a second for every connected
    browser, and unconditionally walking the conversation's directories meant idle
    sessions paid for a filesystem crawl forever.

    The plan panel refreshes on the same tick, so a step that resolves during a long
    stretch of tool calls shows up promptly rather than at the end.

    Parameters:
    ---------
    state (UIState): the state to refresh.

    Returns:
    ----------
    updates (tuple): the sidebar and plan panel, sent only when they changed.
    '''

    if state is None or not state.current_thread_id or not state.running_threads:
        return state, gr.skip(), gr.skip()
    files_changed = refresh_thread_files(state, state.current_thread_id)
    return (
        state,
        conversation_panel_update(state) if files_changed else gr.skip(),
        progress_update(state),
    )


# --- Episodic memory ----------------------------------------------------------


def on_toggle_learning(use_learning: bool, state: UIState):
    state = state or initialize_state()
    state.use_episodic_learning = bool(use_learning)
    return state


def on_extract_learning(state: UIState) -> str:
    '''Record how this conversation was planned, as precedent for future planning.

    Deliberately manual. Extraction reads the whole conversation and calls a model,
    and only the user knows whether a run went well enough to be worth learning from.

    Parameters:
    ---------
    state (UIState): the state holding the conversation to learn from.

    Returns:
    ----------
    message (str): what was recorded, kept as precedent for future planning.
    '''

    if state is None or not state.current_thread_id:
        return "No conversation selected."
    thread_id = state.current_thread_id
    try:
        from core.agents.context import clear_episodic_cache
        from persistence.memory.episodic_memory.episodic_learning import get_orchestrator

        result = get_orchestrator().extract_current_conversation(thread_id)
        # The planner's examples are cached per request text; a new episode must be
        # visible to the next plan.
        clear_episodic_cache()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Episode extraction failed: %s", exc)
        return f"Could not extract: {exc}"

    if result.get("success") and result.get("episodes_extracted"):
        return result.get("message") or "Saved. Future plans will use this as precedent."
    return result.get("message") or "Nothing worth recording from this conversation yet."


__all__ = [
    "initialize_state",
    "on_app_load",
    "on_clear_files",
    "on_conversation_action",
    "on_extract_learning",
    "on_files_uploaded",
    "on_new_task",
    "on_periodic_file_refresh",
    "on_toggle_learning",
    "sync_threads",
]
