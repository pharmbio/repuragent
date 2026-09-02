'''The per-browser session object.

One `UIState` per connected Gradio session. It holds the rendered timeline, the
sidebar's file lists, which threads are running, and the approval gate's payload.

Two fields need care:

* `pending_approval` is a **display projection**, never authoritative. Whether a
  thread is paused is re-read from the graph (`read_pending_approval`), because this
  copy is reset by thread switches and page reloads, and is never set at all when
  the interrupt lands while the user is looking elsewhere.
* `last_panel_markup` / `last_progress_markup` record what was last *sent* to the
  browser, so an unchanged side panel can be skipped. `None` means "nothing sent
  yet" and must force a send.
'''

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

from gradio.components.chatbot import ChatMessage

from app.app_config import AppRunConfig


@dataclass(slots=True)
class ConversationMeta:
    thread_id: str
    title: str
    created_at: str
    updated_at: str


@dataclass
class FileRecord:
    path: str
    hash: Optional[str]
    name: str
    uploaded_at: Optional[datetime] = None
    record_id: Optional[str] = None


@dataclass
class UIState:
    '''Container for the Gradio UI session state.'''

    # Conversations
    thread_ids: List[Dict[str, Any]] = field(default_factory=list)
    current_thread_id: Optional[str] = None
    selected_thread_id: Optional[str] = None

    # Rendered timeline
    messages: List[ChatMessage] = field(default_factory=list)
    message_lookup: Dict[str, int] = field(default_factory=dict)
    agent_blocks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    last_agent_block_id: Optional[str] = None
    tool_call_block_lookup: Dict[str, str] = field(default_factory=dict)
    message_seq: int = 0
    processed_message_ids: Set[str] = field(default_factory=set)
    processed_tools_ids: Set[str] = field(default_factory=set)
    streaming_message_lookup: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # The payload the graph passed to `interrupt()` while paused for plan review, or
    # None when nothing is waiting on the user.
    pending_approval: Optional[Dict[str, Any]] = None

    # Threads that produced output while the user was looking elsewhere.
    stale_threads: Set[str] = field(default_factory=set)

    # Markup last sent for each side panel; compared before re-sending.
    last_panel_markup: Optional[str] = None
    last_progress_markup: Optional[str] = None

    # Files
    thread_files: Dict[str, List[FileRecord]] = field(default_factory=dict)
    uploaded_files: List[FileRecord] = field(default_factory=list)

    # Runs
    last_run_at: Dict[str, datetime] = field(default_factory=dict)
    current_app_config: Optional[AppRunConfig] = None
    stop_signals: Dict[str, bool] = field(default_factory=dict)
    running_threads: Set[str] = field(default_factory=set)
    use_episodic_learning: bool = True

    @property
    def is_awaiting_approval(self) -> bool:
        return self.pending_approval is not None

    @property
    def is_running(self) -> bool:
        '''True when the thread on screen has a run in flight.

        Returns:
        ----------
        running (boolean): True when the thread on screen has a run in flight.
        '''

        return bool(self.current_thread_id and self.current_thread_id in self.running_threads)

    def thread_meta(self, thread_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not thread_id:
            return None
        return next(
            (thread for thread in self.thread_ids if thread.get("thread_id") == thread_id), None
        )

    def ensure_thread_storage(self, thread_id: str) -> None:
        if thread_id not in self.thread_files:
            self.thread_files[thread_id] = []

    def next_message_id(self, prefix: str = "msg") -> str:
        '''A UI-only identifier for ChatMessage metadata.

        Parameters:
        ---------
        prefix (str): what to prefix the identifier with.

        Returns:
        ----------
        message_id (str): a UI-only identifier for ChatMessage metadata.
        '''

        self.message_seq += 1
        return f"{prefix}:{self.message_seq}"
