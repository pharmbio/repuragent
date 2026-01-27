from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set

from gradio.components.chatbot import ChatMessage

from app.app_config import AppRunConfig


@dataclass
class FileRecord:
    path: str
    hash: Optional[str]
    name: str
    uploaded_at: Optional[datetime] = None
    record_id: Optional[str] = None


@dataclass
class UIState:
    """Container for the Gradio UI session state."""

    thread_ids: List[Dict] = field(default_factory=list)
    current_thread_id: Optional[str] = None
    selected_thread_id: Optional[str] = None
    messages: List[ChatMessage] = field(default_factory=list)
    message_lookup: Dict[str, int] = field(default_factory=dict)
    agent_blocks: Dict[str, Dict] = field(default_factory=dict)
    last_agent_block_id: Optional[str] = None
    tool_call_block_lookup: Dict[str, str] = field(default_factory=dict)
    message_seq: int = 0
    processed_message_ids: Set[str] = field(default_factory=set)
    processed_tools_ids: Set[str] = field(default_factory=set)
    processed_content_hashes: Set[int] = field(default_factory=set)
    streaming_message_lookup: Dict[str, Dict[str, int]] = field(default_factory=dict)
    waiting_for_approval: bool = False
    approval_interrupted: bool = False
    stale_threads: Set[str] = field(default_factory=set)
    pending_stream_events: Dict[str, List] = field(default_factory=dict)
    use_episodic_learning: bool = True
    thread_files: Dict[str, List[FileRecord]] = field(default_factory=dict)
    uploaded_files: List[FileRecord] = field(default_factory=list)
    last_run_at: Dict[str, datetime] = field(default_factory=dict)
    thread_output_files: Dict[str, List[FileRecord]] = field(default_factory=dict)
    current_app_config: Optional[AppRunConfig] = None
    stop_signals: Dict[str, bool] = field(default_factory=dict)
    running_threads: Set[str] = field(default_factory=set)

    def ensure_thread_storage(self, thread_id: str) -> None:
        if thread_id not in self.thread_files:
            self.thread_files[thread_id] = []

    def next_message_id(self, prefix: str = "msg") -> str:
        """Create a UI-only identifier for ChatMessage metadata."""
        self.message_seq += 1
        return f"{prefix}:{self.message_seq}"
