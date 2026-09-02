'''What one run needs to know about itself.'''

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config import CONTEXT_COMPRESSION


@dataclass(slots=True)
class AppRunConfig:
    user_request: Optional[str]
    conversation_id: Optional[str]
    # Whether the planning agent is given precedent from episodic memory. A UI
    # toggle, carried into graph state rather than baked into the compiled graph —
    # which is what lets the graph be compiled once per process instead of once per
    # message.
    use_episodic_learning: bool = True
    use_context_compression: bool = CONTEXT_COMPRESSION
