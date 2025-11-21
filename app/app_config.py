from dataclasses import dataclass
from typing import Optional


@dataclass
class AppRunConfig:
    """Configuration describing how to build a LangGraph app for a request."""

    user_request: Optional[str]
    use_episodic_learning: bool = True
