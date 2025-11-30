from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import convert_to_messages
from langgraph.types import Command

from app.app_config import AppRunConfig
from app.config import RECURSION_LIMIT, logger
from core.supervisor.supervisor import create_app


@asynccontextmanager
async def app_session(app_config: AppRunConfig):
    """Create and clean up a LangGraph app for a single async operation."""
    app, memory, connection = await create_app(
        user_request=app_config.user_request,
        use_episodic_learning=app_config.use_episodic_learning,
    )
    try:
        yield app
    finally:
        if connection:
            await connection.close()


def _is_interrupt_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        keyword in message
        for keyword in ("nodeinterrupt", "interrupt", "interrupted", "human input required")
    )


async def stream_langgraph_events(
    app_config: AppRunConfig,
    stream_input: Any,
    thread_id: str,
    *,
    check_for_interrupts: bool = False,
):
    """Yield LangGraph stream chunks followed by a completion event."""
    if not thread_id:
        raise ValueError("No active conversation thread is selected.")

    config = {
        "configurable": {
            "thread_id": thread_id,
        },
        "recursion_limit": RECURSION_LIMIT,
    }

    try:
        async with app_session(app_config) as app:
            stream_iterator = app.astream(
                stream_input,
                config=config,
                stream_mode="updates",
            )

            async for chunk in stream_iterator:
                if isinstance(chunk, tuple) or not isinstance(chunk, dict):
                    continue
                yield ("chunk", chunk)

            interrupted = False
            if check_for_interrupts:
                try:
                    current_state = await app.aget_state(config)
                    if hasattr(current_state, "next") and current_state.next:
                        if "human_chat" in current_state.next:
                            interrupted = True
                except Exception as state_error:  # pragma: no cover - defensive
                    logger.warning("Could not check execution state: %s", state_error)

            yield ("complete", interrupted)

    except Exception as exc:
        if check_for_interrupts and _is_interrupt_exception(exc):
            yield ("complete", True)
            return
        raise


def build_stream_input(user_message: str, *, resume: bool = False) -> Any:
    """Utility for constructing graph input compatible with the Gradio UI."""
    if resume:
        return Command(resume=user_message)
    return {"messages": convert_to_messages([user_message])}
