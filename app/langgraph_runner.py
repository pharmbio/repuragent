from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from langchain_core.messages import convert_to_messages
from langgraph.types import Command

from app.app_config import AppRunConfig
from app.config import RECURSION_LIMIT, logger
from core.supervisor.supervisor import create_app


def _resolve_agent_name(metadata: dict, default: str | None = None) -> str:
    """Best-effort mapping from LangGraph metadata to a display-friendly agent name."""
    node = metadata.get("langgraph_node")
    if node and node not in {"agent"}:
        return node

    triggers = metadata.get("langgraph_triggers") or ()
    for trig in triggers:
        if isinstance(trig, str) and "branch:to:" in trig:
            candidate = trig.split("branch:to:", 1)[-1]
            if candidate and candidate not in {"agent", "__start__", "__end__", "__pregel_pull"}:
                return candidate

    checkpoint = metadata.get("langgraph_checkpoint_ns")
    if isinstance(checkpoint, str) and ":" in checkpoint:
        prefix = checkpoint.split(":", 1)[0]
        if prefix and prefix not in {"agent", "__start__", "__end__", "__pregel_pull"}:
            return prefix

    path = metadata.get("langgraph_path")
    if isinstance(path, (list, tuple)):
        for element in reversed(path):
            if element not in {"__pregel_pull", "__start__", "__end__", "agent"}:
                return element

    return (node or default or "agent")


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
            event_iterator = app.astream_events(
                stream_input,
                config=config,
                version="v1",
            )

            async for event in event_iterator:
                event_type = event.get("event")
                data = event.get("data") or {}
                metadata = event.get("metadata") or {}
                agent_name = _resolve_agent_name(metadata, event.get("name"))

                if event_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    if chunk:
                        yield ("stream_token", {"agent": agent_name, "chunk": chunk})

                elif event_type == "on_chain_stream":
                    chunk = data.get("chunk")
                    if not chunk:
                        continue
                    if metadata.get("langgraph_node"):
                        # Wrap node-level chunks into the format expected by the UI.
                        payload = chunk
                        if not isinstance(chunk, dict) or "messages" not in chunk:
                            payload = {"messages": [chunk]}
                        yield ("chunk", {agent_name: payload})
                    else:
                        # Root graph stream may already be in the expected shape.
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
