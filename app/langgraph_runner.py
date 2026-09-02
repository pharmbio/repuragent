'''Driving the graph and turning its output into three UI events.

`stream_langgraph_events` consumes
`astream(stream_mode=["messages", "updates"], subgraphs=True)` and emits exactly:

* **`chunk`** — what a node committed: AI messages, tool calls, tool results.
* **`token`** — partial model text, coalesced. Display-only.
* **`complete`** — the run settled, with whether it paused for approval.

**Why not `astream_events`.** It flattens the graph: every event inside an agent is
labelled `model` or `tools`, so the producing node had to be *inferred* from
metadata strings. `subgraphs=True` supplies the namespace
(`('execution:<uuid>', 'research_agent:<uuid>')`), which makes attribution a
lookup. It also avoids a duplicate-rendering bug: `astream_events` emits a node's
*return value*, which has no id until LangGraph merges it into state, so the UI had
to mint a synthetic id and then failed to match the same message when the parent
re-emitted it with its real UUID.

**Id de-duplication in the timeline is load-bearing, not an optimisation.** A node
wrapping a subgraph re-emits every message that subgraph produced — here the
`execution` node re-emits the entire conversation each time the supervisor commits.
Only id matching stops the transcript from being rendered several times over.
'''

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessageChunk, convert_to_messages
from langgraph.types import Command

from app.app_config import AppRunConfig
from app.config import (
    CONTEXT_COMPRESSION,
    RECURSION_LIMIT,
    STREAM_FLUSH_CHARS,
    STREAM_FLUSH_SECONDS,
    STREAM_TOKENS,
    logger,
)
from backend.db import get_checkpointer
from backend.utils.output_paths import (
    reset_current_conversation_id,
    set_current_conversation_id,
)
from core.agents.agentic_system import create_app
from core.agents.context import mark_user_request

# Node names that say nothing a user would recognise. In LangChain v1 agents the
# internal nodes are `model` and `tools`.
_INTERNAL_NODES = {"model", "tools", "agent", "__start__", "__end__", "__pregel_pull"}


def resolve_agent_name(
    namespace: Any,
    metadata: Optional[dict] = None,
    default: Optional[str] = None,
) -> str:
    '''Which graph node produced this event, in the UI's vocabulary.

    The node name wins when it is a real one. Inside an agent it is `model` or
    `tools`, so fall back to the **deepest** meaningful namespace entry: with the
    supervisor and its specialists nested in the `execution` subgraph, the deepest
    entry is `research_agent` where the shallowest would only ever say `execution`.

    Parameters:
    ---------
    namespace (Any): the subgraph namespace the event arrived under.
    metadata (dict): the event's metadata, holding the node name when there is one.
    default (str): what to fall back to when neither identifies a node.

    Returns:
    ----------
    name (str): the producing node in the UI's vocabulary. Prefers the node name and falls back to the **deepest** namespace entry, because with the nested `execution` subgraph the shallowest would only ever say `execution`.
    '''

    node = (metadata or {}).get("langgraph_node")
    if node and node not in _INTERNAL_NODES:
        return node
    for entry in reversed(tuple(namespace or ())):
        candidate = str(entry).split(":", 1)[0]
        if candidate and candidate not in _INTERNAL_NODES:
            return candidate
    return node or default or "agent"


def _stream_chunk_text(chunk: Any) -> str:
    '''Text carried by a streamed model chunk, ignoring tool-call deltas.

    Parameters:
    ---------
    chunk (Any): one streamed model chunk.

    Returns:
    ----------
    text (str): the text it carries, ignoring tool-call deltas.
    '''

    if chunk is None:
        return ""
    content = getattr(chunk, "content", None)
    if content is None and isinstance(chunk, dict):
        content = chunk.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return ""


# --- The compiled graph -------------------------------------------------------

_checkpointer_lock: Optional[asyncio.Lock] = None
_app_cache: Dict[bool, Any] = {}
_app_cache_lock: Optional[asyncio.Lock] = None


async def _get_checkpointer():
    global _checkpointer_lock
    if _checkpointer_lock is None:
        _checkpointer_lock = asyncio.Lock()
    async with _checkpointer_lock:
        return await get_checkpointer()


async def get_compiled_app(use_context_compression: bool = CONTEXT_COMPRESSION):
    '''The compiled graph, built at most once per variant.

    The checkpointer is a process-wide singleton, so the graph is safe to reuse.
    The previous build recompiled it on **every user message** — five chat models
    and five agents — because the planning agent's prompt had to be regenerated
    from episodic memory. That now happens in middleware, per model call, so this
    can be cached.

    Parameters:
    ---------
    use_context_compression (boolean): which variant to build.

    Returns:
    ----------
    app (CompiledGraph): the cached compiled graph. Caching matters because the previous build recompiled five chat models and five agents on every user message.
    '''

    global _app_cache_lock
    if _app_cache_lock is None:
        _app_cache_lock = asyncio.Lock()

    key = bool(use_context_compression)
    cached = _app_cache.get(key)
    if cached is not None:
        return cached

    async with _app_cache_lock:
        cached = _app_cache.get(key)
        if cached is not None:
            return cached
        checkpointer = await _get_checkpointer()
        compiled = await create_app(checkpointer, use_context_compression=key)
        _app_cache[key] = compiled
        logger.info("Agent graph compiled (context_compression=%s)", key)
        return compiled


def clear_compiled_app_cache() -> None:
    _app_cache.clear()


@asynccontextmanager
async def app_session(app_config: AppRunConfig):
    yield await get_compiled_app(app_config.use_context_compression)


# --- Reading the approval gate from the graph ---------------------------------


def _interrupt_payloads(snapshot: Any) -> List[Dict[str, Any]]:
    '''The values passed to `interrupt()` by whatever paused this graph.

    `human_chat_node` raises a structured payload and the approval panel is built
    from it. Reading only `snapshot.next` told the UI *that* the graph had paused
    and discarded *why*, which made a paused run indistinguishable from a finished
    one. Both the snapshot's own interrupts and each pending task's are read, so the
    payload survives a checkpointer upgrade.

    Parameters:
    ---------
    snapshot (Any): a graph state snapshot.

    Returns:
    ----------
    payloads (list): the values passed to `interrupt()` by whatever paused the graph.
    '''

    payloads: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def collect(interrupts: Any) -> None:
        for item in tuple(interrupts or ()):
            value = getattr(item, "value", item)
            if not isinstance(value, dict):
                continue
            # The same interrupt is reachable from both paths; key on content.
            key = json.dumps(value, sort_keys=True, default=str)
            if key in seen:
                continue
            seen.add(key)
            payloads.append(value)

    collect(getattr(snapshot, "interrupts", ()))
    for task in tuple(getattr(snapshot, "tasks", ()) or ()):
        collect(getattr(task, "interrupts", ()))
    return payloads


async def read_pending_approval(
    thread_id: str,
    *,
    use_context_compression: bool = CONTEXT_COMPRESSION,
) -> Optional[Dict[str, Any]]:
    '''The plan-review payload if `thread_id` is paused for approval, else None.

    **The graph is the single source of truth for this.** The UI's own flag lives in
    per-browser session state, which thread switches, page reloads and restarts
    reset; when the two disagreed, the next message went in as fresh input instead
    of a resume, and LangGraph silently restarted from START — re-classifying,
    re-planning and interrupting again, which trapped the user in an approval loop
    they could not leave.

    Parameters:
    ---------
    thread_id (str): the conversation to inspect.
    use_context_compression (boolean): which compiled variant to read through.

    Returns:
    ----------
    payload (dict): the plan-review payload when that thread is paused, else None. Always re-read from the graph rather than trusted from session state — sending plain input to an interrupted thread makes LangGraph restart from `START` and re-plan.
    '''

    if not thread_id:
        return None
    try:
        compiled = await get_compiled_app(use_context_compression)
        snapshot = await compiled.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:  # noqa: BLE001 - a read failure must not block sending
        logger.warning("Could not read graph state for thread %s: %s", thread_id, exc)
        return None

    if "human_chat" not in tuple(getattr(snapshot, "next", ()) or ()):
        return None
    payloads = _interrupt_payloads(snapshot)
    review = next((item for item in payloads if item.get("type") == "plan_review"), None)
    # Paused at human_chat without a readable payload is still an approval gate;
    # report one rather than leaving the user with no way forward.
    return review or (payloads[0] if payloads else {"type": "plan_review"})


async def thread_awaits_approval(thread_id: str) -> bool:
    return (await read_pending_approval(thread_id)) is not None


async def read_transcript(thread_id: str) -> List[Dict[str, str]]:
    '''A conversation's messages, straight from its checkpoint.

    The UI normally restores a conversation from the *rendered* timeline snapshot,
    which is what preserves which tool calls succeeded. A conversation written by an
    older version of this app has no snapshot — only the checkpoint — and without
    this it would open as an empty transcript with its history sitting unread in the
    same database file.

    Read through the checkpointer rather than the compiled graph, deliberately: an
    old checkpoint carries channels the current graph does not declare, and this
    only needs `messages`.

    Parameters:
    ---------
    thread_id (str): the conversation to read.

    Returns:
    ----------
    messages (list): `{"role", "content"}` entries in order, or empty when there is nothing to show.
    '''

    try:
        checkpointer = await get_checkpointer()
        checkpoint = await checkpointer.aget({"configurable": {"thread_id": thread_id}})
    except Exception as exc:  # noqa: BLE001 - an unreadable checkpoint is not fatal
        logger.warning("Could not read the transcript for %s: %s", thread_id, exc)
        return []
    if not checkpoint:
        return []

    transcript: List[Dict[str, str]] = []
    for message in checkpoint.get("channel_values", {}).get("messages", []) or []:
        role = getattr(message, "type", None)
        content = getattr(message, "content", None)
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        text = str(content or "").strip()
        # Tool traffic is not rendered from a bare checkpoint: a ToolMessage has no
        # call to attach to once the structured timeline is gone, and an AIMessage
        # that only carried tool_calls has nothing to say.
        if role not in ("human", "ai") or not text:
            continue
        transcript.append({"role": "user" if role == "human" else "assistant", "content": text})
    return transcript


def _is_interrupt_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in ("interrupt", "interrupted", "human input required"))


# --- The event stream ---------------------------------------------------------


async def stream_langgraph_events(
    app_config: AppRunConfig,
    stream_input: Any,
    thread_id: str,
    *,
    check_for_interrupts: bool = False,
):
    if not thread_id:
        raise ValueError("No active conversation thread is selected.")

    config = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": RECURSION_LIMIT,
    }

    # Binding the scope here only reaches consumers that iterate with a plain
    # `async for`. `run_controller` pumps this generator with a Task per
    # `__anext__` so it can race the file-refresh tick, and a Task starts from its
    # own copy of the ambient context — so what is set here survives only the first
    # event. That caller pins an explicit context instead; see
    # `build_conversation_context`. Kept for direct consumers and for tests.
    conversation_token = set_current_conversation_id(thread_id)

    token_buffers: Dict[Tuple[str, str], str] = {}
    token_last_flush: Dict[Tuple[str, str], float] = {}
    streamed_interrupts: List[Dict[str, Any]] = []

    try:
        async with app_session(app_config) as compiled:
            events = compiled.astream(
                stream_input,
                config=config,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            )

            async for item in events:
                if not isinstance(item, tuple) or len(item) != 3:
                    continue
                namespace, mode, data = item

                if mode == "updates":
                    if not isinstance(data, dict):
                        continue
                    if "__interrupt__" in data:
                        # Captured in-band, so recovering the payload no longer
                        # depends on a post-stream state read.
                        for entry in tuple(data.get("__interrupt__") or ()):
                            value = getattr(entry, "value", entry)
                            if isinstance(value, dict):
                                streamed_interrupts.append(value)
                        continue
                    for node_name, update in data.items():
                        if not isinstance(update, dict) or not update.get("messages"):
                            continue
                        agent_name = resolve_agent_name(namespace, None, node_name)
                        yield ("chunk", {agent_name: {"messages": list(update["messages"])}})
                    continue

                if mode != "messages" or not isinstance(data, tuple) or len(data) != 2:
                    continue

                message, metadata = data
                # Completed messages from non-streaming nodes also arrive here, but
                # the node's `updates` entry carries the same message, so they are
                # left to that path rather than rendered twice.
                if not isinstance(message, AIMessageChunk) or not STREAM_TOKENS:
                    continue

                text = _stream_chunk_text(message)
                if not text:
                    continue
                agent_name = resolve_agent_name(namespace, metadata or {})
                key = (agent_name, str(getattr(message, "id", "") or ""))
                buffered = token_buffers.get(key, "") + text
                now = time.monotonic()
                if (
                    now - token_last_flush.get(key, 0.0) >= STREAM_FLUSH_SECONDS
                    or len(buffered) >= STREAM_FLUSH_CHARS
                ):
                    token_last_flush[key] = now
                    token_buffers[key] = ""
                    yield (
                        "token",
                        {
                            agent_name: {
                                "messages": [AIMessageChunk(content=buffered, id=key[1] or None)]
                            }
                        },
                    )
                else:
                    token_buffers[key] = buffered

            interrupted = bool(streamed_interrupts)
            approval: Optional[Dict[str, Any]] = None
            if streamed_interrupts:
                approval = next(
                    (item for item in streamed_interrupts if item.get("type") == "plan_review"),
                    streamed_interrupts[0],
                )
            elif check_for_interrupts:
                try:
                    # Fallback for an interrupt that carried no readable payload.
                    # Read the snapshot only after the iterator is done: a
                    # persistent checkpointer may not expose the checkpoint that
                    # backs the interrupt until then, and would answer with stale
                    # pre-interrupt state.
                    snapshot = await compiled.aget_state(config)
                    if "human_chat" in tuple(getattr(snapshot, "next", ()) or ()):
                        interrupted = True
                        payloads = _interrupt_payloads(snapshot)
                        approval = next(
                            (item for item in payloads if item.get("type") == "plan_review"),
                            payloads[0] if payloads else {"type": "plan_review"},
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to inspect graph state for interrupts: %s", exc)

            yield (
                "complete",
                {"interrupted": interrupted, "approval": approval, "completed_at": time.time()},
            )
    except Exception as exc:
        if check_for_interrupts and _is_interrupt_exception(exc):
            yield (
                "complete",
                {
                    "interrupted": True,
                    "approval": {"type": "plan_review"},
                    "completed_at": time.time(),
                },
            )
            return
        raise
    finally:
        reset_current_conversation_id(conversation_token)


def build_stream_input(
    user_message: str,
    *,
    conversation_id: Optional[str] = None,
    use_episodic_learning: bool = True,
    resume: bool = False,
) -> Any:
    '''Graph input for a fresh turn, or a resume of a paused approval.

    Parameters:
    ---------
    user_message (str): what the user typed.
    conversation_id (str): the conversation being run.
    use_episodic_learning (boolean): whether to offer the planner precedent from past conversations.
    resume (boolean): resume a paused approval instead of starting a fresh turn.

    Returns:
    ----------
    stream_input (any): the graph input — a `Command` when resuming, otherwise a fresh message.
    '''

    if resume:
        return Command(resume=user_message)

    # Tagged so the context layer can tell a new user turn from the HumanMessages
    # that plan review writes into the same transcript.
    messages = [mark_user_request(message) for message in convert_to_messages([user_message])]
    payload: Dict[str, Any] = {"messages": messages, "use_episodic_learning": use_episodic_learning}
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


__all__ = [
    "app_session",
    "build_stream_input",
    "clear_compiled_app_cache",
    "get_compiled_app",
    "read_pending_approval",
    "read_transcript",
    "resolve_agent_name",
    "stream_langgraph_events",
    "thread_awaits_approval",
]
