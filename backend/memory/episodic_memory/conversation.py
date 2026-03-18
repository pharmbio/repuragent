import json
from typing import List, Dict, Set, Any, Optional
from datetime import datetime
import aiosqlite
from app.config import logger
from app.config import SQLITE_DB_PATH
from app.ui.formatters import reconstruct_assistant_response
from backend.memory.episodic_memory.thread_manager import (
    UI_TIMELINE_TABLE,
    UI_TIMELINE_TABLE_SQL,
    add_thread_id,
    generate_new_thread_id,
)


async def get_conversation_history_from_sqlite(thread_id: str, app) -> List[Dict]:
    """Retrieve conversation history from SQLite checkpointer."""
    try:
        if app is None:
            logger.warning("App is None when retrieving conversation history")
            return []
            
        config = {"configurable": {"thread_id": thread_id}}
        state = await app.aget_state(config)
        
        if state and state.values and "messages" in state.values:
            messages = state.values["messages"]
            
            display_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    if hasattr(msg, 'type'):
                        if msg.type == "human":
                            role = "user"
                        elif msg.type == "ai":
                            role = "assistant"
                        else:
                            continue
                    else:
                        continue
                    
                    content = msg.content
                    if isinstance(content, list):
                        text_content = ""
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_content += part.get("text", "")
                        content = text_content
                    
                    if content:
                        display_messages.append({
                            "role": role,
                            "content": content
                        })
            
            return display_messages
        
        return []
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {e}")
        return []


def reconstruct_formatted_message_from_sqlite(messages) -> List[Dict]:
    """Reconstruct formatted assistant messages from raw SQLite messages."""
    try:
        formatted_messages = []
        current_sequence = []
        
        for msg in messages:
            if hasattr(msg, 'type'):
                if msg.type == "human":
                    # If we have accumulated AI messages, process them
                    if current_sequence:
                        formatted_content = reconstruct_assistant_response(current_sequence)
                        if formatted_content:
                            formatted_messages.append({
                                "role": "assistant", 
                                "content": formatted_content
                            })
                        current_sequence = []
                    
                    # Add user message
                    if hasattr(msg, 'content') and msg.content:
                        formatted_messages.append({
                            "role": "user",
                            "content": msg.content
                        })
                
                elif msg.type in {"ai", "tool"}:
                    # Accumulate AI messages for processing
                    current_sequence.append(msg)
        
        # Process any remaining AI messages
        if current_sequence:
            formatted_content = reconstruct_assistant_response(current_sequence)
            if formatted_content:
                formatted_messages.append({
                    "role": "assistant",
                    "content": formatted_content
                })
        
        return formatted_messages
        
    except Exception as e:
        logger.error(f"Error reconstructing formatted messages: {e}")
        return []


async def get_processed_message_ids_from_sqlite(thread_id: str, app) -> Set[str]:
    """Retrieve all message IDs from SQLite to mark as processed."""
    try:
        if app is None:
            logger.warning("App is None when retrieving processed message IDs")
            return set()
            
        config = {"configurable": {"thread_id": thread_id}}
        state = await app.aget_state(config)
        
        processed_ids = set()
        if state and state.values and "messages" in state.values:
            messages = state.values["messages"]
            for msg in messages:
                msg_id = getattr(msg, "id", None)
                if msg_id:
                    processed_ids.add(msg_id)
        
        return processed_ids
    except Exception as e:
        logger.error(f"Error retrieving processed message IDs: {e}")
        return set()


def create_new_conversation() -> Dict[str, Any]:
    """Create a new conversation with a unique thread ID."""
    thread_id = generate_new_thread_id()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to persistent storage
    add_thread_id(thread_id, f"Conversation {timestamp}")
    
    # Initialize conversation data with no default chatbot messages
    conversation_data = {
        "thread_id": thread_id,
        "title": f"Conversation {timestamp}",
        "created_at": timestamp,
        "messages": [],
        "processed_message_ids": set(),
        "processed_tools_ids": set()
    }
    
    return conversation_data


async def _ensure_ui_timeline_table(connection: aiosqlite.Connection) -> None:
    await connection.execute(UI_TIMELINE_TABLE_SQL)
    await connection.commit()


async def load_ui_timeline(thread_id: str) -> Optional[Dict[str, Any]]:
    """Load the persisted UI timeline snapshot for a thread."""
    if not thread_id:
        return None
    try:
        async with aiosqlite.connect(str(SQLITE_DB_PATH), timeout=30) as connection:
            await connection.execute("PRAGMA busy_timeout = 5000")
            await _ensure_ui_timeline_table(connection)
            async with connection.execute(
                f"SELECT snapshot_json FROM {UI_TIMELINE_TABLE} WHERE thread_id = ?",
                (thread_id,),
            ) as cursor:
                row = await cursor.fetchone()
    except Exception as exc:
        logger.warning("Unable to load UI timeline for %s: %s", thread_id, exc)
        return None

    if not row or not row[0]:
        return None
    try:
        snapshot = json.loads(row[0])
    except json.JSONDecodeError as exc:
        logger.warning("Invalid UI timeline snapshot for %s: %s", thread_id, exc)
        return None
    return snapshot if isinstance(snapshot, dict) else None


async def save_ui_timeline(thread_id: str, snapshot: Dict[str, Any]) -> None:
    """Persist the UI timeline snapshot for a thread."""
    if not thread_id or not isinstance(snapshot, dict):
        return

    payload = json.dumps(snapshot, ensure_ascii=True)
    try:
        async with aiosqlite.connect(str(SQLITE_DB_PATH), timeout=30) as connection:
            await connection.execute("PRAGMA busy_timeout = 5000")
            await _ensure_ui_timeline_table(connection)
            await connection.execute(
                f"""
                INSERT INTO {UI_TIMELINE_TABLE} (thread_id, snapshot_json, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(thread_id) DO UPDATE
                SET snapshot_json = excluded.snapshot_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (thread_id, payload),
            )
            await connection.commit()
    except Exception as exc:
        logger.warning("Unable to save UI timeline for %s: %s", thread_id, exc)


async def load_conversation(thread_id: str, app) -> Dict[str, Any]:
    """Load a conversation from persistent storage with formatting preserved."""
    try:
        if app is None:
            logger.warning("App is None when loading conversation")
            return {
                "thread_id": thread_id,
                "messages": [],
                "raw_messages": [],
                "processed_message_ids": set(),
                "has_progress_content": False,
                "timeline_snapshot": None,
            }
            
        config = {"configurable": {"thread_id": thread_id}}
        state = await app.aget_state(config)
        
        messages = []
        raw_messages = []
        if state and state.values and "messages" in state.values:
            raw_messages = state.values["messages"]
            messages = reconstruct_formatted_message_from_sqlite(raw_messages)
        
        if not messages:
            messages = []
        
        processed_message_ids = await get_processed_message_ids_from_sqlite(thread_id, app)
        timeline_snapshot = await load_ui_timeline(thread_id)
        
        has_progress_content = any(
            msg.get("role") == "assistant" and any(
                agent.upper() in msg.get("content", "")
                for agent in ["SUPERVISOR", "RESEARCH_AGENT", "DATA_AGENT", "PREDICTION_AGENT"]
            )
            for msg in messages
        )
        
        return {
            "thread_id": thread_id,
            "messages": messages,
            "raw_messages": raw_messages,
            "processed_message_ids": processed_message_ids,
            "has_progress_content": has_progress_content,
            "timeline_snapshot": timeline_snapshot,
        }
        
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        messages = await get_conversation_history_from_sqlite(thread_id, app) if app else []
        if not messages:
            messages = []
        return {
            "thread_id": thread_id,
            "messages": messages,
            "raw_messages": [],
            "processed_message_ids": set(),
            "has_progress_content": False,
            "timeline_snapshot": await load_ui_timeline(thread_id),
        }


def get_welcome_message() -> Dict[str, str]:
    """Get the standard welcome message."""
    return {
        "role": "assistant",
        "content": (
            "Hello! I'm your **AI Agent for Drug Repurposing**. My team includes:\n\n"
            "- 🧠 **Prediction Agent**: Loads data from CSV files and generates predictions using pre-trained models. "
            "This agent does not analyze or interpret predictions.\n\n"
            "- 🔬 **Research Agent**: Retrieves relevant bioinformatics and cheminformatics data from PubMeD and biological knowledge graphs.\n\n"
            "- 🧰 **Data Agent**: Performs data manipulation, preprocessing, and analysis — but does not perform predictions.\n\n"
            "**How can I assist you today?**"
        )
    }
