from typing import List, Dict, Set, Any, Optional
from datetime import datetime
from app.config import logger
from app.ui.formatters import reconstruct_assistant_response
from backend.memory.episodic_memory.thread_manager import add_thread_id, generate_new_thread_id


def get_conversation_history_from_sqlite(thread_id: str, app) -> List[Dict]:
    """Retrieve conversation history from SQLite checkpointer."""
    try:
        # Check if app is None
        if app is None:
            logger.warning("App is None when retrieving conversation history")
            return []
            
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get the state from the checkpointer
        state = app.get_state(config)
        
        if state and state.values and "messages" in state.values:
            messages = state.values["messages"]
            
            # Convert LangChain messages to display format
            display_messages = []
            for msg in messages:
                if hasattr(msg, 'content') and msg.content:
                    # Determine role
                    if hasattr(msg, 'type'):
                        if msg.type == "human":
                            role = "user"
                        elif msg.type == "ai":
                            role = "assistant"
                        else:
                            continue  # Skip system messages, tool messages, etc.
                    else:
                        continue
                    
                    # Extract content
                    content = msg.content
                    if isinstance(content, list):
                        # Handle multi-part content
                        text_content = ""
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "text":
                                text_content += part.get("text", "")
                        content = text_content
                    
                    if content:  # Only add non-empty messages
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


def get_processed_message_ids_from_sqlite(thread_id: str, app) -> Set[str]:
    """Retrieve all message IDs from SQLite to mark as processed."""
    try:
        # Check if app is None
        if app is None:
            logger.warning("App is None when retrieving processed message IDs")
            return set()
            
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        
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
    
    # Initialize conversation data
    conversation_data = {
        "thread_id": thread_id,
        "title": f"Conversation {timestamp}",
        "created_at": timestamp,
        "messages": [
            {
                "role": "assistant",
                "content": (
                    "Hello! I'm your **AI Agent for Drug Repurposing**. My team includes:\n\n"
                    "- 🧠 **Prediction Agent**: Loads data from CSV files and generates predictions using pre-trained models. "
                    "This agent does not analyze or interpret predictions.\n\n"
                    "- 🔬 **Research Agent**: Retrieves relevant bioinformatics and cheminformatics data from Model Context Protocol (MCP) Servers.\n\n"
                    "- 🧰 **Data Agent**: Performs data manipulation, preprocessing, and analysis — but does not perform predictions.\n\n"
                    "**How can I assist you today?**"
                )
            }
        ],
        "processed_message_ids": set(),
        "processed_tools_ids": set()
    }
    
    return conversation_data


def load_conversation(thread_id: str, app):
    """Load a conversation from persistent storage with formatting preserved."""
    import streamlit as st
    
    try:
        # Check if app is None
        if app is None:
            logger.warning("App is None when loading conversation, creating new conversation")
            # Set session state with welcome message
            st.session_state.current_thread_id = thread_id
            st.session_state.messages = [get_welcome_message()]
            st.session_state.processed_message_ids = set()
            st.session_state.processed_tools_ids = set()
            return
            
        config = {"configurable": {"thread_id": thread_id}}
        state = app.get_state(config)
        
        messages = []
        if state and state.values and "messages" in state.values:
            raw_messages = state.values["messages"]
            # Reconstruct formatted messages from raw SQLite data
            messages = reconstruct_formatted_message_from_sqlite(raw_messages)
        
        # If no messages found, add welcome message
        if not messages:
            messages = [get_welcome_message()]
        
        # Get all historical message IDs from SQLite to mark as processed
        processed_message_ids = get_processed_message_ids_from_sqlite(thread_id, app)
        
        # Set session state
        st.session_state.current_thread_id = thread_id
        st.session_state.messages = messages
        st.session_state.processed_message_ids = processed_message_ids
        st.session_state.processed_tools_ids = set()
        
        # Reset content deduplication for new conversation
        st.session_state.processed_content_hashes = set()
        
        # Check if loaded messages contain progress content to determine expander state
        has_progress_content = False
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check if content contains agent headers that indicate progress content
                if any(agent.upper() in content for agent in ["SUPERVISOR", "RESEARCH_AGENT", "DATA_AGENT", "PREDICTION_AGENT"]):
                    has_progress_content = True
                    break
        
        # Update expander states based on loaded content
        if 'expander_states' not in st.session_state:
            st.session_state.expander_states = {}
        st.session_state.expander_states['show_progress_content'] = has_progress_content
        st.session_state.expander_states['progress_expander'] = True  # Default to expanded
        
        # Update thread-specific files for the loaded conversation
        if hasattr(st.session_state, 'thread_files'):
            from app.streamlit_app import update_current_thread_files
            update_current_thread_files()
        
    except Exception as e:
        logger.error(f"Error loading conversation: {e}")
        # Fallback to basic loading only if app is available
        if app is not None:
            messages = get_conversation_history_from_sqlite(thread_id, app)
        else:
            messages = []
            
        if not messages:
            messages = [get_welcome_message()]
        st.session_state.current_thread_id = thread_id
        st.session_state.messages = messages
        st.session_state.processed_message_ids = set()
        st.session_state.processed_tools_ids = set()
        
        # Reset content deduplication for conversation loading fallback
        st.session_state.processed_content_hashes = set()
        
        # Check if loaded messages contain progress content to determine expander state (fallback)
        has_progress_content = False
        for msg in messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", "")
                # Check if content contains agent headers that indicate progress content
                if any(agent.upper() in content for agent in ["SUPERVISOR", "RESEARCH_AGENT", "DATA_AGENT", "PREDICTION_AGENT"]):
                    has_progress_content = True
                    break
        
        # Update expander states based on loaded content (fallback)
        if 'expander_states' not in st.session_state:
            st.session_state.expander_states = {}
        st.session_state.expander_states['show_progress_content'] = has_progress_content
        st.session_state.expander_states['progress_expander'] = True  # Default to expanded
        
        # Update thread-specific files for the loaded conversation (fallback case)
        if hasattr(st.session_state, 'thread_files'):
            from app.streamlit_app import update_current_thread_files
            update_current_thread_files()


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
