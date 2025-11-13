import os
import hashlib
from datetime import datetime
import streamlit as st
from langchain_core.messages import convert_to_messages
from langgraph.types import Command

from app.config import RECURSION_LIMIT, logger
from core.supervisor.supervisor import create_app
from backend.memory.episodic_memory.thread_manager import load_thread_ids, update_thread_title
from backend.memory.episodic_memory.conversation import (
    create_new_conversation, 
    load_conversation,
    get_welcome_message
)
from app.ui.components import display_header, display_sidebar, display_chat_messages, add_episodic_controls
from app.ui.formatters import separate_agent_outputs



def get_thread_files(thread_id: str) -> list:
    """Get uploaded files for a specific thread."""
    return st.session_state.thread_files.get(thread_id, [])


def set_thread_files(thread_id: str, files: list):
    """Set uploaded files for a specific thread."""
    st.session_state.thread_files[thread_id] = files


def add_file_to_thread(thread_id: str, file_info: dict):
    """Add a file to a specific thread."""
    if thread_id not in st.session_state.thread_files:
        st.session_state.thread_files[thread_id] = []
    st.session_state.thread_files[thread_id].append(file_info)


def remove_file_from_thread(thread_id: str, file_index: int):
    """Remove a file from a specific thread."""
    if thread_id in st.session_state.thread_files and 0 <= file_index < len(st.session_state.thread_files[thread_id]):
        st.session_state.thread_files[thread_id].pop(file_index)


def update_current_thread_files():
    """Update current session uploaded_files with current thread's files."""
    if 'current_thread_id' in st.session_state:
        thread_id = st.session_state.current_thread_id
        thread_files = get_thread_files(thread_id)
        st.session_state.uploaded_files = thread_files
        
        # Update backward compatibility vars
        if thread_files:
            latest_file = thread_files[-1]
            st.session_state.uploaded_file_path = latest_file['path']
            st.session_state.uploaded_file_hash = latest_file['hash']
            st.session_state.uploaded_file_name = latest_file['name']
        else:
            st.session_state.uploaded_file_path = None
            st.session_state.uploaded_file_hash = None
            st.session_state.uploaded_file_name = None


def get_file_hash(uploaded_file) -> str:
    """Generate hash of uploaded file content for comparison."""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        content = uploaded_file.getbuffer()
        # Generate MD5 hash of file content
        file_hash = hashlib.md5(content).hexdigest()
        # Reset file pointer again for potential re-use
        uploaded_file.seek(0)
        return file_hash
    except Exception as e:
        logger.warning(f"Could not generate file hash: {e}")
        return None


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to data directory and return file path."""
    try:
        # Validate file size (limit to 50MB)
        if uploaded_file.size > 50 * 1024 * 1024:
            raise ValueError("File size exceeds 50MB limit")
        
        # Ensure data directory exists
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Create unique filename to avoid overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(uploaded_file.name)
        
        # Sanitize filename to avoid issues
        safe_filename = "".join(c for c in filename if c.isalnum() or c in (' ', '.', '_', '-'))
        unique_filename = f"{safe_filename}_{timestamp}{ext}"
        file_path = os.path.join(data_dir, unique_filename)
        
        # Save file with unique name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File uploaded: {file_path} (size: {uploaded_file.size} bytes)")
        return file_path
    except Exception as e:
        logger.error(f"Error saving uploaded file: {e}")
        raise e


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    # Initialize episodic learning system
    if 'episodic_orchestrator' not in st.session_state:
        try:
            from backend.memory.episodic_memory.episodic_learning import get_orchestrator
            st.session_state.episodic_orchestrator = get_orchestrator()
            logger.info("Episodic learning system initialized")
        except Exception as e:
            logger.warning(f"Could not initialize episodic learning: {e}")
            st.session_state.episodic_orchestrator = None
    
    # Initialize thread-specific file upload tracking
    if 'thread_files' not in st.session_state:
        st.session_state.thread_files = {}  # Dictionary: {thread_id: [{'path': str, 'hash': str, 'name': str}]}
    
    # Keep backward compatibility - will be set when thread is active
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = []
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'uploaded_file_hash' not in st.session_state:
        st.session_state.uploaded_file_hash = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    # Initialize episodic learning settings
    if 'use_episodic_learning' not in st.session_state:
        st.session_state.use_episodic_learning = True
    
    # Initialize content deduplication tracking
    if 'processed_content_hashes' not in st.session_state:
        st.session_state.processed_content_hashes = set()
    
    # Initialize expander states for UI persistence
    if 'expander_states' not in st.session_state:
        st.session_state.expander_states = {
            'progress_expander': True,  # Default expanded for progress
            'show_progress_content': False  # Whether progress content exists
        }
    
    # Note: We don't initialize the app here anymore since we need user context for enhancement
    
    # Load existing thread IDs
    if 'thread_ids' not in st.session_state:
        st.session_state.thread_ids = load_thread_ids()

    # Initialize current conversation if none exists
    if 'current_thread_id' not in st.session_state:
        if st.session_state.thread_ids:
            # Load the most recent conversation
            recent_thread = st.session_state.thread_ids[-1]
            load_conversation(recent_thread["thread_id"], None)  # We'll create app when needed
        else:
            # Create a new conversation
            new_conv = create_new_conversation()
            st.session_state.current_thread_id = new_conv["thread_id"]
            st.session_state.messages = new_conv["messages"]
            st.session_state.processed_message_ids = new_conv["processed_message_ids"]
            st.session_state.processed_tools_ids = new_conv["processed_tools_ids"]
            st.session_state.thread_ids = load_thread_ids()  # Reload after adding new thread
    
    # Update current thread files (this ensures files are loaded for the current thread)
    update_current_thread_files()
    
    # Initialize human-in-the-loop state
    if 'waiting_for_approval' not in st.session_state:
        st.session_state.waiting_for_approval = False
    if 'current_plan' not in st.session_state:
        st.session_state.current_plan = None
    if 'approval_interrupted' not in st.session_state:
        st.session_state.approval_interrupted = False


def get_or_create_app_for_request(user_request: str):
    """Get or create an app enhanced for the specific user request."""
    try:
        # Create enhanced app for this request
        use_learning = st.session_state.get('use_episodic_learning', True)
        
        app = create_app(
            user_request=user_request if use_learning else None,
            use_episodic_learning=use_learning
        )
        
        # Show episodic learning info in sidebar
        if use_learning and st.session_state.episodic_orchestrator:
            display_episodic_info(user_request)
        
        return app
        
    except Exception as e:
        logger.error(f"Error creating enhanced app: {e}")
        # Fallback to standard app
        return create_app(use_episodic_learning=False)


def display_episodic_info(user_request: str):
    """Display episodic learning information in the sidebar."""
    try:
        orchestrator = st.session_state.episodic_orchestrator
        if not orchestrator:
            return
            
        # Get context for the current request
        context = orchestrator.get_episodic_context(user_request, max_examples=3)
        
        with st.sidebar:
            st.header("🧠 Episodic Learning")
            
            # Show relevance score
            relevance = context.get('context_relevance_score', 0.0)
            st.metric("Context Relevance", f"{relevance:.2f}")
            
            # Show number of examples
            examples_count = len(context.get('task_examples', []))
            st.metric("Examples Found", examples_count)
            
            if relevance > 0.65 and examples_count > 0:
                st.success("🎯 Enhanced with learned patterns!")
                
                # Show examples used
                with st.expander("Examples Used"):
                    for i, example in enumerate(context['task_examples'][:2], 1):
                        st.write(f"**{i}.** {example[:100]}...")
                
                # Show learned notes
                if context.get('notes'):
                    with st.expander("Learned Insights"):
                        for note in context['notes'][:2]:
                            st.write(f"• {note[:150]}...")
                            
            elif examples_count > 0:
                st.info("📚 Some examples found, but low relevance")
            else:
                st.info("📖 No relevant examples - using standard approach")
                
    except Exception as e:
        logger.warning(f"Error displaying episodic info: {e}")


def create_persistent_dual_display(is_processing=True):
    """Create persistent dual display containers for progress and final output."""
    # Create main container for final output
    final_container = st.empty()
    
    # Create expander for progress with persistent state
    progress_has_content = st.session_state.expander_states.get('show_progress_content', False)
    expander_expanded = st.session_state.expander_states.get('progress_expander', True)
    
    # Show expander if there's content OR we're actively processing
    if progress_has_content or is_processing:
        # Dynamic title based on processing state
        title = "🔄 Processing Progress" if is_processing else "🔄 Agent Activity Log"
        with st.expander(title, expanded=expander_expanded):
            progress_container = st.empty()
    else:
        progress_container = None
    
    return final_container, progress_container


def _is_interrupt_exception(exc: Exception) -> bool:
    """Check if an exception indicates a human-in-the-loop interrupt."""
    message = str(exc).lower()
    return any(keyword in message for keyword in (
        "nodeinterrupt",
        "interrupt",
        "interrupted",
        "human input required"
    ))


def process_stream_updates(
    app,
    stream_input,
    config: dict,
    final_container,
    progress_container,
    *,
    check_for_interrupts: bool = False
):
    """Stream LangGraph updates, update UI containers, and manage interrupt state."""
    accumulated_progress = ""
    accumulated_final = ""
    local_processed_message_ids = st.session_state.processed_message_ids.copy()
    if 'expander_states' not in st.session_state:
        st.session_state.expander_states = {
            'progress_expander': True,
            'show_progress_content': False
        }
    
    processed_content_hashes = st.session_state.get('processed_content_hashes')
    if processed_content_hashes is None:
        processed_content_hashes = set()
        st.session_state.processed_content_hashes = processed_content_hashes
    
    try:
        stream_iterator = app.stream(
            stream_input,
            config=config,
            stream_mode="updates"
        )
        
        for chunk in stream_iterator:
            if isinstance(chunk, tuple):
                continue
            if not isinstance(chunk, dict):
                logger.warning(f"Unexpected chunk format: {type(chunk)}")
                continue
            
            progress_message, final_message = separate_agent_outputs(
                chunk,
                local_processed_message_ids,
                processed_content_hashes
            )
            
            if progress_message:
                accumulated_progress += progress_message
                if progress_container:
                    progress_container.markdown(accumulated_progress)
                st.session_state.expander_states['show_progress_content'] = True
            
            if final_message:
                accumulated_final += final_message
                final_container.markdown(accumulated_final)
        
        if check_for_interrupts:
            try:
                current_state = app.get_state(config)
                if hasattr(current_state, "next") and current_state.next:
                    if 'human_chat' in current_state.next:
                        st.session_state.waiting_for_approval = True
                        st.session_state.current_plan = accumulated_final
                        st.session_state.approval_interrupted = True
                        st.session_state.processed_message_ids = local_processed_message_ids
                        final_container.markdown(accumulated_final)
                        return accumulated_progress, accumulated_final, True
            except Exception as state_error:
                logger.warning(f"Could not check execution state: {state_error}")
        
        st.session_state.processed_message_ids = local_processed_message_ids
        return accumulated_progress, accumulated_final, False
    
    except Exception as exc:
        if check_for_interrupts and _is_interrupt_exception(exc):
            st.session_state.waiting_for_approval = True
            st.session_state.current_plan = accumulated_final
            st.session_state.approval_interrupted = True
            st.session_state.processed_message_ids = local_processed_message_ids
            final_container.markdown(accumulated_final)
            return accumulated_progress, accumulated_final, True
        raise


def handle_human_approval():
    """Handle plan approval from user."""
    if st.session_state.waiting_for_approval and st.session_state.current_plan:
        app = get_or_create_app_for_request("approval")
        
        config = {
            "configurable": {
                "thread_id": st.session_state.current_thread_id, 
                
            }, 
            "recursion_limit": RECURSION_LIMIT
        }
        
        # Resume with approval signal
        try:
            with st.spinner("Proceeding with plan execution..."):
                # Create persistent dual display - actively processing
                final_container, progress_container = create_persistent_dual_display(is_processing=True)
                
                _, accumulated_final, _ = process_stream_updates(
                    app,
                    Command(resume="approved"),
                    config,
                    final_container,
                    progress_container
                )
                
                # Add the supervisor execution to chat history (only final output - progress is temporary UI)
                if accumulated_final.strip():
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": accumulated_final
                    })
                
                # Reset approval state
                st.session_state.waiting_for_approval = False
                st.session_state.current_plan = None
                st.session_state.approval_interrupted = False
                st.rerun()
                
        except Exception as e:
            st.error(f"Error proceeding with execution: {e}")
            logger.error(f"Error in plan execution: {e}")


def handle_user_input(prompt: str, app=None):
    """Handle user input and generate response."""
    # If we're waiting for approval, handle the continuation
    if st.session_state.waiting_for_approval:
        # Continue with the workflow - reuse the existing app to preserve episodic learning context
        if app is None:
            # Get the original user request from session state or first user message
            original_request = None
            for msg in st.session_state.messages:
                if msg.get("role") == "user" and msg.get("content"):
                    original_request = msg["content"]
                    break
            app = get_or_create_app_for_request(original_request or prompt)
        
        config = {
            "configurable": {
                "thread_id": st.session_state.current_thread_id, 
            },
            "recursion_limit": RECURSION_LIMIT
        }
        
        # Add user input to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Resume with user input - let the graph decide routing
        try:
            with st.chat_message("assistant"):
                with st.spinner("Processing..."):
                    # Create persistent dual display - actively processing
                    final_container, progress_container = create_persistent_dual_display(is_processing=True)
                    
                    _, accumulated_final, _ = process_stream_updates(
                        app,
                        Command(resume=prompt),
                        config,
                        final_container,
                        progress_container
                    )
                    
                    # Add refined plan to chat history (only final output - progress is temporary UI)
                    if accumulated_final.strip():
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": accumulated_final
                        })
                    
                    # Reset approval state since we continued
                    st.session_state.waiting_for_approval = False
                    st.session_state.current_plan = None
                    st.session_state.approval_interrupted = False
                    
                    st.rerun()
                    
        except Exception as e:
            st.error(f"Error processing input: {e}")
            logger.error(f"Error in workflow continuation: {e}")
        
        return
    
    # Create enhanced app for this specific request
    if app is None:
        app = get_or_create_app_for_request(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Update conversation title if this is the first user message
    if len(st.session_state.messages) == 2:  # First user message (after welcome message)
        new_title = prompt[:30] + "..." if len(prompt) > 30 else prompt
        update_thread_title(st.session_state.current_thread_id, new_title)
        # Reload thread list to reflect title change
        st.session_state.thread_ids = load_thread_ids()
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        logger.info("Generating response...")
        
        with st.spinner("Processing..."):        
            # Create persistent dual display containers - actively processing
            final_container, progress_container = create_persistent_dual_display(is_processing=True)
            
            langchain_messages = convert_to_messages([prompt])
            config = {
                "configurable": {
                    "thread_id": st.session_state.current_thread_id,      
                },
                "recursion_limit": RECURSION_LIMIT
            }
            

            try:
                _, accumulated_reply, interrupted = process_stream_updates(
                    app,
                    {"messages": langchain_messages},
                    config,
                    final_container,
                    progress_container,
                    check_for_interrupts=True
                )
                
                # Always persist the assistant response for traceability (progress stays transient)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": accumulated_reply
                })
                
                if interrupted:
                    return
                
                st.session_state.approval_interrupted = False
                
            except Exception as e:
                st.error(f"Error processing request: {e}")
                logger.error(f"Error in stream_response: {e}")



def main():
    """Main Streamlit application entry point."""
    # Initialize session state
    initialize_session_state()
    
    # Display header
    display_header()
    
    # Add episodic learning controls
    add_episodic_controls()
    
    # Display standard sidebar (create a basic app for compatibility)
    try:
        basic_app = create_app(use_episodic_learning=False)
        display_sidebar(basic_app)
    except Exception as e:
        logger.warning(f"Could not create basic app for sidebar: {e}")
    
    logger.info("Enhanced app started with episodic learning and human-in-the-loop")
    
    # Display all previous messages
    display_chat_messages(st.session_state.messages)
    
    # File upload section - compact expandable design with multi-file support
    file_count = len(st.session_state.uploaded_files)
    if file_count == 0:
        expander_title = "📎 Upload Files"
    elif file_count == 1:
        expander_title = f"📎 Upload Files • {os.path.basename(st.session_state.uploaded_files[0]['name'])}"
    else:
        expander_title = f"📎 Upload Files • {file_count} files ready"
    
    with st.expander(expander_title, expanded=file_count == 0):
        col1, col2 = st.columns([4, 1])
        with col1:
            uploaded_files = st.file_uploader(
                "Choose data files", 
                key=f"file_uploader_{st.session_state.current_thread_id}",
                help="Supported formats: CSV, TXT, JSON, Excel, etc.",
                label_visibility="collapsed",
                accept_multiple_files=True
            )
        with col2:
            if st.session_state.uploaded_files:
                if st.button("🗑️ Clear All", key=f"clear_files_{st.session_state.current_thread_id}", help="Remove all uploaded files"):
                    # Clear thread-specific files
                    set_thread_files(st.session_state.current_thread_id, [])
                    # Update current session state
                    update_current_thread_files()
        
        # Display currently uploaded files
        if st.session_state.uploaded_files:
            st.write("**Ready to process:**")
            for i, file_info in enumerate(st.session_state.uploaded_files):
                col_file, col_remove = st.columns([5, 1])
                with col_file:
                    st.write(f"📄 {file_info['name']}")
                with col_remove:
                    if st.button("🗑️", key=f"remove_file_{i}_{st.session_state.current_thread_id}", help=f"Remove {file_info['name']}"):
                        # Remove from thread-specific storage
                        remove_file_from_thread(st.session_state.current_thread_id, i)
                        # Update current session state
                        update_current_thread_files()
        
        # Handle file upload separately (no AI processing) - improved duplicate prevention for multiple files
        if uploaded_files:
            current_thread_id = st.session_state.current_thread_id
            for uploaded_file in uploaded_files:
                current_hash = get_file_hash(uploaded_file)
                current_name = uploaded_file.name
                
                # Check if this file (by hash) is already uploaded to current thread
                thread_files = get_thread_files(current_thread_id)
                existing_hashes = [f['hash'] for f in thread_files]
                if current_hash not in existing_hashes:
                    try:
                        file_path = save_uploaded_file(uploaded_file)
                        file_info = {
                            'path': file_path,
                            'hash': current_hash,
                            'name': current_name
                        }
                        
                        # Add to thread-specific storage
                        add_file_to_thread(current_thread_id, file_info)
                        
                        # Update current session state
                        update_current_thread_files()
                        
                        st.success(f"✅ File ready: {uploaded_file.name}")
                    except Exception as e:
                        st.error(f"Error uploading {uploaded_file.name}: {e}")
            
            # Files are automatically displayed in the expander, no rerun needed
    
    # Clean chat input
    prompt = st.chat_input("Your message")
    
    # Only handle text input for AI processing
    if prompt:
        final_prompt = prompt
        
        # Append file paths if there are uploaded files
        if st.session_state.uploaded_files:
            if len(st.session_state.uploaded_files) == 1:
                final_prompt += f"\n\nUploaded file: {st.session_state.uploaded_files[0]['path']}"
            else:
                final_prompt += f"\n\nUploaded files:"
                for file_info in st.session_state.uploaded_files:
                    final_prompt += f"\n- {file_info['path']}"
            
            # Note: Files are kept uploaded and available for subsequent requests
            # Users can manually clear them using the "Clear All" button if needed
        
        handle_user_input(final_prompt)


if __name__ == "__main__":
    main()
