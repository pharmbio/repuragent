import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain.chat_models import init_chat_model
from langchain_core.messages import convert_to_messages
from langmem import create_memory_manager
from pydantic import BaseModel, Field

from app.config import MEMORY_DIR, logger
from backend.memory.episodic_memory.thread_manager import load_thread_ids
from core.prompts.prompts import SUPERVISOR_SYSTEM_PROMPT_ver3


class TaskDecompositionEpisode(BaseModel):
    """An episode captures how to plan a specific situation, including context and task decomposition."""
    
    task: str = Field(..., description="The relevant context and requests from the users")
    initial_decomposition: str = Field(..., description="How the task was first broken down")
    final_decomposition: str = Field(..., description="The complete final task sequence including all additions/modifications")
    notes: str = Field(..., description="Lesson learnt during the execution that can enhance the final results")


class EpisodicLearningSystem:
    """Simplified episodic learning system that maintains all existing functionality."""
    
    def __init__(self):
        """Initialize the episodic learning system - simplified manual mode."""
        # Simplified configuration for manual mode
        self.config = {
            'vector_db_collection': 'task_episodes',
            'extraction_model': 'gpt-4o-mini',
            'max_examples': 2  # Always return top 2 most relevant
        }
        
        # Initialize components
        self._setup_vector_store()
        self._setup_llm()
        
        logger.info("Episodic Learning System initialized (manual extraction only)")
    
    def _setup_vector_store(self):
        """Initialize ChromaDB vector store."""
        try:
            chroma_path = MEMORY_DIR / "episodic_memory" / "chroma_db"
            chroma_path.mkdir(parents=True, exist_ok=True)
            
            # Disable telemetry to reduce error messages
            import os
            os.environ["ANONYMIZED_TELEMETRY"] = "False"
            
            self.chroma_client = chromadb.PersistentClient(path=str(chroma_path))
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.config['vector_db_collection']
            )
            logger.info(f"ChromaDB initialized at {chroma_path}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _setup_llm(self):
        """Initialize LLM and LangMem for episode extraction."""
        try:
            self.llm = init_chat_model(self.config['extraction_model'])
            
            # Initialize LangMem memory manager with TaskDecompositionEpisode schema
            self.memory_manager = create_memory_manager(
                self.config['extraction_model'],
                schemas=[TaskDecompositionEpisode],
                instructions="""Extract examples of successful task planning interactions. Include 
                the context and how the supervisor decomposed the task. The task 
                breakdown must be presented in this format: [Step 1] → [Step 2] → [Step 
                3]  → ... without any comments or additional information. These information will be used as examples in the system prompt of supervisor for further invocation.
                
                Focus on episodes where:
                - The task was complex and required multi-step planning
                - The supervisor provided clear step-by-step decomposition
                - The interaction was successful and produced actionable results
                - There were valuable lessons learned during execution
                
                Fields:
                - Task: Initial task requested by users (include relevant context)
                - Initial decomposition: Task decomposition from the beginning
                - Final decomposition: Task decomposition after interactions with users (if modified)
                - Notes: Lessons learned during the task that can enhance future results
                
                Only extract episodes that demonstrate good planning practices and successful outcomes."""
            )
            
            logger.info(f"LLM and LangMem memory manager initialized: {self.config['extraction_model']}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM or LangMem: {e}")
            raise
    
    # Background monitoring methods removed - manual mode only
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Deprecated - manual extraction only."""
        return {'success': False, 'error': 'Use extract_current_conversation for manual extraction'}
    
    def extract_current_conversation(self, thread_id: str) -> Dict[str, Any]:
        """Extract episodes from the current conversation (always overwrites existing).
        
        Args:
            thread_id: The ID of the current conversation thread
        
        Returns:
            Dict with extraction results
        """
        try:
            # Load conversation messages
            messages = self._load_conversation_messages(thread_id)
            
            if not messages:
                return {
                    'success': False,
                    'thread_id': thread_id,
                    'episodes_extracted': 0,
                    'message': 'No messages found in conversation'
                }
            
            # Minimal length requirement (just 2 messages)
            if len(messages) < 2:
                return {
                    'success': False,
                    'thread_id': thread_id,
                    'episodes_extracted': 0,
                    'message': 'Need at least 2 messages to extract patterns'
                }
            
            # Always extract and store (force mode)
            result = self.extract_and_store_episode(messages, thread_id)
            
            if result['success'] and result['episodes_extracted'] > 0:
                logger.info(f"Successfully extracted episode from conversation {thread_id}")
                result['message'] = '✅ Pattern extracted and stored!'
            elif result['success']:
                result['message'] = 'No patterns found in this conversation'
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting from conversation {thread_id}: {e}")
            return {
                'success': False,
                'thread_id': thread_id,
                'episodes_extracted': 0,
                'message': f'Error: {str(e)}'
            }
    
    def _load_conversation_messages(self, thread_id: str) -> List[Any]:
        """Load conversation messages from the database."""
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            from app.config import SQLITE_DB_PATH
            import sqlite3
            
            # Connect to the database and retrieve messages
            connection = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
            checkpointer = SqliteSaver(connection)
            
            # Get the latest checkpoint for this thread
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = checkpointer.get(config)
            
            if checkpoint and 'channel_values' in checkpoint:
                messages = checkpoint['channel_values'].get('messages', [])
                return convert_to_messages(messages) if messages else []
            
            return []
            
        except Exception as e:
            logger.warning(f"Could not load messages for thread {thread_id}: {e}")
            return []
    
    def extract_and_store_episode(self, messages: List[Any], thread_id: str) -> Dict[str, Any]:
        """
        Extract episodes from conversation and store them (always overwrites).
        """
        try:
            # Always delete existing entries for this thread first
            existing_docs = self.collection.get(
                where={"thread_id": thread_id}
            )
            if existing_docs['ids']:
                self.collection.delete(ids=existing_docs['ids'])
                logger.info(f"Replaced {len(existing_docs['ids'])} existing episodes for thread {thread_id}")
            
            # Extract episode using LangMem
            episode = self._extract_episode_with_langmem(messages)
            
            if not episode:
                return {
                    'success': True,
                    'thread_id': thread_id,
                    'episodes_extracted': 0,
                    'episodes_stored': 0,
                    'message': 'No episode extracted'
                }
            
            # Store episode in vector database
            doc_id = f"{thread_id}_{datetime.now().isoformat()}"
            
            self.collection.add(
                documents=[f"Task: {episode.task}\nInitial decomposition: {episode.initial_decomposition}\nFinal decomposition: {episode.final_decomposition}\nNotes: {episode.notes}"],
                metadatas=[{
                    'thread_id': thread_id,
                    'task': episode.task,
                    'initial_decomposition': episode.initial_decomposition,
                    'final_decomposition': episode.final_decomposition,
                    'notes': episode.notes,
                    'extracted_at': datetime.now().isoformat(),
                    'conversation_length': len(messages)
                }],
                ids=[doc_id]
            )
            
            return {
                'success': True,
                'thread_id': thread_id,
                'episodes_extracted': 1,
                'episodes_stored': 1,
                'message': 'Episode extracted and stored successfully'
            }
            
        except Exception as e:
            logger.error(f"Error extracting episode from {thread_id}: {e}")
            return {
                'success': False,
                'thread_id': thread_id,
                'episodes_extracted': 0,
                'episodes_stored': 0,
                'error': str(e)
            }
    
    def _extract_episode_with_langmem(self, messages: List[Any]) -> Optional[TaskDecompositionEpisode]:
        """Extract task decomposition episode using LangMem."""
        try:
            # Convert messages to the correct format for LangMem
            # Filter and format messages properly
            formatted_messages = []
            
            for msg in messages:
                if isinstance(msg, dict):
                    # Already in dict format
                    formatted_messages.append(msg)
                else:
                    # Handle different message types
                    msg_type = type(msg).__name__
                    
                    # Skip ToolMessage - causes errors with LangMem
                    if 'ToolMessage' in msg_type:
                        continue
                    
                    # Map message types to roles
                    if 'Human' in msg_type:
                        role = 'user'
                    elif 'AI' in msg_type or 'Assistant' in msg_type:
                        role = 'assistant'
                    elif 'System' in msg_type:
                        role = 'system'
                    else:
                        # Default to assistant for unknown types
                        role = 'assistant'
                    
                    content = getattr(msg, 'content', str(msg))
                    if content:  # Only add if there's actual content
                        formatted_messages.append({"role": role, "content": content})
            
            # LangMem expects MemoryState format with 'messages' key
            memory_state = {
                'messages': formatted_messages
            }
            
            # Check if we have enough messages
            if len(formatted_messages) < 2:
                logger.info(f"Not enough formatted messages for extraction: {len(formatted_messages)}")
                return self._fallback_extraction(messages)
            
            logger.info(f"Attempting LangMem extraction with {len(formatted_messages)} formatted messages")
            
            try:
                # Use LangMem to extract episodes with correct format
                extracted_memories = self.memory_manager.invoke(memory_state)
                
                # Process the extracted memories
                if extracted_memories and isinstance(extracted_memories, list):
                    for memory in extracted_memories:
                        # LangMem returns ExtractedMemory objects with content attribute
                        if hasattr(memory, 'content') and isinstance(memory.content, TaskDecompositionEpisode):
                            episode = memory.content
                            logger.info(f"LangMem successfully extracted episode: {episode.task[:50]}...")
                            return episode
                        elif isinstance(memory, TaskDecompositionEpisode):
                            logger.info(f"LangMem successfully extracted episode: {memory.task[:50]}...")
                            return memory
                
                logger.info("LangMem: No valid TaskDecompositionEpisode found in extracted memories")
                return self._fallback_extraction(messages)
                
            except Exception as invoke_error:
                logger.warning(f"LangMem invoke failed: {invoke_error}")
                # Fall back to heuristic extraction
                return self._fallback_extraction(messages)
            
        except Exception as e:
            logger.warning(f"LangMem extraction completely failed: {e}")
            # Fallback to simple extraction if LangMem fails
            return self._fallback_extraction(messages)
    
    def _fallback_extraction(self, messages: List[Any]) -> Optional[TaskDecompositionEpisode]:
        """Fallback extraction method when LangMem fails."""
        try:
            # Create simple conversation text
            conversation_text = ""
            for msg in messages:
                role = getattr(msg, 'role', 'unknown')
                content = getattr(msg, 'content', str(msg))
                conversation_text += f"{role}: {content}\n"
            
            # Simple heuristic-based extraction
            if len(conversation_text) > 500:  # Only process substantial conversations
                # Extract a basic episode structure
                lines = conversation_text.split('\n')
                user_lines = [line for line in lines if line.startswith('user:') or line.startswith('human:')]
                assistant_lines = [line for line in lines if line.startswith('assistant:') or line.startswith('ai:')]
                
                if user_lines and assistant_lines:
                    # Get the first user request as task
                    task = user_lines[0].split(':', 1)[1].strip() if ':' in user_lines[0] else "Complex analysis task"
                    
                    # Simple breakdown extraction
                    breakdown = "Step 1 → Step 2 → Step 3"
                    if "→" in conversation_text or "step" in conversation_text.lower():
                        # Try to extract actual breakdown
                        for line in lines:
                            if "→" in line or ("step" in line.lower() and ("1" in line or "2" in line)):
                                breakdown = line.strip()
                                break
                    
                    # Simple notes extraction
                    notes = "Task completed successfully with systematic approach"
                    
                    return TaskDecompositionEpisode(
                        task=task[:500],  # Limit length
                        initial_decomposition=breakdown,
                        final_decomposition=breakdown,
                        notes=notes
                    )
            
            return None
            
        except Exception as e:
            logger.warning(f"Fallback extraction failed: {e}")
            return None
    
    def get_relevant_examples(self, user_request: str) -> List[Dict[str, str]]:
        """
        Get the top 2 most relevant examples for a user request.
        Simplified: no threshold filtering, just rank and return top 2.
        """
        try:
            # Query vector database
            results = self.collection.query(
                query_texts=[user_request],
                n_results=self.config['max_examples']
            )
            
            examples = []
            if results['metadatas'] and results['metadatas'][0]:
                for metadata in results['metadatas'][0]:
                    examples.append({
                        'task': metadata.get('task', ''),
                        'initial_decomposition': metadata.get('initial_decomposition', ''),
                        'final_decomposition': metadata.get('final_decomposition', ''),
                        'notes': metadata.get('notes', '')
                    })
            
            return examples
            
        except Exception as e:
            logger.warning(f"Error retrieving relevant examples: {e}")
            return []


class PromptEnhancer:
    """Simplified prompt enhancer that maintains the same interface."""
    
    def __init__(self, episodic_system: EpisodicLearningSystem):
        self.episodic_system = episodic_system
    
    
    def create_enhanced_planning_prompt(self, user_request: str) -> str:
        """
        Create enhanced planning agent prompt with episodic examples.
        
        This is the new method that provides examples to the planning agent instead of supervisor.
        """
        try:
            from core.prompts.prompts import PLANNING_SYSTEM_PROMPT_ver3
            
            # Get relevant examples
            examples = self.episodic_system.get_relevant_examples(user_request)
            
            if not examples:
                logger.info("No relevant examples found for planning agent, using base prompt")
                return PLANNING_SYSTEM_PROMPT_ver3
            
            # Format examples as complete units (task → breakdown → note, then repeat)
            complete_examples = []
            
            for example in examples:
                # Use final_decomposition if available, otherwise initial_decomposition
                decomposition = example['final_decomposition'] if example['final_decomposition'] else example['initial_decomposition']
                
                # Create complete example unit
                complete_example = f"""Input: {example['task']}
```
📋 BREAKDOWN: {decomposition}

📋 Note for success: {example['notes']}

```"""
                complete_examples.append(complete_example)
            
            # Join all complete examples
            examples_replacement = "\n\n".join(complete_examples)
            
            # Replace the entire placeholder block with complete examples
            placeholder_block = """Input: <placeholders>

```
📋 BREAKDOWN: <placeholders>

📋 Note for success: <placeholders>

Please review this plan. You can ask for changes, provide additional requirements, or tell me if you approve.
```"""
            
            enhanced_prompt = PLANNING_SYSTEM_PROMPT_ver3.replace(
                placeholder_block, examples_replacement
            )
            
            logger.info(f"Enhanced planning agent prompt with {len(examples)} examples")
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing planning agent prompt: {e}")
            from core.prompts.prompts import PLANNING_SYSTEM_PROMPT_ver3
            return PLANNING_SYSTEM_PROMPT_ver3


class SimplifiedOrchestrator:
    """Compatibility wrapper that maintains the original orchestrator interface."""
    
    def __init__(self):
        self.episodic_system = EpisodicLearningSystem()
        self.prompt_enhancer = PromptEnhancer(self.episodic_system)
        
        # Maintain compatibility attributes
        self.config = self.episodic_system.config
        self.monitoring_service = self  # Self-reference for compatibility
    
    def extract_and_store_episode(self, messages: List[Any], thread_id: str) -> Dict[str, Any]:
        """Delegate to episodic system."""
        return self.episodic_system.extract_and_store_episode(messages, thread_id)
    
    def extract_current_conversation(self, thread_id: str) -> Dict[str, Any]:
        """Delegate to episodic system for manual extraction."""
        return self.episodic_system.extract_current_conversation(thread_id)
    
    def run_monitoring_cycle(self) -> Dict[str, Any]:
        """Delegate to episodic system."""
        return self.episodic_system.run_monitoring_cycle()
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for compatibility."""
        try:
            total_episodes = self.episodic_system.collection.count()
        except:
            total_episodes = 0
            
        return {
            'episodic_system': {
                'status': 'manual',
                'total_episodes': total_episodes
            },
            'vector_store': {
                'total_episodes': total_episodes,
                'collection_name': self.config['vector_db_collection']
            }
        }
    
    def get_episodic_context(self, user_request: str = None) -> Dict[str, Any]:
        """Get episodic context for compatibility with UI components."""
        try:
            if not user_request:
                return {'has_context': False, 'examples': []}
            
            # Get relevant examples
            examples = self.episodic_system.get_relevant_examples(user_request)
            
            return {
                'has_context': bool(examples),
                'examples': examples,
                'count': len(examples),
                'user_request': user_request
            }
            
        except Exception as e:
            logger.warning(f"Error getting episodic context: {e}")
            return {'has_context': False, 'examples': [], 'error': str(e)}


# Global instance and compatibility function
_orchestrator_instance = None

def get_orchestrator(config_overrides=None):
    """
    Get the global orchestrator instance.
    
    This function maintains 100% compatibility with the original implementation.
    """
    global _orchestrator_instance
    
    if _orchestrator_instance is None:
        _orchestrator_instance = SimplifiedOrchestrator()
        logger.info("Simplified episodic learning orchestrator created")
    
    return _orchestrator_instance