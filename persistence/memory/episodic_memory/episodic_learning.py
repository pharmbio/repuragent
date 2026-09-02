import asyncio
import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, convert_to_messages
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field

from app.config import (
    DATABASE_PATH,
    EPISODE_EXTRACTION_MODEL,
    EPISODIC_MAX_EXAMPLES,
    MEMORY_DIR,
    OPENAI_API_KEY,
    logger,
)
from core.agents.context import render_transcript


class TaskDecompositionEpisode(BaseModel):
    '''An episode captures how to plan a specific situation, including context and task decomposition.'''

    task: str = Field(..., description="The relevant context and requests from the users")
    initial_decomposition: str = Field(..., description="How the task was first broken down")
    final_decomposition: str = Field(..., description="The complete final task sequence including all additions/modifications")
    notes: str = Field(..., description="Lesson learnt during the execution that can enhance the final results")


class EpisodeExtraction(BaseModel):
    '''What the extractor returns, including the option to record nothing.

    A bare `TaskDecompositionEpisode` has four required fields, so a model asked for one
    must produce something — even for a conversation with no plan in it. The result was
    fabricated precedent, served back to the planning agent as if it were experience.
    `worth_recording` is what lets it decline.
    '''

    worth_recording: bool = Field(
        description=(
            "True only if this conversation shows a genuine multi-step plan that was "
            "executed. False for a short request, a question, an abandoned run, or "
            "anything whose decomposition would not help plan a future task."
        )
    )
    reason: str = Field(
        default="",
        description="One line on why it is or is not worth recording.",
    )
    episode: Optional[TaskDecompositionEpisode] = Field(
        default=None,
        description="The episode, when worth_recording is true. Omit otherwise.",
    )


EXTRACTION_INSTRUCTIONS = """You read a finished drug-repurposing conversation and record
how it was planned, so that a future request of the same shape can be planned from
precedent rather than from scratch.

Record an episode only when the conversation actually contains one: a multi-step plan that
was approved and executed. A single lookup, a question about the assistant, or a run that
was abandoned has nothing to teach — say so with `worth_recording: false` rather than
inventing a decomposition. A fabricated episode is worse than none, because it will be
shown to the planner as experience.

When it is worth recording:

- **task** — the user's request, with the context needed to recognise a similar one later.
- **initial_decomposition** — the breakdown the planner first proposed, as
  `[Step 1] → [Step 2] → [Step 3]`, step titles only, no commentary.
- **final_decomposition** — the breakdown as it stood when execution began, after any
  revisions the user asked for. Same format. If it was never revised, repeat the initial one.
- **notes** — what would make the next attempt at this shape of task go better: an ordering
  that mattered, a step that turned out to be unnecessary, a dependency that was missed, a
  tool that failed and what was done instead. Concrete lessons only; no praise, and nothing
  the transcript does not support."""


class EpisodicLearningSystem:
    '''Simplified episodic learning system that maintains all existing functionality.'''

    def __init__(self):
        '''Initialize the episodic learning system - simplified manual mode.'''

        # Simplified configuration for manual mode
        self.config = {
            'vector_db_collection': 'task_episodes',
            'extraction_model': EPISODE_EXTRACTION_MODEL,
            'max_examples': EPISODIC_MAX_EXAMPLES,
        }

        # Initialize components
        self._setup_vector_store()
        self._setup_llm()

        logger.info("Episodic Learning System initialized (manual extraction only)")

    def _setup_vector_store(self):
        '''Initialize ChromaDB vector store.'''

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
        '''Initialize the extraction model.

        `langmem.create_memory_manager` used to do this. It was dropped because it pins
        `langgraph<0.7`, which cannot coexist with the v1 line this codebase runs on — a
        clean `pip install` of the two fails outright. What it provided was one
        structured-extraction call, which the model does natively.
        '''

        try:
            self.llm = init_chat_model(
                self.config["extraction_model"],
                model_provider="openai",
                api_key=OPENAI_API_KEY,
            )
            self.extractor = self.llm.with_structured_output(EpisodeExtraction)

            logger.info("Episode extractor initialized: %s", self.config["extraction_model"])
        except Exception as e:
            logger.error(f"Failed to initialize the episode extractor: {e}")
            raise

    # Background monitoring methods removed - manual mode only

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        '''Deprecated - manual extraction only.

        Returns:
        ----------
        status (dict): the no-op result kept for callers that still invoke it — extraction is manual now.
        '''

        return {'success': False, 'error': 'Use extract_current_conversation for manual extraction'}

    def extract_current_conversation(self, thread_id: str) -> Dict[str, Any]:
        '''Extract episodes from the current conversation (always overwrites existing).
        
        Parameters:
        ---------
        thread_id (str): The ID of the current conversation thread

        Returns:
        ----------
        result (dict): Dict with extraction results
        '''

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
        '''Load conversation messages synchronously by delegating to the async loader.

        Parameters:
        ---------
        thread_id (str): the conversation to load.

        Returns:
        ----------
        messages (list): its messages, by driving the async loader to completion.
        '''

        try:
            return asyncio.run(self._aload_conversation_messages(thread_id))
        except RuntimeError as runtime_error:
            # asyncio.run cannot be nested inside a running loop, fall back to a dedicated loop
            if "asyncio.run()" in str(runtime_error):
                loop = asyncio.new_event_loop()
                try:
                    return loop.run_until_complete(self._aload_conversation_messages(thread_id))
                finally:
                    loop.close()
            logger.warning(f"Async loading failed for thread {thread_id}: {runtime_error}")
            return []
        except Exception as e:
            logger.warning(f"Could not load messages for thread {thread_id}: {e}")
            return []

    async def _aload_conversation_messages(self, thread_id: str) -> List[Any]:
        '''Async helper to load conversation messages from the SQLite checkpointer.

        A **fresh** connection rather than the app's shared checkpointer, because the
        caller is a synchronous Gradio handler that drives this with `asyncio.run` on
        a loop of its own: an `aiosqlite` connection belongs to the loop that opened
        it, so reusing the singleton here would await futures from another loop. WAL
        mode makes a second reader free.

        Parameters:
        ---------
        thread_id (str): the conversation to load.

        Returns:
        ----------
        messages (list): its messages, read from the SQLite checkpointer.
        '''

        if not DATABASE_PATH.exists():
            logger.warning("No database at %s; cannot load conversation messages", DATABASE_PATH)
            return []

        async with AsyncSqliteSaver.from_conn_string(str(DATABASE_PATH)) as checkpointer:
            checkpoint = await checkpointer.aget({"configurable": {"thread_id": thread_id}})

        if not checkpoint:
            logger.info(f"No checkpoint found for thread {thread_id}")
            return []

        messages = checkpoint.get('channel_values', {}).get('messages', [])
        if not messages:
            logger.info(f"No messages stored in checkpoint for thread {thread_id}")
            return []

        try:
            return convert_to_messages(messages)
        except Exception as conversion_error:
            logger.warning(f"Could not convert checkpoint messages for thread {thread_id}: {conversion_error}")
            return messages

    def extract_and_store_episode(self, messages: List[Any], thread_id: str) -> Dict[str, Any]:
        '''Extract episodes from conversation and store them (always overwrites).

        Parameters:
        ---------
        messages (list): the conversation to learn from.
        thread_id (str): which conversation it is, used as the record's key.

        Returns:
        ----------
        result (dict): what was stored. Always overwrites, so re-extracting a conversation does not accumulate duplicates.
        '''

        try:
            # Always delete existing entries for this thread first
            existing_docs = self.collection.get(
                where={"thread_id": thread_id}
            )
            if existing_docs['ids']:
                self.collection.delete(ids=existing_docs['ids'])
                logger.info(f"Replaced {len(existing_docs['ids'])} existing episodes for thread {thread_id}")

            episode = self._extract_episode(messages)

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

    def _extract_episode(self, messages: List[Any]) -> Optional[TaskDecompositionEpisode]:
        '''One structured call over the conversation. None when there is nothing to record.

        The heuristic fallback this replaced returned a hard-coded
        `"Step 1 → Step 2 → Step 3"` breakdown and the note `"Task completed successfully
        with systematic approach"` whenever the real extraction failed. Those were stored as
        episodes and later handed to the planning agent as precedent, so a failure quietly
        became fabricated experience. There is no fallback now: extraction either produces a
        real episode or records nothing.

        Parameters:
        ---------
        messages (list): the conversation to summarize.

        Returns:
        ----------
        episode (TaskDecompositionEpisode): how the task was decomposed, or None when there is nothing worth recording.
        '''

        transcript = render_transcript(messages)
        if len(transcript.strip()) < 400:
            logger.info("Conversation too short to hold a plan worth recording")
            return None

        try:
            result: EpisodeExtraction = self.extractor.invoke(
                [
                    SystemMessage(content=EXTRACTION_INSTRUCTIONS),
                    HumanMessage(content=f"--- CONVERSATION ---\n{transcript}\n"),
                ]
            )
        except Exception as exc:  # noqa: BLE001 - recording a lesson is never critical
            logger.warning("Episode extraction failed: %s", exc)
            return None

        if not result.worth_recording or result.episode is None:
            logger.info("Nothing worth recording: %s", result.reason or "no reason given")
            return None
        return result.episode

    def get_relevant_examples(self, user_request: str) -> List[Dict[str, str]]:
        '''Get the top 2 most relevant examples for a user request.
        Simplified: no threshold filtering, just rank and return top 2.

        Parameters:
        ---------
        user_request (str): the request to find precedent for.

        Returns:
        ----------
        examples (list): the two most similar past tasks and how they were decomposed.
        '''

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


# `PromptEnhancer` used to rewrite a placeholder block inside the planning
# prompt with retrieved examples, which meant the whole graph had to be rebuilt
# per user message just to change that string. Examples now reach the planner
# through its context middleware instead — see
# `core/agents/context.py::episodic_examples_block`.


class SimplifiedOrchestrator:
    '''Compatibility wrapper that maintains the original orchestrator interface.'''

    def __init__(self):
        self.episodic_system = EpisodicLearningSystem()

        # Maintain compatibility attributes
        self.config = self.episodic_system.config
        self.monitoring_service = self  # Self-reference for compatibility

    def extract_and_store_episode(self, messages: List[Any], thread_id: str) -> Dict[str, Any]:
        '''Delegate to episodic system.

        Parameters:
        ---------
        messages (list): the conversation to learn from.
        thread_id (str): which conversation it is.

        Returns:
        ----------
        result (dict): whatever the episodic system reported.
        '''

        return self.episodic_system.extract_and_store_episode(messages, thread_id)

    def extract_current_conversation(self, thread_id: str) -> Dict[str, Any]:
        '''Delegate to episodic system for manual extraction.

        Parameters:
        ---------
        thread_id (str): the conversation to extract.

        Returns:
        ----------
        result (dict): whatever the episodic system reported.
        '''

        return self.episodic_system.extract_current_conversation(thread_id)

    def run_monitoring_cycle(self) -> Dict[str, Any]:
        '''Delegate to episodic system.

        Returns:
        ----------
        status (dict): whatever the episodic system reported.
        '''

        return self.episodic_system.run_monitoring_cycle()

    def get_system_status(self) -> Dict[str, Any]:
        '''Get system status for compatibility.

        Returns:
        ----------
        status (dict): the store's state, in the shape the older callers expect.
        '''

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
        '''Get episodic context for compatibility with UI components.

        Parameters:
        ---------
        user_request (str): the request to find precedent for, or None.

        Returns:
        ----------
        context (dict): the examples block the UI components expect.
        '''

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
    '''Get the global orchestrator instance.
    
    This function maintains 100% compatibility with the original implementation.

    Parameters:
    ---------
    config_overrides (dict): settings to override on the shared instance.

    Returns:
    ----------
    orchestrator (SimplifiedOrchestrator): the global instance, built once.
    '''

    global _orchestrator_instance

    if _orchestrator_instance is None:
        _orchestrator_instance = SimplifiedOrchestrator()
        logger.info("Simplified episodic learning orchestrator created")

    return _orchestrator_instance
