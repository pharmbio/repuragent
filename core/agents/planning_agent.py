from typing import Optional

from langgraph.prebuilt import create_react_agent

from app.config import logger
from core.prompts.prompts import PLANNING_SYSTEM_PROMPT_ver3
from backend.memory.episodic_memory.episodic_learning import get_orchestrator
from backend.utils.research_tools import literature_search_pubmed, protocol_search_sop


def build_planning_agent_core(llm, user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """
    Create core planning agent with optional episodic learning enhancement.
    
    Args:
        llm: Language model instance
        user_request: Current user request for context-aware enhancement
        use_episodic_learning: Whether to use episodic learning for planning agent
        
    Returns:
        Enhanced or standard planning agent
    """
    if not use_episodic_learning or not user_request:
        tools = [literature_search_pubmed, protocol_search_sop]
        planning_agent = create_react_agent(
            model=llm, 
            tools=tools, 
            name='planning_agent',
            prompt=PLANNING_SYSTEM_PROMPT_ver3
        )
        return planning_agent
    
    try:
        # Get the orchestrator
        orchestrator = get_orchestrator()
        
        # Create enhanced planning prompt
        enhanced_prompt = orchestrator.prompt_enhancer.create_enhanced_planning_prompt(user_request)
        
        # Create planning agent with enhanced prompt
        tools = [literature_search_pubmed, protocol_search_sop]
        planning_agent = create_react_agent(
            model=llm, 
            tools=tools, 
            name='planning_agent',
            prompt=enhanced_prompt
        )
        
        logger.info(f"Enhanced planning agent created for request: {user_request[:50]}...")
        return planning_agent
        
    except Exception as e:
        logger.warning(f"Error creating enhanced planning agent, falling back to standard: {e}")
        tools = [literature_search_pubmed, protocol_search_sop]
        planning_agent = create_react_agent(
            model=llm, 
            tools=tools, 
            name='planning_agent',
            prompt=PLANNING_SYSTEM_PROMPT_ver3
        )
        
        return planning_agent


def build_planning_agent(llm, user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """
    Main entry point for building planning agent.
    """
    return build_planning_agent_core(llm, user_request, use_episodic_learning)
