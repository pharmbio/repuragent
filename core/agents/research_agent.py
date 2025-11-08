from langgraph.prebuilt import create_react_agent

from app.config import logger
from core.prompts.prompts import RESEARCH_SYSTEM_PROMPT_ver3
from backend.utils.research_tools import literature_search_pubmed, protocol_search_sop
from backend.utils.kgg_tools import (
        search_disease_id,
        create_knowledge_graph,
        extract_drugs_from_kg,
        extract_proteins_from_kg,
        extract_pathways_from_kg,
        extract_mechanism_of_actions_from_kg,
        getDrugsforProteins, 
        getDrugsforMechanisms,
        getDrugsforPathways
    )
logger.info("Using KGG-based modular tools")
KGG_AVAILABLE = True

def build_research_agent(llm):
    # Build tools list based on availability
    tools = [
        literature_search_pubmed,
        protocol_search_sop,
        search_disease_id,
        create_knowledge_graph,
        extract_drugs_from_kg,
        extract_proteins_from_kg,
        extract_pathways_from_kg,
        extract_mechanism_of_actions_from_kg,
        getDrugsforProteins,
        getDrugsforPathways,
        getDrugsforMechanisms
    ]


    research_agent = create_react_agent(
        model = llm, 
        tools = tools, 
        name='research_agent',
        prompt = RESEARCH_SYSTEM_PROMPT_ver3)
    
    return research_agent
