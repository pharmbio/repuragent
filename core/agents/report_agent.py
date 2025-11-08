from langgraph.prebuilt import create_react_agent

from app.config import logger
from core.prompts.prompts import REPORT_SYSTEM_PROMPT


def build_report_agent(llm):
    """
    Build report agent for generating comprehensive workflow summaries.
    
    Args:
        llm: Language model instance
        
    Returns:
        Report agent
    """
    try:
        # Create report agent without additional tools - it uses conversation history
        report_agent = create_react_agent(
            model=llm, 
            tools=[], 
            name='report_agent',
            prompt=REPORT_SYSTEM_PROMPT
        )
        
        logger.info("Report agent created successfully")
        return report_agent
        
    except Exception as e:
        logger.error(f"Error creating report agent: {e}")
        raise e
