import sqlite3
from typing import Optional, Literal
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor
from langgraph.graph import START, END
from langchain_core.messages import HumanMessage
from langgraph.types import Command

from app.config import SQLITE_DB_PATH, logger
from core.agents.prediction_agent import build_prediction_agent
from core.agents.research_agent import build_research_agent
from core.agents.data_agent import build_data_agent
from core.agents.planning_agent import build_planning_agent
from core.agents.report_agent import build_report_agent
from core.prompts.prompts import SUPERVISOR_SYSTEM_PROMPT_ver3


def initialize_agents(llm, user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """Initialize all agents with optional episodic learning for planning agent."""
    planning_llm = init_chat_model('gpt-4o', model_provider = 'openai')
    data_llm = init_chat_model('gpt-5', model_provider = 'openai')
    research_llm = init_chat_model('gpt-5-mini', model_provider = 'openai')
    prediction_llm = init_chat_model('gpt-4o', model_provider = 'openai')
    report_llm = init_chat_model('gpt-5', model_provider = 'openai')

    research_agent = build_research_agent(research_llm)
    data_agent = build_data_agent(data_llm)
    prediction_agent = build_prediction_agent(prediction_llm)
    planning_agent = build_planning_agent(planning_llm, user_request, use_episodic_learning)
    report_agent = build_report_agent(report_llm)
    
    return research_agent, data_agent, prediction_agent, planning_agent, report_agent


def initialize_memory():
    """Initialize memory/checkpointer."""
    try:
        connection = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
        memory = SqliteSaver(connection)
        logger.info(f"SqliteSaver initialized with database: {SQLITE_DB_PATH}")
        return memory
    except Exception as e:
        logger.error(f"Error initializing SqliteSaver: {e}")
        # Fallback to in-memory for development
        memory = MemorySaver()
        logger.warning("Falling back to MemorySaver")
        return memory


def _latest_user_text(state) -> str:
    """Extract the latest user message from the state."""
    msgs = state.get("messages") or []
    # Walk from the end to get the most recent user message
    for m in reversed(msgs):
        # LangChain message objects
        if isinstance(m, HumanMessage):
            return m.content or ""
        # Fallbacks for other LC message types that expose `.type`
        if getattr(m, "type", None) == "human":
            return getattr(m, "content", "") or ""
        # Dict-style messages
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "") or ""
    return ""


def route_from_start(state) -> Literal["plan", "skip"]:
    """Route user requests either to planning agent first or directly to supervisor."""
    user_text = _latest_user_text(state)

    # Default to planning if we couldn't find any user text
    if not user_text:
        return "plan"

    # Initialize LLM for routing decision
    llm = init_chat_model('gpt-4o', model_provider='openai')

    prompt = (
        "You are a router for an agent workflow.\n"
        "If the request is concrete and ready to execute, answer 'skip'.\n"
        "If the request is vague/complex and needs decomposition, answer 'plan'.\n"
        f"Request: {user_text}\n"
        "Answer with exactly one word: skip or plan."
    )

    out = llm.invoke(prompt)
    # llm.invoke returns an AIMessage (LangChain) or a string depending on your wrapper
    out_text = getattr(out, "content", str(out)).strip().lower()

    return "skip" if out_text.startswith("skip") else "plan"


def route_from_planning(state) -> Literal["human_chat", "supervisor"]:
    """Route from planning agent - check if human message contains approval terms."""
    messages = state.get("messages", [])
    
    # Get all human messages in chronological order
    human_messages = []
    for msg in messages:
        human_content = None
        
        # Check different message formats for human messages
        if hasattr(msg, 'type') and msg.type == 'human':
            human_content = msg.content
        elif isinstance(msg, dict) and msg.get('role') == 'user':
            human_content = msg.get('content', '')
        elif hasattr(msg, 'role') and msg.role == 'user':
            human_content = msg.content
        
        if human_content:
            human_messages.append(human_content)
    
    # CRITICAL FIX: Only check messages AFTER the first one (exclude original user request)
    if len(human_messages) <= 1:
        # Only the original request exists, no approval possible yet
        logger.info("Only original request exists, routing to human_chat for plan review")
        return "human_chat"
    
    # Check the most recent human message (excluding the first) for approval terms
    most_recent_feedback = human_messages[-1]
    content_lower = most_recent_feedback.lower().strip()
    approval_terms = [
        "approved", "approve", "looks good", "send to supervisor", 
        "proceed", "go ahead", "execute", "ok", "good", "yes"
    ]
    
    # Check if any approval term is found in feedback messages
    for term in approval_terms:
        if term in content_lower:
            logger.info(f"Approval detected in human feedback: '{most_recent_feedback}' contains '{term}'")
            return "supervisor"
    
    # No approval found in feedback messages
    logger.info(f"Human feedback found but no approval: '{most_recent_feedback}'")
    return "human_chat"



def human_chat_node(state):
    """Handle human-in-the-loop conversation for plan approval."""
    from langgraph.types import interrupt
    
    # Get the plan from state
    messages = state.get("messages", [])
    planning_output = ""
    
    # Extract the latest planning agent output
    for msg in reversed(messages):
        if hasattr(msg, 'name') and msg.name == 'planning_agent':
            planning_output = msg.content
            break
        elif isinstance(msg, dict) and msg.get('name') == 'planning_agent':
            planning_output = msg.get('content', '')
            break
    
    # Interrupt for human input with the current plan
    human_input = interrupt({
        "type": "plan_review",
        "plan": planning_output, 
        "message": "Please review the plan above. You can:\n1. Ask for changes or refinements\n2. Click 'Approve Plan' to proceed with execution"
    })
    
    # Check if human approved the plan or wants to refine it
    if human_input and human_input.lower().strip() == "approved":
        return {"plan_approved": True}
    else:
        # Continue conversation - add human feedback to messages
        if human_input:
            messages = state.get("messages", [])
            messages.append(HumanMessage(content=human_input))
        return {"messages": messages, "plan_approved": False}



def create_app(user_request: Optional[str] = None, use_episodic_learning: bool = True):
    """
    Initialize the LangGraph application with episodic learning for planning agent.
    
    Args:
        user_request: Current user request for context-aware planning agent enhancement
        use_episodic_learning: Whether to use episodic learning for planning agent
    """
    llm = init_chat_model('gpt-5-mini', model_provider = 'openai')
    
    # Build agents with episodic learning for planning agent
    research_agent, data_agent, prediction_agent, planning_agent, report_agent = initialize_agents(
        llm, user_request, use_episodic_learning
    )
    
    # Initialize memory
    memory = initialize_memory()
    
    # Create supervisor with execution agents (planning agent added separately, report agent included but routed to END)
    supervisor_agent = create_supervisor(
        [research_agent, prediction_agent, data_agent, report_agent],
        model=llm,
        output_mode="full_history",
        prompt=SUPERVISOR_SYSTEM_PROMPT_ver3,
        add_handoff_message = True,
        supervisor_name='supervisor'
    )

    # Modify the graph structure to add routing and human-in-the-loop
    # Remove the default START -> supervisor edge
    supervisor_agent.edges.remove(('__start__', 'supervisor'))
    
    # Add planning agent as a separate node
    supervisor_agent.add_node('planning_agent', planning_agent)
    
    # Add human chat node for plan approval
    supervisor_agent.add_node('human_chat', human_chat_node)
    
    # Remove default edge from report_agent back to supervisor (similar to planning_agent)
    supervisor_agent.edges.remove(('report_agent', 'supervisor'))
    
    # Add conditional routing from START
    supervisor_agent.add_conditional_edges(
        START,
        route_from_start,
        {"plan": "planning_agent", "skip": "supervisor"},
    )
    
    # Add conditional routing from planning agent
    supervisor_agent.add_conditional_edges(
        'planning_agent',
        route_from_planning,
        {"human_chat": "human_chat", "supervisor": "supervisor"},
    )
    
    # Add edge from human_chat back to planning_agent for refinements
    supervisor_agent.add_edge('human_chat', 'planning_agent')
    
    # Add edge from report_agent to END (report_agent is managed by supervisor, no direct edge needed)
    supervisor_agent.add_edge('report_agent', END)

    app = supervisor_agent.compile(checkpointer=memory)
    
    if use_episodic_learning and user_request:
        logger.info("Created app with episodic learning enhancement and separate planning node")
    else:
        logger.info("Created app with standard agents and separate planning node")
    
    return app