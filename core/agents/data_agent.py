from typing import Annotated, List, Optional, Sequence
from typing_extensions import TypedDict
from logging import getLogger

logger = getLogger(__name__)

from langchain_core.messages import SystemMessage, BaseMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Checkpointer
from langgraph.store.base import BaseStore
from langchain_core.language_models import LanguageModelLike
from langchain_core.runnables import RunnableConfig
from langgraph.managed import IsLastStep, RemainingSteps
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from backend.utils.local_python_executor import local_python_executor, reset_executor_state, BASE_BUILTIN_MODULES
from backend.utils.fuzzy_path import prompt_with_file_path
from core.prompts.prompts import DATA_SYSTEM_PROMPT_ver3

DEFAULT_AUTHORIZED_IMPORTS = ['json','pathlib', 'sqlalchemy', 'dotenv', 'os', 'sys', 'pandas','rdkit', 'numpy', 'matplotlib', 'rdkit', 'seaborn', 'scipy', 'sklearn', 'fuzzywuzzy', "Bio", 'streamlit']
authorized_imports = sorted(set(BASE_BUILTIN_MODULES) | set(DEFAULT_AUTHORIZED_IMPORTS))

@tool
def python_executor(code: str):
    """Execute Python code safely with restricted imports.
    
    Variables defined in previous executions are preserved and available in subsequent
    executions, providing persistent state across code blocks within the session.
    
    Args:
        code (str): The code to execute.
        
    Returns:
        The result of the execution.
    """
    
    return local_python_executor(code, authorized_imports)


@tool
def reset_python_state():
    """Reset the Python execution state.
    
    This clears all variables and functions defined in previous executions,
    providing a clean slate for new code execution. Use this when you need
    to start fresh or clear accumulated state.
    
    Returns:
        str: Confirmation message.
    """
    reset_executor_state()
    return "Python execution state has been reset. All previous variables and functions have been cleared."




# Define the state for our graph - aligned with template's AgentState
class AgentState(TypedDict):
    """The state of the agent."""

    messages: Annotated[Sequence[BaseMessage], add_messages]

    is_last_step: IsLastStep

    remaining_steps: RemainingSteps

# From LangGraph prebuilt 
def create_agent_builder(
    model: Optional[LanguageModelLike] = None,
    tools: List = None,
    prompt: Optional[str] = None,
    checkpointer: Optional[Checkpointer] = None,
    store: Optional[BaseStore] = None,
    interrupt_before: Optional[List[str]] = None,
    interrupt_after: Optional[List[str]] = None,
    debug: bool = False,
    name: Optional[str] = None
) -> CompiledStateGraph:
    """Creates an agent graph that calls tools in a loop until a stopping condition is met.
    
    Args:
        model: The LangChain chat model that supports tool calling. Defaults to gpt-4o.
        tools: A list of tools. Defaults to [python_tool, prompt_with_file_path].
        prompt: System prompt for the LLM. Defaults to DATA_SYSTEM_PROMPT_ver2.
        checkpointer: An optional checkpoint saver object.
        store: An optional store object.
        interrupt_before: An optional list of node names to interrupt before.
        interrupt_after: An optional list of node names to interrupt after.
        debug: A flag indicating whether to enable debug mode.
        name: An optional name for the CompiledStateGraph.
        
    Returns:
        A compiled LangGraph StateGraph.
    """
    
    # Set defaults
    if model is None:
        raise ValueError("A model must be provided.")
    if tools is None:
        raise ValueError("A tool must be provided.")
    if prompt is None:
        prompt = "You are a helpful data analyst."
    
    # Create tools node
    tools_node = ToolNode(tools=tools)
    
    # Bind tools to model
    model_with_tools = model.bind_tools(tools=tools)
    
    # Define the agent node - aligned with template's call_model pattern
    def call_model(state: AgentState, config: RunnableConfig) -> AgentState:
        """Call the model with the current state."""
        messages = state["messages"]
        # Add system prompt to the beginning
        payload = [SystemMessage(content=prompt)] + messages
        response = model_with_tools.invoke(payload, config)
        # Add agent name if provided
        if name:
            response.name = name
        return {"messages": [response]}
    
    async def acall_model(state: AgentState, config: RunnableConfig) -> AgentState:
        """Async version of call_model."""
        messages = state["messages"]
        payload = [SystemMessage(content=prompt)] + messages
        response = await model_with_tools.ainvoke(payload, config)
        if name:
            response.name = name
        return {"messages": [response]}
    
    # Define the conditional logic - aligned with template's should_continue pattern
    def should_continue(state: AgentState) -> str:
        """Determine whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        # Use tools_condition logic but return string instead of calling it directly
        
        if isinstance(last_message, AIMessage) and last_message.tool_calls:
            return "tools"
        else:
            return "__end__"
    
    # Node names - aligned with template
    AGENT_NODE = "agent"
    TOOLS_NODE = "tools"
    
    # Create the graph
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node(AGENT_NODE, call_model)
    builder.add_node(TOOLS_NODE, tools_node)
    
    # Add edges - aligned with template's pattern
    builder.add_conditional_edges(
        AGENT_NODE,
        should_continue,
        path_map=[TOOLS_NODE, "__end__"]
    )
    builder.add_edge(TOOLS_NODE, AGENT_NODE)
    
    # Set entry point
    builder.set_entry_point(AGENT_NODE)
    
    # Compile with template's parameters
    return builder.compile(
        checkpointer=checkpointer,
        store=store,
        interrupt_before=interrupt_before,
        interrupt_after=interrupt_after,
        debug=debug,
        name=name,
    )

# Keep for backwards compatibility
create_tool_calling_executor = create_agent_builder

__all__ = [
    "create_agent_builder",
    "create_tool_calling_executor", 
    "AgentState",
]


def build_data_agent(llm):
    return create_agent_builder(model=llm, tools=[python_executor, reset_python_state, prompt_with_file_path], prompt = DATA_SYSTEM_PROMPT_ver3, name="data_agent")
