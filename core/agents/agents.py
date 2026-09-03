from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from langchain.agents import create_agent

from core.agents.context import AgentGraphState, bound_tool_result, build_context_middleware
from core.agents.handoff import build_handoff_tools
from core.prompts.prompts import (
    DATA_SYSTEM_PROMPT,
    PLANNING_SYSTEM_PROMPT,
    PREDICTION_SYSTEM_PROMPT,
    RESEARCH_SYSTEM_PROMPT,
    SUPERVISOR_POSTURE_FOLLOWUP,
    SUPERVISOR_POSTURE_FREE,
    SUPERVISOR_POSTURE_PLAN,
    SUPERVISOR_SYSTEM_PROMPT,
)
from core.tools.chemical_tools import annotate_chemicals
from core.tools.knowledge_graph_tools import (
    create_knowledge_graph,
    extract_drugs_from_kg,
    extract_mechanism_of_actions_from_kg,
    extract_pathways_from_kg,
    extract_proteins_from_kg,
    getDrugsforMechanisms,
    getDrugsforPathways,
    getDrugsforProteins,
    search_disease_id,
)
from core.tools.plan_tools import plan_status, plan_update
from core.tools.prediction_tools import (
    AMES_classifier,
    BBB_classifier,
    CYP1A2_classifier,
    CYP2C9_classifier,
    CYP2C19_classifier,
    CYP2D6_classifier,
    CYP3A4_classifier,
    Lipophilicity_regressor,
    PAMPA_classifier,
    PGP_classifier,
    Solubility_regressor,
    hERG_classifier,
    predict_repurposedrugs,
)
from core.tools.python_executor import python_executor, reset_python_state
from core.tools.read_files import read_files
from core.tools.research_tools import literature_search_litsense, protocol_search_sop

RESEARCH_TOOLS = [
    literature_search_litsense,
    protocol_search_sop,
    search_disease_id,
    create_knowledge_graph,
    extract_drugs_from_kg,
    extract_proteins_from_kg,
    extract_pathways_from_kg,
    extract_mechanism_of_actions_from_kg,
    getDrugsforProteins,
    getDrugsforPathways,
    getDrugsforMechanisms,
    annotate_chemicals,
    python_executor,
    reset_python_state, 
    read_files
]

PREDICTION_TOOLS = [
    CYP3A4_classifier,
    CYP2C19_classifier,
    CYP2D6_classifier,
    CYP1A2_classifier,
    CYP2C9_classifier,
    hERG_classifier,
    AMES_classifier,
    PGP_classifier,
    PAMPA_classifier,
    BBB_classifier,
    Solubility_regressor,
    Lipophilicity_regressor,
    predict_repurposedrugs,
]

DATA_TOOLS = [python_executor, reset_python_state, read_files, protocol_search_sop]

REPORT_TOOLS = [read_files, python_executor]

PLANNING_TOOLS = [
    literature_search_litsense,
    protocol_search_sop,
    read_files
]

# One posture block per route. The supervisor's tools and specialists are identical
# in all three cases and only its guidance differs, so this is a middleware
# concern rather than three separate agents.
SUPERVISOR_POSTURES: Dict[str, str] = {
    "complex": SUPERVISOR_POSTURE_PLAN,
    "simple": SUPERVISOR_POSTURE_FREE,
    "follow_up": SUPERVISOR_POSTURE_FREE,
}

def _supervisor_posture(state: Dict[str, Any]) -> str:
    category = str(state.get("task_category") or "complex")
    if category == "follow_up":
        return SUPERVISOR_POSTURE_FOLLOWUP
    return SUPERVISOR_POSTURES.get(category, SUPERVISOR_POSTURE_PLAN)


def _build(
    *,
    model,
    name: str,
    prompt: str,
    tools: List[Any],
    compress: bool,
    posture_block: Optional[Callable[[Dict[str, Any]], str]] = None,
    include_plan: bool = True,
    include_artifacts: bool = True,
    include_episodes: bool = False,
    include_goal: bool = True,
    isolated: bool = False,
    collapse_delegations: bool = False,
):
    return create_agent(
        model=model,
        tools=tools,
        system_prompt=prompt,
        name=name,
        state_schema=AgentGraphState,
        middleware=[
            bound_tool_result,
            build_context_middleware(
                role=name,
                compress=compress,
                posture_block=posture_block,
                include_plan=include_plan,
                include_artifacts=include_artifacts,
                include_episodes=include_episodes,
                include_goal=include_goal,
                isolated=isolated,
                collapse_delegations=collapse_delegations,
            )
        ],
    )


def _build_specialist(model, *, name: str, prompt: str, tools: List[Any], compress: bool):
    '''A context-isolated specialist.

    Parameters:
    ---------
    model (chat model): the model this specialist runs on.
    name (str): the node name, which is also what the handoff tool routes to.
    prompt (str): its role prompt, ending in the `WHAT TO HAND BACK` section.
    tools (list): the tools it may call.
    compress (boolean): whether older turns are folded into the carried summary.

    Returns:
    ----------
    agent (CompiledGraph): the specialist, built with `state_schema=AgentGraphState` so the scope and plan pointer reach its middleware.
    '''

    return _build(
        model=model,
        name=name,
        prompt=prompt,
        tools=tools,
        compress=compress,
        isolated=True,  # only recieving brief
        include_goal=False,
        include_plan=False,
        include_artifacts=True,
    )


def build_planning_agent(model, *, compress: bool = True):
    '''The planner: no tools, and precedent from episodic memory.

    Parameters:
    ---------
    model (chat model): the model the planner runs on.
    compress (boolean): whether older turns are folded into the carried summary.

    Returns:
    ----------
    agent (CompiledGraph): the planner — no tools, and episodic precedent in its context.
    '''

    return _build(
        model=model,
        name="planning_agent",
        prompt=PLANNING_SYSTEM_PROMPT,
        tools=PLANNING_TOOLS,
        compress=compress,
        include_plan=False,
        include_episodes=True,
    )


def build_supervisor_agent(model, *, compress: bool = True):
    '''The orchestrator: handoff tools plus the two plan-file tools.

    Parameters:
    ---------
    model (chat model): the model the supervisor runs on.
    compress (boolean): whether older turns are folded into the carried summary.

    Returns:
    ----------
    agent (CompiledGraph): the orchestrator, holding the handoff tools plus `plan_status` and `plan_update`.
    '''

    return _build(
        model=model,
        name="supervisor",
        prompt=SUPERVISOR_SYSTEM_PROMPT,
        tools=[*build_handoff_tools(), plan_status, plan_update],
        compress=compress,
        posture_block=_supervisor_posture,
        collapse_delegations=True,
    )


def build_research_agent(model, *, compress: bool = True):
    return _build_specialist(
        model, name="research_agent", prompt=RESEARCH_SYSTEM_PROMPT, tools=RESEARCH_TOOLS, compress=compress
    )


def build_prediction_agent(model, *, compress: bool = True):
    return _build_specialist(
        model, name="prediction_agent", prompt=PREDICTION_SYSTEM_PROMPT, tools=PREDICTION_TOOLS, compress=compress
    )


def build_data_agent(model, *, compress: bool = True):
    return _build_specialist(
        model, name="data_agent", prompt=DATA_SYSTEM_PROMPT, tools=DATA_TOOLS, compress=compress
    )


def build_report_agent(model, *, name: str, prompt: str, compress: bool = True):
    '''

    Parameters:
    ---------
    model (chat model): the model this report variant runs on.
    name (str): the node name, which `REPORT_NODES` styling keys off.
    prompt (str): the report format for this route.
    compress (boolean): whether older turns are folded into the carried summary.

    Returns:
    ----------
    agent (CompiledGraph): the report agent, which keeps the full view because it is the one that has to cite evidence.
    '''

    return _build(
        model=model,
        name=name,
        prompt=prompt,
        tools=REPORT_TOOLS,
        compress=compress,
    )


__all__ = [
    "DATA_TOOLS",
    "PREDICTION_TOOLS",
    "REPORT_TOOLS",
    "RESEARCH_TOOLS",
    "build_data_agent",
    "build_planning_agent",
    "build_prediction_agent",
    "build_report_agent",
    "build_research_agent",
    "build_supervisor_agent",
]
