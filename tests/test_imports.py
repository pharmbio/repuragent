'''Every module imports.

Cheap, and it catches the failure mode a refactor produces most often: a module
moved, its importers not updated. It also guards the specific breakage that made
this refactor necessary — `langchain.retrievers` and friends disappearing in
LangChain v1, which took down the SOP retriever and everything downstream of it.
'''

from __future__ import annotations

import importlib
from typing import List

MODULES: List[str] = [
    # configuration and foundations
    "app.config",
    "app.app_config",
    "app.state",
    # persistence and scoping
    "backend.db",
    "backend.db.connection",
    "backend.db.checkpointer",
    "backend.db.repository",
    "backend.db.legacy_import",
    "backend.utils.storage_paths",
    "backend.utils.output_paths",
    "backend.utils.cancellation",
    "backend.utils.plan_store",
    "backend.utils.figure_capture",
    "backend.utils.local_python_executor",
    "backend.utils.retention",
    # retrieval
    "backend.sop_rag.config",
    "backend.sop_rag.bm25_corpus",
    "backend.sop_rag.sop_retriever",
    "backend.sop_rag.sop_indexer",
    # knowledge graph
    "backend.kgg.api_utils",
    # tools
    "core.tools.plan_tools",
    "core.tools.python_executor",
    "core.tools.read_files",
    "core.tools.research_tools",
    "core.tools.prediction_tools",
    "core.tools.knowledge_graph_tools",
    "core.tools.chemical_tools",
    # agents and graph
    "core.prompts.prompts",
    "core.agents.context",
    "core.agents.handoff",
    "core.agents.agents",
    "core.agents.agentic_system",
    # memory
    "persistence.memory.episodic_memory.episodic_learning",
    # app layer
    "app.langgraph_runner",
    "app.conversation_store",
    "app.timeline_store",
    "app.files",
    "app.downloads",
    "app.run_controller",
    "app.session",
    "app.partners",
    # ui
    "app.ui.formatters",
    "app.ui.tool_display",
    "app.ui.chat_timeline",
    "app.ui.approval",
    "app.ui.progress_panel",
    "app.ui.conversation_panel",
    "app.ui.projection",
    "app.ui.theme",
    "app.ui.scripts",
    "app.ui.assets",
    "app.ui.layout",
    "app.gradio_app",
    "main",
]


def test_all_modules_import() -> None:
    failures = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the point of the test
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
    assert not failures, "modules failed to import:\n  " + "\n  ".join(failures)


def test_langchain_v1_surface() -> None:
    '''The v1 entry points this codebase depends on.'''

    from langchain.agents import create_agent  # noqa: F401
    from langchain.agents.middleware import AgentState, wrap_model_call  # noqa: F401
    from langchain_classic.retrievers.multi_vector import MultiVectorRetriever  # noqa: F401
    from langchain_classic.storage import LocalFileStore  # noqa: F401
    from langchain_core.documents import Document  # noqa: F401
    from langgraph.prebuilt import InjectedState  # noqa: F401
    from langgraph.types import Command, interrupt  # noqa: F401


def test_tool_inventory_is_complete() -> None:
    '''Every tool the agents had before is still wired to an agent.'''

    from core.agents.agents import DATA_TOOLS, PREDICTION_TOOLS, REPORT_TOOLS, RESEARCH_TOOLS
    from core.agents.handoff import SPECIALISTS, build_handoff_tools
    from core.tools.plan_tools import plan_status, plan_update

    names = {
        tool.name
        for tool in [*RESEARCH_TOOLS, *PREDICTION_TOOLS, *DATA_TOOLS, *REPORT_TOOLS,
                     *build_handoff_tools(), plan_status, plan_update]
    }
    expected = {
        # research
        "literature_search_litsense", "protocol_search_sop", "search_disease_id",
        "create_knowledge_graph", "extract_drugs_from_kg", "extract_proteins_from_kg",
        "extract_pathways_from_kg", "extract_mechanism_of_actions_from_kg",
        "getDrugsforProteins", "getDrugsforPathways", "getDrugsforMechanisms",
        "annotate_chemicals",
        # prediction
        "CYP3A4_classifier", "CYP2C19_classifier", "CYP2D6_classifier", "CYP1A2_classifier",
        "CYP2C9_classifier", "hERG_classifier", "AMES_classifier", "PGP_classifier",
        "PAMPA_classifier", "BBB_classifier", "Solubility_regressor",
        "Lipophilicity_regressor", "predict_repurposedrugs",
        # data and shared
        "python_executor", "reset_python_state", "read_files",
        # orchestration
        "plan_status", "plan_update",
        *(f"transfer_to_{name}" for name in SPECIALISTS),
    }
    missing = expected - names
    assert not missing, f"tools lost in the refactor: {sorted(missing)}"


def test_nothing_still_reaches_for_auth_postgres_or_the_librarian() -> None:
    '''The three removals, checked as text rather than asserted in prose.

    An import that no longer resolves fails `test_all_modules_import` loudly. What
    that test cannot see is a *surviving mention* — a config key nothing sets, an
    `except psycopg.OperationalError` in a branch no test walks, a tool name left in
    a display map. Those are silent, and they are exactly what a removal leaves
    behind, so the source tree is searched for them directly.
    '''

    import re
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    # Whole words, so `authorized_imports` and `author` are not false positives.
    forbidden = re.compile(
        r"\b(psycopg|asyncpg|AsyncPostgresSaver|DATABASE_URL|AUTH_JWT_SECRET|AUTH_PEPPER"
        r"|SENDGRID_API_KEY|librarian_search|literature_search_librarian|LIBRARIAN_\w+)\b"
        r"|langgraph\.checkpoint\.postgres"
        r"|\bbackend\.auth\b|\bapp\.auth_routes\b|\bapp\.demo_threads\b|\bbackend\.db\.pool\b"
    )
    searched = ["app", "backend", "core", "persistence/memory", "tests", "main.py", "requirements.txt"]

    offences = []
    for entry in searched:
        target = repo_root / entry
        paths = [target] if target.is_file() else sorted(target.rglob("*.py")) + sorted(target.rglob("*.sql"))
        for path in paths:
            if "__pycache__" in path.parts or path.name == Path(__file__).name:
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                match = forbidden.search(line)
                if match:
                    offences.append(f"{path.relative_to(repo_root)}:{number}: {match.group(0)}")
    assert not offences, "removed subsystems still referenced:\n  " + "\n  ".join(offences)


if __name__ == "__main__":
    test_all_modules_import()
    print("all modules import OK")
    test_langchain_v1_surface()
    print("langchain v1 surface OK")
    test_tool_inventory_is_complete()
    print("tool inventory complete OK")
    test_nothing_still_reaches_for_auth_postgres_or_the_librarian()
    print("no auth/postgres/librarian references OK")
