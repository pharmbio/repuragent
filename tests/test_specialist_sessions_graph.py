'''The real graph with scripted models: sessions survive a reload and compaction.'''

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import core.agents.agentic_system as asys
import core.agents.context as context
from app.langgraph_runner import build_stream_input
from core.agents.sessions import session_meta
from tests.fakes import ScriptedChatModel, scripted, tool_call

ROLES = ("supervisor", "research", "prediction", "data", "planning", "report")

# What the supervisor delegates in each turn: (table marker, session_id to continue or None).
Turns = Sequence[Sequence[Tuple[str, Optional[str]]]]


def _models(turns: Turns) -> Dict[str, ScriptedChatModel]:
    '''Scripted models: the supervisor delegates each step, the data agent runs one tool and reports.'''

    supervisor: List[AIMessage] = []
    data: List[AIMessage] = []
    for steps in turns:
        for marker, session_id in steps:
            args = {
                "objective": f"Profile table {marker} and report its row count",
                "inputs": f"table {marker}",
                "expected_output": "One line with the table name and its row count",
            }
            if session_id:
                args["session_id"] = session_id
            supervisor.append(AIMessage(content="", tool_calls=[tool_call("transfer_to_data_agent", args, f"call_{marker}")]))
            data.append(AIMessage(content="", tool_calls=[tool_call("reset_python_state", {}, f"work_{marker}")]))
            data.append(AIMessage(content=f"data report: finished {marker}"))
        supervisor.append(AIMessage(content="supervisor: done"))
    models = {role: scripted(label=role) for role in ROLES}
    models["supervisor"] = scripted(*supervisor, label="supervisor")
    models["data"] = scripted(*data, label="data")
    return models


def _data_inputs(models: Dict[str, ScriptedChatModel]) -> List[List[str]]:
    '''What the data agent was given at the start of each delegation (it makes two calls per delegation).'''

    return [[str(m.content) for m in call if not isinstance(m, SystemMessage)] for call in models["data"].calls[::2]]


async def _run_turn(app, config, turn: int) -> None:
    stream_input = build_stream_input(
        f"turn {turn}", conversation_id=config["configurable"]["thread_id"], use_episodic_learning=False
    )
    async for _ in app.astream(stream_input, config=config, stream_mode=["messages", "updates"], subgraphs=True):
        pass


async def _run(db: str, models, turns_before_reload: List[int], turns_after_reload: List[int], compression: bool):
    config = {"configurable": {"thread_id": "conv-sessions"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = asys.build_graph(saver, models=models, use_context_compression=compression)
        for turn in turns_before_reload:
            await _run_turn(app, config, turn)
    # A new connection and a newly compiled graph: everything must come from the checkpoint.
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = asys.build_graph(saver, models=models, use_context_compression=compression)
        for turn in turns_after_reload:
            await _run_turn(app, config, turn)
        return (await app.aget_state(config)).values["messages"]


def _run_offline(models, turns_before_reload, turns_after_reload, compression, summaries=()):
    '''Run the graph against a temporary checkpoint with the helper models replaced, then restore them.'''

    summaries = iter(summaries)
    asys.reset_model_cache()
    asys.install_helper_models(
        task_classifier=RunnableLambda(lambda _: asys.TaskClassification(category="simple")),
        context_summarizer=RunnableLambda(lambda _: asys.ContextDigest(summary=next(summaries))),
    )
    try:
        with tempfile.TemporaryDirectory() as workdir:
            db = str(Path(workdir) / "cp.db")
            return asyncio.run(_run(db, models, turns_before_reload, turns_after_reload, compression))
    finally:
        asys.reset_model_cache()


def _sessions(messages):
    return [(meta["id"], meta["mode"]) for meta in map(session_meta, messages) if meta]


def test_continued_session_after_reload_sees_only_its_own_work():
    models = _models([[("ALPHA", None), ("BETA", None)], [("GAMMA", "data-1")]])

    messages = _run_offline(models, [1], [2], compression=False)

    assert _sessions(messages) == [("data-1", "fresh"), ("data-2", "fresh"), ("data-1", "continue")]
    fresh_alpha, fresh_beta, continued = _data_inputs(models)
    assert len(fresh_alpha) == 1 and "Profile table ALPHA" in fresh_alpha[0]
    assert len(fresh_beta) == 1 and "Profile table BETA" in fresh_beta[0]

    seen = "\n".join(continued)
    assert "Profile table ALPHA" in seen and "data report: finished ALPHA" in seen
    assert "BETA" not in seen
    assert continued[-1].startswith("TASK BRIEF") and "Profile table GAMMA" in continued[-1]
    assert seen.count("Profile table GAMMA") == 1


def test_session_list_and_continuation_survive_compaction():
    models = _models([[("ALPHA", None), ("BETA", None)], [("DELTA", None)], [("GAMMA", "data-1")]])
    triggers = (context.SUMMARY_TRIGGER_MIN_MESSAGES_FIRST, context.SUMMARY_TRIGGER_MIN_MESSAGES)
    context.SUMMARY_TRIGGER_MIN_MESSAGES_FIRST = context.SUMMARY_TRIGGER_MIN_MESSAGES = 1
    try:
        messages = _run_offline(models, [1, 2], [3], compression=True,
                                summaries=["first summary", "second summary", "third summary"])
    finally:
        context.SUMMARY_TRIGGER_MIN_MESSAGES_FIRST, context.SUMMARY_TRIGGER_MIN_MESSAGES = triggers

    summaries_written = [m for m in messages if (m.response_metadata or {}).get("is_summary")]
    assert len(summaries_written) >= 2
    assert _sessions(messages) == [
        ("data-1", "fresh"), ("data-2", "fresh"), ("data-3", "fresh"), ("data-1", "continue"),
    ]

    # The supervisor's first call in turn 3 comes after both summaries.
    turn_three_prompt = str(models["supervisor"].calls[-2][0].content)
    for session_id in ("data-1", "data-2", "data-3"):
        assert f"- {session_id} · data_agent ·" in turn_three_prompt

    seen = "\n".join(_data_inputs(models)[-1])
    assert "data report: finished ALPHA" in seen
    assert "BETA" not in seen and "DELTA" not in seen
