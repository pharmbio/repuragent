'''The real graph with scripted models: sessions survive a reload and compaction.'''

from __future__ import annotations

import asyncio
import re
import uuid
from typing import Dict, List

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

import core.agents.agentic_system as asys
import core.agents.context as context
from app.langgraph_runner import build_stream_input
from core.agents.sessions import session_meta

ROLES = ("supervisor", "research", "prediction", "data", "planning", "report")


class Script:
    '''What the supervisor delegates in each turn, and what every model received.'''

    def __init__(self, turns: Dict[int, List[Dict[str, str]]]):
        self.turns = turns
        self.turn = 0
        self.done = 0
        self.data_inputs: List[List[str]] = []
        self.supervisor_prompts: List[str] = []


class ScriptModel(BaseChatModel):
    role: str
    script: Script

    @property
    def _llm_type(self) -> str:
        return "script"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, *args, **kwargs):
        raise NotImplementedError

    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._reply(messages))])

    def _reply(self, messages) -> AIMessage:
        script = self.script
        if self.role == "supervisor":
            script.supervisor_prompts.append(str(messages[0].content))
            todo = script.turns[script.turn]
            if script.done >= len(todo):
                return AIMessage(content="supervisor: done")
            step = todo[script.done]
            args = {
                "objective": f"Profile table {step['marker']} and report its row count",
                "inputs": f"table {step['marker']}",
                "expected_output": "One line with the table name and its row count",
            }
            if step.get("session_id"):
                args["session_id"] = step["session_id"]
            call = {"name": "transfer_to_data_agent", "id": f"call_{uuid.uuid4().hex[:8]}", "args": args}
            return AIMessage(content="", tool_calls=[call])

        if self.role == "data":
            if isinstance(messages[-1], ToolMessage):
                script.done += 1
                brief = [m for m in messages if isinstance(m, HumanMessage)][-1]
                marker = re.search(r"Profile table (\w+)", str(brief.content)).group(1)
                return AIMessage(content=f"data report: finished {marker}")
            script.data_inputs.append(
                [str(m.content) for m in messages if not isinstance(m, SystemMessage)]
            )
            call = {"name": "reset_python_state", "args": {}, "id": f"w_{uuid.uuid4().hex[:8]}"}
            return AIMessage(content="", tool_calls=[call])

        return AIMessage(content=f"{self.role}: final")


async def _run_turn(app, config, script: Script, turn: int) -> None:
    script.turn, script.done = turn, 0
    stream_input = build_stream_input(
        f"turn {turn}", conversation_id=config["configurable"]["thread_id"], use_episodic_learning=False
    )
    async for _ in app.astream(stream_input, config=config, stream_mode=["messages", "updates"], subgraphs=True):
        pass


async def _run(db: str, script: Script, turns_before_reload: List[int], turns_after_reload: List[int],
               compression: bool):
    models = {role: ScriptModel(role=role, script=script) for role in ROLES}
    config = {"configurable": {"thread_id": f"conv-{uuid.uuid4().hex[:6]}"}, "recursion_limit": 100}
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = asys.build_graph(saver, models=models, use_context_compression=compression)
        for turn in turns_before_reload:
            await _run_turn(app, config, script, turn)
    # A new connection and a newly compiled graph: everything must come from the checkpoint.
    async with AsyncSqliteSaver.from_conn_string(db) as saver:
        app = asys.build_graph(saver, models=models, use_context_compression=compression)
        for turn in turns_after_reload:
            await _run_turn(app, config, script, turn)
        return (await app.aget_state(config)).values["messages"]


@pytest.fixture(autouse=True)
def _offline_helpers(monkeypatch):
    monkeypatch.setattr(
        asys, "_task_classifier", RunnableLambda(lambda _: asys.TaskClassification(category="simple"))
    )


def _sessions(messages):
    return [(meta["id"], meta["mode"]) for meta in map(session_meta, messages) if meta]


def test_continued_session_after_reload_sees_only_its_own_work(tmp_path):
    script = Script({
        1: [{"marker": "ALPHA"}, {"marker": "BETA"}],
        2: [{"marker": "GAMMA", "session_id": "data-1"}],
    })

    messages = asyncio.run(_run(str(tmp_path / "cp.db"), script, [1], [2], compression=False))

    assert _sessions(messages) == [("data-1", "fresh"), ("data-2", "fresh"), ("data-1", "continue")]
    fresh_alpha, fresh_beta, continued = script.data_inputs
    assert len(fresh_alpha) == 1 and "Profile table ALPHA" in fresh_alpha[0]
    assert len(fresh_beta) == 1 and "Profile table BETA" in fresh_beta[0]

    seen = "\n".join(continued)
    assert "Profile table ALPHA" in seen and "data report: finished ALPHA" in seen
    assert "BETA" not in seen
    assert continued[-1].startswith("TASK BRIEF") and "Profile table GAMMA" in continued[-1]
    assert seen.count("Profile table GAMMA") == 1


def test_session_list_and_continuation_survive_compaction(tmp_path, monkeypatch):
    summaries = iter(["first summary", "second summary"])
    monkeypatch.setattr(
        asys, "_context_summarizer",
        RunnableLambda(lambda _: asys.ContextDigest(summary=next(summaries))),
    )
    monkeypatch.setattr(context, "SUMMARY_TRIGGER_MIN_MESSAGES_FIRST", 1)
    monkeypatch.setattr(context, "SUMMARY_TRIGGER_MIN_MESSAGES", 1)
    script = Script({
        1: [{"marker": "ALPHA"}, {"marker": "BETA"}],
        2: [{"marker": "DELTA"}],
        3: [{"marker": "GAMMA", "session_id": "data-1"}],
    })

    messages = asyncio.run(_run(str(tmp_path / "cp.db"), script, [1, 2], [3], compression=True))

    summaries_written = [m for m in messages if (m.response_metadata or {}).get("is_summary")]
    assert len(summaries_written) >= 2
    assert _sessions(messages) == [
        ("data-1", "fresh"), ("data-2", "fresh"), ("data-3", "fresh"), ("data-1", "continue"),
    ]

    # The supervisor's first call in turn 3 comes after both summaries.
    turn_three_prompt = script.supervisor_prompts[-2]
    for session_id in ("data-1", "data-2", "data-3"):
        assert f"- {session_id} · data_agent ·" in turn_three_prompt

    seen = "\n".join(script.data_inputs[-1])
    assert "data report: finished ALPHA" in seen
    assert "BETA" not in seen and "DELTA" not in seen
