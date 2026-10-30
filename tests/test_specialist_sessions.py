from __future__ import annotations

from typing import List, Optional

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from core.agents.context import build_specialist_input_messages
from core.agents.handoff import build_handoff_tools, render_brief, validate_brief
from core.agents.sessions import SESSION_KEY, mint_session_id

BRIEF_TEXT = (
    "TASK BRIEF for data_agent\n\n"
    "Objective: Clean the uploaded dataset and report the row count\n"
    "Inputs: none\n"
    "Expected output: the cleaned CSV path and its row count"
)

OBJECTIVE = "Inspect the uploaded table and standardize its units"
EXPECTED = "the cleaned CSV path and the row count"


def _transfer_call(call_id: str, agent_name: str = "data_agent") -> dict:
    return {"name": f"transfer_to_{agent_name}", "id": call_id, "args": {}}


def _python_call(call_id: str) -> dict:
    return {"name": "python_executor", "id": call_id, "args": {"code": "df.shape"}}


def _brief(
    call_id: str,
    session_id: Optional[str],
    mode: str = "fresh",
    agent_name: str = "data_agent",
    text: Optional[str] = None,
    objective: str = OBJECTIVE,
) -> ToolMessage:
    extra = {}
    if session_id is not None:
        extra = {SESSION_KEY: {"id": session_id, "role": agent_name, "mode": mode, "objective": objective}}
    return ToolMessage(
        content=text or f"TASK BRIEF for {agent_name}: brief {call_id}",
        name=f"transfer_to_{agent_name}",
        tool_call_id=call_id,
        additional_kwargs=extra,
    )


def _delegation(
    call_id: str,
    session_id: Optional[str],
    mode: str = "fresh",
    *,
    work: List[str] = (),
    report: Optional[str] = None,
    agent_name: str = "data_agent",
    objective: str = OBJECTIVE,
) -> List[BaseMessage]:
    '''A supervisor handoff, its brief, the specialist's tool work, and its report.'''

    messages: List[BaseMessage] = [
        AIMessage(content="", name="supervisor", tool_calls=[_transfer_call(call_id, agent_name)]),
        _brief(call_id, session_id, mode, agent_name, objective=objective),
    ]
    for number, result in enumerate(work, 1):
        work_id = f"{call_id}-w{number}"
        messages.append(AIMessage(content="", name=agent_name, tool_calls=[_python_call(work_id)]))
        messages.append(ToolMessage(content=result, tool_call_id=work_id))
    if report is not None:
        messages.append(AIMessage(content=report, name=agent_name))
    return messages


def _data_tool():
    return build_handoff_tools(["data_agent"])[0]


def _call_handoff(history: List[BaseMessage], call_id: str, **overrides):
    args = dict(objective=OBJECTIVE, inputs="none", expected_output=EXPECTED)
    args.update(overrides)
    return _data_tool().func(tool_call_id=call_id, state={"messages": history}, **args)


def _supervisor_turn(*call_ids: str, agent_name: str = "data_agent") -> AIMessage:
    return AIMessage(
        content="", name="supervisor", tool_calls=[_transfer_call(call_id, agent_name) for call_id in call_ids]
    )


# Fresh input stays exactly as it is today


def test_fresh_input_unchanged():
    # The expected value is written out by hand, not computed by the code under test.
    messages = [
        HumanMessage(content="clean the dataset"),
        AIMessage(content="", name="supervisor", tool_calls=[_transfer_call("t1")]),
        ToolMessage(content=BRIEF_TEXT, name="transfer_to_data_agent", tool_call_id="t1"),
        AIMessage(content="", name="data_agent", tool_calls=[_python_call("c1")]),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]
    expected = [
        HumanMessage(content=BRIEF_TEXT),
        AIMessage(content="", name="data_agent", tool_calls=[_python_call("c1")]),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]

    assert build_specialist_input_messages({"messages": messages}, agent_name="data_agent") == expected


# The handoff tool: choosing, checking and recording the session


def test_fresh_handoff_records_a_new_session_without_changing_the_brief():
    long_objective = "Inspect the uploaded table and standardize its units " * 5
    command = _call_handoff(
        [HumanMessage(content="go"), _supervisor_turn("t1")], "t1", objective=long_objective
    )

    brief = command.update["messages"][-1]
    meta = brief.additional_kwargs[SESSION_KEY]
    assert brief.content == render_brief(
        agent_name="data_agent", objective=long_objective, inputs="none", expected_output=EXPECTED
    )
    assert (meta["id"], meta["role"], meta["mode"]) == ("data-1", "data_agent", "fresh")
    assert len(meta["objective"]) <= 120
    assert meta["objective"].startswith("Inspect the uploaded table")


def test_replayed_handoff_reuses_its_session_id():
    history = [*_delegation("t1", "data-1"), *_delegation("t2", "data-2")]

    assert mint_session_id(history, "data_agent", "t1") == "data-1"
    assert mint_session_id(history, "data_agent", "t2") == "data-2"
    assert mint_session_id(history, "data_agent", "t3") == "data-3"


def test_new_session_id_follows_the_highest_number():
    history = [*_delegation("t1", "data-1"), *_delegation("t3", "data-3")]

    assert mint_session_id(history, "data_agent", "t9") == "data-4"


def test_malformed_metadata_and_other_roles_are_ignored_when_choosing_an_id():
    history = [
        ToolMessage(content="b", name="transfer_to_data_agent", tool_call_id="m1",
                    additional_kwargs={SESSION_KEY: "data-5"}),
        ToolMessage(content="b", name="transfer_to_data_agent", tool_call_id="m2",
                    additional_kwargs={SESSION_KEY: {"id": 7, "role": "data_agent"}}),
        ToolMessage(content="b", name="transfer_to_data_agent", tool_call_id="m3",
                    additional_kwargs={SESSION_KEY: {"id": "data-8"}}),
        *_delegation("r1", "research-4", agent_name="research_agent"),
        *_delegation("t1", "data-x"),
    ]

    assert mint_session_id(history, "data_agent", "new") == "data-1"
    # A research brief answering the same call id is not a replay of this data handoff.
    assert mint_session_id(history, "data_agent", "r1") == "data-1"


def test_refused_handoff_does_not_use_up_an_id():
    refused = _call_handoff([_supervisor_turn("t1")], "t1", objective="too short")
    assert isinstance(refused, str) and refused.startswith("Brief not delivered")

    history = [
        _supervisor_turn("t1"),
        ToolMessage(content=refused, name="transfer_to_data_agent", tool_call_id="t1"),
        _supervisor_turn("t2"),
    ]
    command = _call_handoff(history, "t2")

    assert command.update["messages"][-1].additional_kwargs[SESSION_KEY]["id"] == "data-1"


def test_superseded_handoff_does_not_use_up_an_id():
    history = [HumanMessage(content="go"), _supervisor_turn("t1", "t2")]

    superseded = _call_handoff(history, "t2")
    command = _call_handoff(history, "t1")

    assert isinstance(superseded, str) and superseded.startswith("Not delivered")
    assert command.update["messages"][-1].additional_kwargs[SESSION_KEY]["id"] == "data-1"


def test_continue_records_the_session_it_continues():
    history = [*_delegation("t1", "data-1", report="done"), _supervisor_turn("t2")]

    command = _call_handoff(history, "t2", session_id="data-1")

    meta = command.update["messages"][-1].additional_kwargs[SESSION_KEY]
    assert (meta["id"], meta["role"], meta["mode"]) == ("data-1", "data_agent", "continue")


@pytest.mark.parametrize("session_id", ["data-9", "research-1"])
def test_unknown_or_wrong_role_session_is_refused_with_the_valid_ids(session_id):
    history = [
        *_delegation("r1", "research-1", agent_name="research_agent", report="done"),
        *_delegation("t1", "data-1", report="done"),
        *_delegation("t2", "data-2", report="done"),
        _supervisor_turn("t3"),
    ]

    refused = _call_handoff(history, "t3", session_id=session_id)

    assert isinstance(refused, str)
    assert f"{session_id!r} is not a data_agent session" in refused
    assert "Valid data_agent sessions: data-1, data-2." in refused


def test_vague_reference_wording_depends_on_mode():
    args = dict(agent_name="data_agent", objective=OBJECTIVE, inputs="the file from the previous step",
                expected_output=EXPECTED)

    fresh = validate_brief(**args)
    continuing = validate_brief(**args, continuing=True)

    assert "cannot see any earlier step" in fresh
    assert "cannot see any earlier step" not in continuing
    assert "will see its own earlier work in this session" in continuing
    assert "must still give the actual value" in continuing


def test_continue_keeps_every_brief_check():
    history = [*_delegation("t1", "data-1", report="done"), _supervisor_turn("t2")]

    refused = _call_handoff(
        history, "t2", session_id="data-1", objective="short", inputs="as before", expected_output=""
    )

    assert "`objective` is too short" in refused
    assert "`expected_output` is missing" in refused
    assert "will see its own earlier work in this session" in refused
