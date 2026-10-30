from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import List, Optional

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

import core.agents.context as context
from core.agents.context import (
    ISOLATION_NOTICE,
    build_context_middleware,
    build_specialist_input_messages,
    make_summary_message,
)
import core.agents.sessions as sessions
from core.agents.handoff import build_handoff_tools, render_brief, validate_brief
from core.agents.sessions import (
    SESSION_KEY,
    SESSION_LIST_MAX_CHARS,
    mint_session_id,
    render_session_list,
)

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


def _text(messages: List[BaseMessage]) -> str:
    return "\n".join(str(message.content) for message in messages)


def _specialist_view(messages: List[BaseMessage], agent_name: str = "data_agent") -> List[BaseMessage]:
    return build_specialist_input_messages({"messages": messages}, agent_name=agent_name)


def _assert_pairs_intact(messages: List[BaseMessage]) -> None:
    declared = set()
    answered = set()
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            declared.add(call["id"])
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in declared
            answered.add(message.tool_call_id)
    assert declared == answered


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


def test_fresh_brief_with_session_metadata_gives_todays_input():
    messages = [
        HumanMessage(content="turn 1"),
        *_delegation("t1", "data-1", work=["mixed units in column A"], report="data-1 done"),
        HumanMessage(content="turn 2"),
        AIMessage(content="", name="supervisor", tool_calls=[_transfer_call("t2")]),
        _brief("t2", "data-2", text=BRIEF_TEXT),
        AIMessage(content="", name="data_agent", tool_calls=[_python_call("c1")]),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]
    expected = [
        HumanMessage(content=BRIEF_TEXT),
        AIMessage(content="", name="data_agent", tool_calls=[_python_call("c1")]),
        ToolMessage(content="ok", tool_call_id="c1"),
    ]

    assert _specialist_view(messages) == expected


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


# Continued history: only the session's own earlier work, bounded


def test_continue_selects_only_its_own_session():
    messages = [
        HumanMessage(content="turn 1"),
        *_delegation("t1", "data-1", work=["mixed units in column A"], report="data-1 report"),
        *_delegation("t2", "data-2", work=["independent check complete"], report="data-2 report"),
        AIMessage(content="supervisor wrap-up", name="supervisor"),
        HumanMessage(content="turn 2"),
        *_delegation("t3", "data-1", "continue", work=["current step output"]),
    ]

    view = _specialist_view(messages)
    text = _text(view)

    assert "mixed units in column A" in text
    assert "data-1 report" in text
    assert "independent check complete" not in text
    assert "data-2 report" not in text
    assert "supervisor wrap-up" not in text and "turn 2" not in text
    assert text.count("brief t3") == 1
    assert text.count("current step output") == 1
    assert [str(m.content) for m in view if isinstance(m, HumanMessage)] == [
        "TASK BRIEF for data_agent: brief t1",
        "TASK BRIEF for data_agent: brief t3",
    ]
    _assert_pairs_intact(view)


def test_continue_includes_earlier_continuations_of_the_same_session():
    messages = [
        *_delegation("t1", "data-1", work=["first part"], report="r1"),
        *_delegation("t2", "data-1", "continue", work=["second part"], report="r2"),
        *_delegation("t3", "data-1", "continue"),
    ]

    text = _text(_specialist_view(messages))

    assert text.index("first part") < text.index("second part") < text.index("brief t3")


@pytest.mark.parametrize(
    "terminator",
    [
        AIMessage(content="TERMINATOR", name="supervisor"),
        HumanMessage(content="TERMINATOR"),
        AIMessage(content="TERMINATOR", name="plan_init"),
        AIMessage(content="TERMINATOR", name="plan_finalize"),
        AIMessage(content="TERMINATOR", name="report_agent"),
        make_summary_message("TERMINATOR", {}),
        AIMessage(content="TERMINATOR", name="research_agent"),
        ToolMessage(content="TERMINATOR", name="transfer_to_data_agent", tool_call_id="other"),
        ToolMessage(content="TERMINATOR", tool_call_id="not-declared-by-data-agent"),
    ],
    ids=["supervisor", "human", "plan_init", "plan_finalize", "report", "summary",
         "other_agent", "next_brief", "foreign_tool_result"],
)
def test_earlier_span_ends_at_the_first_message_that_is_not_the_specialists(terminator):
    messages = [
        *_delegation("t1", "data-1", work=["work one"]),
        terminator,
        AIMessage(content="LEAKED", name="data_agent"),
        *_delegation("t2", "data-1", "continue"),
    ]

    text = _text(_specialist_view(messages))

    assert "work one" in text
    assert "TERMINATOR" not in text
    assert "LEAKED" not in text


def test_only_the_most_recent_earlier_spans_are_kept(monkeypatch):
    monkeypatch.setattr(context, "SPECIALIST_SESSION_MAX_SPANS", 2)
    messages = []
    for number in range(1, 6):
        messages += _delegation(f"t{number}", "data-1", "fresh" if number == 1 else "continue",
                                work=[f"work-{number}"], report=f"report-{number}")
    messages += _delegation("t6", "data-1", "continue")

    view = _specialist_view(messages)
    text = _text(view)

    assert isinstance(view[0], HumanMessage)
    assert view[0].content == "(Session data-1: 3 earlier part(s) omitted, to keep the context short.)"
    assert all(f"work-{number}" not in text for number in (1, 2, 3))
    assert "work-4" in text and "work-5" in text
    _assert_pairs_intact(view)


def test_character_budget_drops_the_oldest_whole_spans(monkeypatch):
    monkeypatch.setattr(context, "SPECIALIST_SESSION_HISTORY_MAX_CHARS", 700)
    messages = []
    for number in range(1, 4):
        messages += _delegation(f"t{number}", "data-1", "fresh" if number == 1 else "continue",
                                work=[f"work-{number} " + "x" * 300], report=f"report-{number}")
    messages += _delegation("t4", "data-1", "continue", work=["current"])

    view = _specialist_view(messages)
    text = _text(view)

    assert view[0].content == "(Session data-1: 2 earlier part(s) omitted, to keep the context short.)"
    assert "work-1" not in text and "work-2" not in text
    assert "work-3" in text and "report-3" in text
    assert text.count("current") == 1
    _assert_pairs_intact(view)


def test_single_oversized_span_keeps_only_its_brief_and_final_report(monkeypatch):
    monkeypatch.setattr(context, "SPECIALIST_SESSION_HISTORY_MAX_CHARS", 300)
    messages = [
        *_delegation("t1", "data-1", work=["huge " + "y" * 1000], report="final report one"),
        *_delegation("t2", "data-1", "continue", work=["current"]),
    ]

    view = _specialist_view(messages)

    assert [type(message).__name__ for message in view] == [
        "HumanMessage", "HumanMessage", "AIMessage", "HumanMessage", "AIMessage", "ToolMessage",
    ]
    assert view[0].content == (
        "(Session data-1: the working steps of the oldest part shown are omitted; its brief "
        "and final report are kept, to keep the context short.)"
    )
    assert view[1].content == "TASK BRIEF for data_agent: brief t1"
    assert view[2].content == "final report one"
    assert "huge" not in _text(view)
    _assert_pairs_intact(view)


# The continuation note in the specialist's system prompt


class _Request(SimpleNamespace):
    def override(self, **changes):
        return _Request(**{**vars(self), **changes})


def _system_prompt(middleware, messages: List[BaseMessage]) -> str:
    request = _Request(state={"messages": messages}, system_message=SystemMessage(content="BASE"))

    async def handler(final):
        return final

    return asyncio.run(middleware.awrap_model_call(request, handler)).system_message.content


def _specialist_middleware():
    return build_context_middleware(
        role="data_agent", isolated=True, include_goal=False, include_plan=False, include_artifacts=True
    )


def test_continuation_note_follows_the_isolation_notice():
    continued = [*_delegation("t1", "data-1", report="done"), *_delegation("t2", "data-1", "continue")]

    prompt = _system_prompt(_specialist_middleware(), continued)

    assert ISOLATION_NOTICE + "\n\nYou are continuing session data-1." in prompt


def test_fresh_specialist_prompt_has_no_continuation_note():
    middleware = _specialist_middleware()
    prompt = _system_prompt(middleware, [*_delegation("t1", "data-1")])

    assert ISOLATION_NOTICE in prompt
    assert "continuing session" not in prompt
    # Byte-identical to the prompt for a legacy brief, which predates sessions.
    assert prompt == _system_prompt(middleware, [*_delegation("t1", None)])


# The supervisor's session list


def _session_lines(block: str) -> List[str]:
    return [line for line in block.splitlines() if line.startswith("- ") and not line.startswith("- older:")]


def test_session_list_is_ordered_by_most_recent_use():
    messages = [
        *_delegation("t1", "data-1", report="data report text"),
        *_delegation("r1", "research-1", agent_name="research_agent", report="done"),
        *_delegation("t2", "data-2", report="done"),
        *_delegation("t3", "data-1", "continue", report="done"),
    ]

    block = render_session_list(messages)

    assert [line.split(" · ")[0] for line in _session_lines(block)] == ["- data-1", "- data-2", "- research-1"]
    assert f"- research-1 · research_agent · {OBJECTIVE}" in block
    assert "data report text" not in block


def test_session_list_details_the_newest_sessions_and_lists_older_ids(monkeypatch):
    monkeypatch.setattr(sessions, "SPECIALIST_SESSION_LIST_MAX_ITEMS", 4)
    messages = []
    for number in range(1, 7):
        messages += _delegation(f"t{number}", f"data-{number}")

    block = render_session_list(messages)

    assert len(_session_lines(block)) == 4
    assert _session_lines(block)[0].startswith("- data-6 ·")
    assert block.splitlines()[-1] == "- older: data-2, data-1"


def test_session_list_stays_within_its_total_and_counts_what_does_not_fit():
    messages = []
    for number in range(1, 401):
        messages += _delegation(f"t{number}", f"data-{number}", objective="o" * 300)

    block = render_session_list(messages)
    older = block.splitlines()[-1]
    shown = older.removeprefix("- older: ").split(" +")[0].split(", ")
    more = int(older.rsplit("+", 1)[1].removesuffix(" more"))

    assert len(block) <= SESSION_LIST_MAX_CHARS
    assert all(len(line.split(" · ")[2]) <= 120 for line in _session_lines(block))
    assert len(_session_lines(block)) + len(shown) + more == 400


def test_session_list_is_empty_without_sessions():
    assert render_session_list([*_delegation("t1", None), HumanMessage(content="hi")]) == ""


def test_only_the_supervisor_gets_the_session_list():
    messages = [HumanMessage(content="go"), *_delegation("t1", "data-1", report="done")]
    supervisor = build_context_middleware(role="supervisor", collapse_delegations=True)
    report = build_context_middleware(role="report_agent")
    planner = build_context_middleware(role="planning_agent", include_plan=False, include_episodes=False)

    assert "data-1 · data_agent" in _system_prompt(supervisor, messages)
    assert "data-1" not in _system_prompt(report, messages)
    assert "data-1" not in _system_prompt(planner, messages)
    assert "data-1" not in _system_prompt(_specialist_middleware(), messages)
