from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from core.agents.context import build_specialist_input_messages

BRIEF_TEXT = (
    "TASK BRIEF for data_agent\n\n"
    "Objective: Clean the uploaded dataset and report the row count\n"
    "Inputs: none\n"
    "Expected output: the cleaned CSV path and its row count"
)

def _transfer_call(call_id: str, agent_name: str = "data_agent") -> dict:
    return {"name": f"transfer_to_{agent_name}", "id": call_id, "args": {}}


def _python_call(call_id: str) -> dict:
    return {"name": "python_executor", "id": call_id, "args": {"code": "df.shape"}}


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
