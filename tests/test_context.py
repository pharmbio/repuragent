'''Context assembly: turn boundaries, verbatim anchors, tool-traffic pruning.

These are the properties a follow-up depends on. "Now rank those by hERG risk" only
works if the previous request and the answer it produced survive compression intact,
and if the turn boundary between them is found correctly despite plan review writing
its own HumanMessages into the same transcript.
'''

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from tests.harness import scoped_results_root


def _user(text: str, *, marked: bool = True) -> HumanMessage:
    from core.agents.context import mark_user_request

    message = HumanMessage(content=text)
    return mark_user_request(message) if marked else message


def _report(text: str, name: str = "report_agent_complex") -> AIMessage:
    return AIMessage(content=text, name=name)


def test_marked_requests_define_turns() -> None:
    from core.agents.context import (
        has_completed_turn,
        latest_user_request_text,
    )

    messages = [
        _user("Find candidates for AML"),
        AIMessage(content="a plan", name="planning_agent"),
        # Plan feedback is a HumanMessage too, and must not open a turn.
        HumanMessage(content="approved"),
        _report("# Response Summary\n## Answer\nTwenty candidates."),
        _user("Now rank those by hERG risk"),
    ]
    assert latest_user_request_text(messages) == "Now rank those by hERG risk"
    assert has_completed_turn(messages) is True

    first_only = messages[:4]
    assert has_completed_turn(first_only) is False


def test_unmarked_history_still_splits_correctly() -> None:
    '''Conversations checkpointed before the marker existed must still work.'''

    from core.agents.context import latest_user_request_text

    messages = [
        _user("Find candidates for AML", marked=False),
        AIMessage(content="a plan", name="planning_agent"),
        HumanMessage(content="approved"),  # follows the planner: plan feedback
        _report("done"),
        _user("Add lipophilicity", marked=False),
    ]
    assert latest_user_request_text(messages) == "Add lipophilicity"


def test_completed_turns_survive_compression_verbatim() -> None:
    from core.agents.context import build_turn_anchors

    messages = [
        _user("Find candidates for AML"),
        AIMessage(content="delegating", name="supervisor"),
        ToolMessage(content="412 drugs", name="extract_drugs_from_kg", tool_call_id="t1"),
        _report("# Response Summary\n## Answer\nTwenty candidates, ranked."),
        _user("Now rank those by hERG risk"),
    ]
    anchors = build_turn_anchors(messages, upto_index=4, keep=3)
    texts = [message.content for message in anchors]
    # The request and the answer it produced, and none of the delegation chatter.
    assert texts[0] == "Find candidates for AML"
    assert "Twenty candidates, ranked." in texts[1]
    assert not any("412 drugs" in text for text in texts)


def test_an_unanswered_turn_is_labelled_as_such() -> None:
    from core.agents.context import build_turn_anchors

    messages = [_user("Find candidates for AML"), AIMessage(content="delegating", name="supervisor")]
    anchors = build_turn_anchors(messages, upto_index=2, keep=3)
    assert anchors[-1].content == "(That request did not reach a final answer.)"


def test_tool_results_are_shortened_never_dropped() -> None:
    '''Dropping a ToolMessage would orphan its tool_call and fail history validation.'''

    from app.config import TOOL_RESULT_ELIDED_CHARS
    from core.agents.context import prune_tool_traffic

    messages = [_user("go")]
    for index in range(10):
        messages.append(
            AIMessage(
                content="",
                name="research_agent",
                tool_calls=[{"name": "getDrugsforProteins", "args": {}, "id": f"c{index}"}],
            )
        )
        messages.append(
            ToolMessage(content="X" * 60000, name="getDrugsforProteins", tool_call_id=f"c{index}")
        )

    pruned = prune_tool_traffic(messages)
    assert len(pruned) == len(messages)
    tool_messages = [message for message in pruned if isinstance(message, ToolMessage)]
    assert len(tool_messages) == 10
    # The oldest collapse to a stub; the newest keep enough to work with.
    assert len(tool_messages[0].content) < TOOL_RESULT_ELIDED_CHARS + 400
    assert "trimmed to keep the working context small" in tool_messages[0].content
    assert len(tool_messages[-1].content) > TOOL_RESULT_ELIDED_CHARS * 5
    # Every tool_call still has its answer, in order.
    call_ids = [
        call["id"]
        for message in pruned
        if isinstance(message, AIMessage)
        for call in (message.tool_calls or [])
    ]
    assert call_ids == [message.tool_call_id for message in tool_messages]


def test_summary_messages_are_recognised_and_excluded() -> None:
    from core.agents.context import (
        build_llm_input_messages,
        is_summary_message,
        latest_summary_record,
        make_summary_message,
        messages_since_last_summary,
    )

    summary = make_summary_message("earlier work", {"facts": ["MONDO:0018874 is AML"]})
    assert is_summary_message(summary)

    messages = [_user("first"), _report("first answer"), summary, _user("second")]
    index, text, memory = latest_summary_record(messages)
    assert index == 2 and text == "earlier work"
    assert memory["facts"] == ["MONDO:0018874 is AML"]
    assert [m.content for m in messages_since_last_summary(messages)] == ["second"]

    rendered = build_llm_input_messages({"messages": messages}, compress=True)
    # The summary is delivered as a system message, not as a bare AI turn, and the
    # structured memory comes with it.
    system_text = "\n".join(m.content for m in rendered if isinstance(m, SystemMessage))
    assert "earlier work" in system_text
    assert "MONDO:0018874 is AML" in system_text
    assert not any(is_summary_message(m) for m in rendered)

    uncompressed = build_llm_input_messages({"messages": messages}, compress=False)
    assert not any(is_summary_message(m) for m in uncompressed)


def test_structured_memory_merges_without_duplicating() -> None:
    from core.agents.context import normalize_memory

    prior = {
        "facts": ["hERG risk is high for 3 candidates"],
        "outputs": [{"path": "/x/master_ranked.csv", "description": "ranked candidates"}],
        "decisions": [],
        "open_questions": [],
    }
    merged = normalize_memory(
        {
            "facts": ["MONDO:0018874 is AML", "hERG risk is high for 3 candidates"],
            "outputs": [{"path": "/x/master_ranked.csv", "description": "ranked candidates"}],
            "decisions": ["Excluded drugs already indicated for AML"],
        },
        prior,
    )
    assert merged["facts"].count("hERG risk is high for 3 candidates") == 1
    assert len(merged["outputs"]) == 1
    assert merged["decisions"] == ["Excluded drugs already indicated for AML"]


def test_pinned_block_carries_scope_plan_and_constraints() -> None:
    from core.agents.context import build_pinned_context_block

    with scoped_results_root():
        state = {
            "conversation_id": "conv",
            "plan_path": "/tmp/u1/conv/plan.md",
            "plan_progress": "Run 1: 1/4 steps resolved · next: step 2 (Extract candidates)",
            "approval_constraints": ["Only clinical phase 3 or later"],
        }
        messages = [_user("Find candidates for AML")]
        block = build_pinned_context_block(state, messages)

        assert "Active output scope" in block
        assert "conv" in block
        assert "Find candidates for AML" in block  # the standing goal
        assert "/tmp/u1/conv/plan.md" in block
        assert "next: step 2" in block
        assert "Only clinical phase 3 or later" in block
        # The plan *text* must not be pinned; only the pointer and the progress line.
        assert "TASK BREAKDOWN" not in block

        without_plan = build_pinned_context_block(state, messages, include_plan=False)
        assert "plan.md" not in without_plan


def test_prior_context_describes_the_last_exchange_for_routing() -> None:
    from core.agents.context import describe_prior_context

    messages = [
        _user("Find candidates for AML"),
        _report("# Response Summary\n## Answer\nTwenty candidates."),
        _user("Now rank those by hERG risk"),
    ]
    described = describe_prior_context(messages)
    assert "Conversation goal" in described
    assert "Twenty candidates." in described
    # Only one turn: nothing to describe, so the classifier is told it is the first.
    assert describe_prior_context(messages[:1]) == ""


def test_transcript_render_is_flat_and_labelled() -> None:
    from core.agents.context import render_transcript

    text = render_transcript(
        [
            _user("Find candidates"),
            AIMessage(
                content="delegating",
                name="supervisor",
                tool_calls=[{"name": "transfer_to_research_agent", "args": {}, "id": "c1"}],
            ),
            ToolMessage(content="412 drugs", name="extract_drugs_from_kg", tool_call_id="c1"),
        ]
    )
    assert "### USER" in text
    assert "### ASSISTANT (supervisor)" in text
    assert "[called tools: transfer_to_research_agent]" in text
    assert "### TOOL RESULT (extract_drugs_from_kg)" in text


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
