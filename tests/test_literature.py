'''The literature tool: what it renders, and how it fails.

Nothing here reaches LitSense for real — it is an NCBI endpoint, and a suite that
needs the network is a suite that fails for reasons that have nothing to do with the
code. What is worth pinning offline is everything around the call. The rendering,
because the agent quotes from it and a passage without its PMID is evidence that
cannot be cited; and the failure path, because a search that raises out of the tool
aborts the run, while a search that *reports* a failure lets the supervisor try
something else.

The web build has a second literature tool here — the vendored EMBL AI Librarian.
It is deliberately absent from this build, and `test_imports` fails if a mention of
it survives anywhere.
'''

from __future__ import annotations

from typing import Any, List

from core.tools.research_tools import LitSenseObject


def _passage(**overrides: Any) -> LitSenseObject:
    '''One result shaped exactly as the LitSense API returns it.'''

    fields = {
        "text": "Metformin use was associated with a lower cancer incidence.",
        "score": 0.91,
        "annotations": [],
        "pmid": 31234567,
        "pmcid": "PMC6543210",
        "section": "RESULTS",
    }
    fields.update(overrides)
    return LitSenseObject(**fields)


class _ScriptedLitSense:
    '''A `PyLitSense` stand-in: records the call, then returns or raises.'''

    last: "_ScriptedLitSense | None" = None

    def __init__(self, results: List[LitSenseObject] | None = None, error: Exception | None = None) -> None:
        self.results = results if results is not None else [_passage()]
        self.error = error
        self.query_str: str | None = None
        self.limit: int | None = None

    def __call__(self) -> "_ScriptedLitSense":
        # `literature_search_litsense` constructs the engine itself, so the class
        # is what gets swapped and calling it hands back this same instance.
        _ScriptedLitSense.last = self
        return self

    def query(self, query_str: str, **kwargs: Any) -> List[LitSenseObject]:
        self.query_str = query_str
        self.limit = kwargs.get("limit")
        if self.error is not None:
            raise self.error
        return self.results


def _with_engine(engine: _ScriptedLitSense):
    '''Swap `PyLitSense` for `engine`, and hand back the original.'''

    from core.tools import research_tools

    original = research_tools.PyLitSense
    research_tools.PyLitSense = engine
    return research_tools, original


def test_a_passage_carries_everything_needed_to_cite_it() -> None:
    from core.tools.research_tools import literature_search_litsense

    engine = _ScriptedLitSense([_passage(), _passage(pmid=42, text="A second finding.")])
    module, original = _with_engine(engine)
    try:
        result = literature_search_litsense.invoke({"query": "metformin and cancer", "limit": 2})
    finally:
        module.PyLitSense = original

    assert "PMID: 31234567" in result
    assert "Metformin use was associated with a lower cancer incidence." in result
    # Numbered, because the agent refers back to a passage by its position.
    assert "Passage #1" in result and "Passage #2" in result


def test_the_limit_and_query_reach_the_api() -> None:
    from core.tools.research_tools import literature_search_litsense

    engine = _ScriptedLitSense()
    module, original = _with_engine(engine)
    try:
        literature_search_litsense.invoke({"query": "hERG liability of azoles", "limit": 7})
    finally:
        module.PyLitSense = original

    assert engine.query_str == "hERG liability of azoles"
    assert engine.limit == 7


def test_an_empty_search_reads_as_a_result_not_a_failure() -> None:
    from core.tools.research_tools import literature_search_litsense

    engine = _ScriptedLitSense(results=[])
    module, original = _with_engine(engine)
    try:
        result = literature_search_litsense.invoke({"query": "phlogiston in mitochondria"})
    finally:
        module.PyLitSense = original

    assert "No relevant literature found" in result
    # "Error" would have the supervisor treat a genuine absence as a broken tool.
    assert "Error" not in result


def test_a_transport_failure_is_reported_not_raised() -> None:
    '''A raised exception aborts the whole run; a returned message does not.

    LitSense is a public NCBI endpoint and does go down. The supervisor can route
    around a tool that says it failed, and cannot route around a traceback.
    '''

    from core.tools.research_tools import literature_search_litsense

    engine = _ScriptedLitSense(error=ConnectionError("connection reset by peer"))
    module, original = _with_engine(engine)
    try:
        result = literature_search_litsense.invoke({"query": "metformin and cancer"})
    finally:
        module.PyLitSense = original

    assert "Error retrieving literature" in result
    assert "connection reset by peer" in result


def test_both_remaining_search_tools_are_registered() -> None:
    '''LitSense and SOP retrieval reach the agents that need them.

    Requirement of this build: removing the Librarian must not quietly take the
    other two with it. The planner gets the fast pair because it runs while the user
    waits at the approval gate.
    '''

    from core.agents.agents import PLANNING_TOOLS, RESEARCH_TOOLS

    research = {tool.name for tool in RESEARCH_TOOLS}
    planning = {tool.name for tool in PLANNING_TOOLS}

    assert {"literature_search_litsense", "protocol_search_sop"} <= research
    assert {"literature_search_litsense", "protocol_search_sop"} <= planning


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
