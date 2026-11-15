'''The seams between modules.

Splitting a 2 700-line module into twenty leaves gaps that no import can catch: a
handler wired with the wrong number of inputs, or a method called on a collaborator
that never had it. Both are runtime errors on a user's first click, and both are
structural enough to check here.

The web build hit exactly this when sign-in moved out of the monolith: the calls
landed on the service before the methods did. This build has no sign-in, so the same
check is pointed at the conversation store's collaborator instead.
'''

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import List, Tuple

EVENT_METHODS = {"click", "submit", "change", "tick", "upload", "load"}


def _handler_arity(module, name: str) -> Tuple[int, bool]:
    '''`(positional parameter count, accepts *args)` for a handler function.

    Parameters:
    ---------
    module (module): where the handler is defined.
    name (str): the handler's name.

    Returns:
    ----------
    arity (tuple): `(positional parameter count, accepts *args)`.
    '''

    function = getattr(module, name)
    signature = inspect.signature(function)
    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    star = any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    return len(positional), star


def _wired_events() -> List[Tuple[str, int]]:
    '''`(handler name, number of wired inputs)` for every event in `layout.py`.

    Returns:
    ----------
    events (list): `(handler name, number of wired inputs)` for every event in `layout.py`, so a handler cannot drift from its wiring.
    '''

    source = pathlib.Path("app/ui/layout.py").read_text()
    tree = ast.parse(source)
    events: List[Tuple[str, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr not in EVENT_METHODS:
            continue

        handler = node.args[0] if node.args else None
        if handler is None:
            handler = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "fn"), None
            )
        if not isinstance(handler, ast.Name):
            continue

        inputs = next((keyword.value for keyword in node.keywords if keyword.arg == "inputs"), None)
        if inputs is None:
            count = 0
        elif isinstance(inputs, ast.List):
            count = len(inputs.elts)
        elif isinstance(inputs, ast.Constant) and inputs.value is None:
            count = 0
        else:
            count = 1  # a single component passed bare
        events.append((handler.id, count))
    return events


def test_every_handler_accepts_its_wired_inputs() -> None:
    from app.ui import layout

    events = _wired_events()
    # A floor, not a count: it catches the parser silently finding nothing, which
    # would make this test vacuously pass. Four fewer than the web build, whose
    # sign-in, registration, password-reset and sign-out clicks are gone.
    assert len(events) >= 13, f"expected the full wiring, found {len(events)}"

    problems: List[str] = []
    for name, wired in events:
        if not hasattr(layout, name):
            problems.append(f"{name} is wired but not imported into layout")
            continue
        arity, star = _handler_arity(layout, name)
        if star:
            continue
        if arity != wired:
            problems.append(f"{name} takes {arity} argument(s) but is wired with {wired}")
    assert not problems, "Gradio wiring mismatch:\n  " + "\n  ".join(problems)


def test_repository_has_the_api_the_stores_call() -> None:
    from backend.db.repository import ConversationRepository

    for name in (
        "ensure_schema",
        "upsert_thread",
        "list_threads",
        "get_thread",
        "update_thread_title",
        "get_thread_timeline",
        "update_thread_timeline",
        "delete_thread",
    ):
        assert callable(getattr(ConversationRepository, name, None)), (
            f"ConversationRepository.{name} is missing"
        )


def test_timeline_scoping_takes_a_thread() -> None:
    '''A rendered timeline belongs to one conversation, and is addressed by its id.'''

    from backend.db.repository import ConversationRepository

    for name in ("get_thread_timeline", "update_thread_timeline"):
        parameters = list(inspect.signature(getattr(ConversationRepository, name)).parameters)
        assert parameters[1] == "thread_id", (name, parameters)


def test_no_module_still_points_at_the_old_layout() -> None:
    '''Nothing may import the modules this refactor removed.'''

    stale = (
        "core.supervisor",
        "langgraph_supervisor",
        "backend.utils.kgg_tools",
        "backend.utils.research_tools",
        "backend.utils.prediction_tools",
        "backend.utils.chemical_annotator.tools",
        "persistence.memory.episodic_memory.thread_manager",
        "persistence.memory.episodic_memory.conversation",
        "kgg.kgg_apiutils",
    )
    # The three subsystems this build removes — auth, PostgreSQL, the Librarian —
    # are guarded by `test_imports`, which searches for their *names* rather than
    # their imports and would flag this list if it repeated them.
    offenders: List[str] = []
    for path in list(pathlib.Path("app").rglob("*.py")) + list(
        pathlib.Path("core").rglob("*.py")
    ) + list(pathlib.Path("backend").rglob("*.py")) + list(pathlib.Path("tests").rglob("*.py")):
        text = path.read_text()
        for name in stale:
            if name in text and "test_wiring" not in path.name:
                offenders.append(f"{path}: {name}")
    assert not offenders, "stale imports:\n  " + "\n  ".join(offenders)


def test_config_is_the_only_place_naming_a_model() -> None:
    '''Model ids belong in one file, so a deployment can change them in one place.'''

    offenders: List[str] = []
    for path in list(pathlib.Path("app").rglob("*.py")) + list(pathlib.Path("core").rglob("*.py")):
        if path.name == "config.py":
            continue
        text = path.read_text()
        for marker in ("gpt-5", "gpt-4", "claude-", "o3-", "text-embedding"):
            if marker in text:
                offenders.append(f"{path}: {marker}")
    assert not offenders, "hard-coded model ids:\n  " + "\n  ".join(offenders)


if __name__ == "__main__":
    from tests.harness import run_module_tests

    run_module_tests(globals())
