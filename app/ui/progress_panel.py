'''The live plan, read straight from `plan.md`.

"Which step are we on?" is the question a long repurposing run raises most often,
and before this it was answerable only by expanding a collapsed tool result. The
panel sits directly under the transcript and refreshes on the same tick as the file
list.

Because it reads the file rather than the transcript, it **cannot** disagree with
what actually happened — which is the whole reason the plan is a file. That is also
why `plan_init` / `plan_finalize` messages and `plan_update` / `plan_status` traffic
are suppressed from the transcript: they would only compete with this.
'''

from __future__ import annotations

from html import escape
from typing import Any, Optional

import gradio as gr

from app.config import logger
from app.ui.chat_timeline import AGENT_TITLES
from backend.utils import plan_store

_STATUS_STYLE = {
    plan_store.COMPLETED: ("done", "✓"),
    plan_store.IN_PROGRESS: ("active", "▸"),
    plan_store.BLOCKED: ("blocked", "!"),
    plan_store.SKIPPED: ("skipped", "–"),
}


def _agent_label(agent: str) -> str:
    if not agent:
        return ""
    return AGENT_TITLES.get(agent, agent.replace("_", " ").title())


def _step_markup(step: plan_store.PlanStep) -> str:
    css_state, mark = _STATUS_STYLE.get(step.status, ("pending", ""))
    parts = [
        f"<li class='plan-step plan-step--{css_state}'>",
        f"<span class='plan-step__mark' aria-hidden='true'>{escape(mark)}</span>",
        "<span class='plan-step__body'>",
        f"<span class='plan-step__title'>{step.number}. {escape(step.title)}</span>",
    ]
    meta = []
    agent = _agent_label(step.agent)
    if agent:
        meta.append(escape(agent))
    if step.note:
        meta.append(escape(step.note))
    if meta:
        parts.append(f"<span class='plan-step__note'>{' · '.join(meta)}</span>")
    parts.append("</span></li>")
    return "".join(parts)


def progress_markup(
    conversation_id: Optional[str],
    *,
    running: bool = False,
) -> str:
    '''Render the active run of this conversation's plan, or nothing.

    `running` adds a pulsing dot under the caption. A step can sit at
    `in_progress` in `plan.md` for minutes without a token reaching the browser,
    and a stopped or crashed run leaves that status behind — so "is anything
    happening?" is answered from the live session, not from the file.

    Parameters:
    ---------
    conversation_id (str): the conversation whose plan to render.
    running (boolean): whether a run is in flight, which changes the styling.

    Returns:
    ----------
    markup (str): the active run rendered from `plan.md`, or an empty string when there is no plan.
    '''

    if not conversation_id:
        return ""
    try:
        document = plan_store.load_document(conversation_id=conversation_id)
    except OSError as exc:
        logger.warning("Could not read the plan file for the progress panel: %s", exc)
        return ""

    run = document.active
    if run is None or not run.steps:
        return ""

    done, total = run.progress()
    percent = int(round(100 * done / total)) if total else 0
    unresolved = run.unresolved
    caption = (
        f"Step {unresolved[0].number} of {total} · {escape(unresolved[0].title)}"
        if unresolved
        else f"All {total} steps resolved"
    )

    header = [
        "<div class='plan-panel__head'>",
        "<span class='plan-panel__title'>Task monitor</span>",
        f"<span class='plan-panel__count'>{done}/{total}</span>",
        "</div>",
    ]
    if run.goal:
        header.append(f"<div class='plan-panel__goal'>{escape(run.goal)}</div>")
    header.append(f"<div class='plan-panel__caption'>{caption}</div>")
    if running:
        header.append(
            "<div class='plan-panel__live'>"
            "<span class='plan-panel__pulse' aria-hidden='true'></span>"
            "<span>Working…</span>"
            "</div>"
        )
    header.append(
        "<div class='plan-panel__bar' role='progressbar' "
        f"aria-valuenow='{percent}' aria-valuemin='0' aria-valuemax='100'>"
        f"<span style='width:{percent}%'></span></div>"
    )
    if run.constraints:
        conditions = "".join(f"<li>{escape(item)}</li>" for item in run.constraints)
        header.append(
            "<div class='plan-panel__conditions'>"
            "<span class='plan-panel__conditions-title'>Your approval conditions</span>"
            f"<ul>{conditions}</ul></div>"
        )

    steps = "".join(_step_markup(step) for step in run.steps)
    return (
        "<details class='plan-panel' open>"
        "<summary class='plan-panel__summary'>" + "".join(header) + "</summary>"
        f"<ul class='plan-panel__steps'>{steps}</ul>"
        "</details>"
    )


def progress_update(state: Any):
    '''Send the panel only when it differs from what was last sent.

    Same reasoning as the file list: re-sending a `gr.HTML` value swaps its DOM,
    which springs this panel's `<details>` back open and loses scroll position. The
    markup derives from `plan.md` plus the run flag and carries no timestamps, so
    it compares cleanly between renders — the pulse appears when a run starts and
    disappears on the render that follows the run leaving `running_threads`.

    Parameters:
    ---------
    state (UIState): the state to render the panel for.

    Returns:
    ----------
    update (gr.update or gr.skip): the panel, or `gr.skip()` when unchanged — otherwise `gr.HTML` swaps its DOM on every streamed event.
    '''

    markup = progress_markup(
        state.current_thread_id, running=state.is_running
    )
    if markup == state.last_progress_markup:
        return gr.skip()
    state.last_progress_markup = markup
    return gr.update(value=markup, visible=bool(markup))


__all__ = ["progress_markup", "progress_update"]
