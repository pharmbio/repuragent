from __future__ import annotations

from html import escape
from typing import Any, Dict, Optional

import gradio as gr

DEFAULT_APPROVAL_MESSAGE = ("Review the plan above. Approve it to start execution, or describe the changes you want and the planner will revise it.")

APPROVE_TEXT = "Approved. Please proceed with the plan as written."

REQUEST_CHANGES_HINT = ("Describe what should change about the plan, then send. The planner will revise it and bring it back for another review.")


def approval_banner_markup(payload: Optional[Dict[str, Any]]) -> str:
    if not payload:
        return ""
    message = str(payload.get("message") or DEFAULT_APPROVAL_MESSAGE)
    return (
        "<div class='approval-panel' role='status' aria-live='polite'>"
        "<div class='approval-panel__title'>"
        "<span class='approval-panel__icon' aria-hidden='true'>⏸</span>"
        "Waiting for your approval"
        "</div>"
        f"<div class='approval-panel__message'>{escape(message)}</div>"
        "<div class='approval-panel__hint'>"
        "You can also approve with conditions — e.g. "
        "<em>“go ahead, but only drugs in phase 3 or later”</em> — by typing them below. "
        "They are recorded with the plan and override the matching steps."
        "</div>"
        "</div>"
    )


def approval_updates(payload: Optional[Dict[str, Any]]):
    '''Updates for (banner, approve button, request-changes button).

    Parameters:
    ---------
    payload (dict): the plan-review payload, or None when the thread is not paused.

    Returns:
    ----------
    updates (tuple): updates for the banner, the approve button and the request-changes button.
    '''

    waiting = payload is not None
    return (
        gr.update(value=approval_banner_markup(payload), visible=waiting),
        gr.update(visible=waiting),
        gr.update(visible=waiting),
    )


__all__ = [
    "APPROVE_TEXT",
    "DEFAULT_APPROVAL_MESSAGE",
    "REQUEST_CHANGES_HINT",
    "approval_banner_markup",
    "approval_updates",
]
