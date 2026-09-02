'''The sidebar: conversations, their files, and inline previews of figures.

Three performance properties are deliberate, because this panel is re-rendered on
every streamed event:

* **Files are rendered for the active card only.** Clicking any other card activates
  that thread and re-renders, so a collapsed card's list is never actually read —
  and building them all put an O(conversations) filesystem crawl on startup.
* **Thumbnails are cached by `(path, mtime, size)`.** Encoding four full-size figures
  was almost all of the panel's render cost, paid on every event. The mtime and size
  are in the key so a figure regenerated at the same path still refreshes.
* **The panel is only sent when it changes.** `gr.HTML` swaps its whole DOM subtree
  whenever its value changes, taking scroll position and any open `<details>` with
  it — so during a run the file list could not be scrolled at all.

For a visualisation task the figure *is* the deliverable, so artifacts are shown,
not just named.
'''

from __future__ import annotations

import base64
import io
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional
    Image = None

from app.config import logger
from app.downloads import DOWNLOAD_ROUTE, build_download_payload, encode_download_token
from app.state import ConversationMeta, UIState

MAX_VISIBLE_FILES = 100
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
MAX_THUMBNAILS = 8
THUMBNAIL_MAX_PX = 220


@lru_cache(maxsize=64)
def _encode_thumbnail(path_value: str, mtime_ns: int, size: int) -> Optional[str]:
    '''Encode one image to a data URI. Keyed on identity *and* content.

    Parameters:
    ---------
    path_value (str): the image to encode.
    mtime_ns (int): its modification time, part of the cache key.
    size (int): its byte count, also part of the cache key.

    Returns:
    ----------
    src (str): the image as a data URI, or None when it is missing. Keyed on identity *and* content, so a regenerated figure at the same path is not served stale.
    '''

    del mtime_ns, size  # part of the cache key only
    if Image is None:
        return None
    try:
        with Image.open(path_value) as image:
            image.load()
            if image.mode not in ("RGB", "RGBA"):
                image = image.convert("RGB")
            image.thumbnail((THUMBNAIL_MAX_PX, THUMBNAIL_MAX_PX))
            buffer = io.BytesIO()
            image.save(buffer, format="PNG", optimize=True)
    except Exception as exc:  # noqa: BLE001 - a bad image must not break the sidebar
        logger.debug("Could not thumbnail %s: %s", path_value, exc)
        return None
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _thumbnail_data_uri(path_value: str) -> Optional[str]:
    path = Path(path_value)
    if path.suffix.lower() not in IMAGE_SUFFIXES:
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return _encode_thumbnail(str(path), stat.st_mtime_ns, stat.st_size)


def _render_thread_files(state: UIState, thread_id: str) -> str:
    files = state.thread_files.get(thread_id, [])
    if not files:
        return "<p class='conversation-card__empty'>No files yet.</p>"

    items: List[str] = []
    thumbnails_used = 0
    for record in files[:MAX_VISIBLE_FILES]:
        payload = build_download_payload(record, thread_id)
        if payload:
            token = encode_download_token(payload)
            href = f"{DOWNLOAD_ROUTE}?token={token}"
            name_markup = (
                "<a class='conversation-card__file-link' href='{href}' target='_blank' "
                "rel='noopener' data-download-link='{token}' data-file-name='{name}' "
                "download='{name}'>{label}</a>"
            ).format(
                href=escape(href, quote=True),
                token=escape(token, quote=True),
                name=escape(record.name, quote=True),
                label=escape(record.name),
            )
        else:
            name_markup = f"<span class='conversation-card__file-name'>{escape(record.name)}</span>"

        preview = ""
        if thumbnails_used < MAX_THUMBNAILS:
            data_uri = _thumbnail_data_uri(record.path)
            if data_uri:
                thumbnails_used += 1
                preview = (
                    "<div class='conversation-card__thumb'>"
                    f"<img src='{escape(data_uri, quote=True)}' "
                    f"alt='{escape(record.name, quote=True)}' loading='lazy' /></div>"
                )

        items.append(
            "<li class='conversation-card__file-item' title='{title}'>{name}{preview}</li>".format(
                title=escape(record.path), name=name_markup, preview=preview
            )
        )

    more = ""
    if len(files) > MAX_VISIBLE_FILES:
        more = (
            f"<li class='conversation-card__file-more'>"
            f"+{len(files) - MAX_VISIBLE_FILES} more…</li>"
        )
    return (
        "<div class='conversation-card__files-container'>"
        f"<ul class='conversation-card__files'>{''.join(items)}</ul>{more}</div>"
    )


def _thread_badge(state: UIState, thread_id: str, *, is_active: bool) -> str:
    '''Status dot for a thread the user is not currently looking at.

    Parameters:
    ---------
    state (UIState): the state holding each thread's status.
    thread_id (str): the thread to render a badge for.
    is_active (boolean): whether this is the thread on screen.

    Returns:
    ----------
    badge (str): the status dot, so a run on another thread is still visible.
    '''

    if thread_id in state.running_threads:
        return (
            "<span class='conversation-card__badge conversation-card__badge--running' "
            "title='Still running'>●</span>"
        )
    if not is_active and thread_id in state.stale_threads:
        return (
            "<span class='conversation-card__badge conversation-card__badge--updated' "
            "title='New activity since you last viewed this'>●</span>"
        )
    return ""


def conversation_panel_markup(state: UIState) -> str:
    cards: List[str] = [
        "<div class='conversation-list__container' id='conversation-list-root'>",
        "<div class='conversation-list__header'>Conversations</div>",
    ]
    if not state.thread_ids:
        cards.append("<p class='conversation-card__empty'>No conversations yet.</p></div>")
        return "\n".join(cards)

    for thread in state.thread_ids:
        thread_id = thread["thread_id"]
        is_active = thread_id == state.current_thread_id
        body = _render_thread_files(state, thread_id) if is_active else ""
        delete_button = (
            "<button type='button' class='conversation-card__delete' "
            f"data-delete-thread='{escape(thread_id, quote=True)}' "
            "data-confirm-message='Delete this conversation and its files?'>🗑</button>"
        )
        cards.append(
            "<details class='conversation-card {classes}' data-thread-id='{tid}' {open_attr}>"
            "<summary>"
            "<div class='conversation-card__title-row'>"
            "<span class='conversation-card__chevron' aria-hidden='true'></span>"
            "<span class='conversation-card__title'>{title}</span>"
            "{badge}{delete}"
            "</div>"
            "</summary>"
            "<div class='conversation-card__body'>{files}</div>"
            "</details>".format(
                classes="is-active" if is_active else "",
                tid=escape(thread_id, quote=True),
                open_attr="open" if is_active else "",
                title=escape(thread.get("title") or "Conversation"),
                badge=_thread_badge(state, thread_id, is_active=is_active),
                delete=delete_button,
                files=body,
            )
        )
    cards.append("</div>")
    return "\n".join(cards)


def conversation_panel_update(state: UIState):
    '''Send the sidebar only when it actually differs from what was last sent.

    Parameters:
    ---------
    state (UIState): the state to render the sidebar for.

    Returns:
    ----------
    update (gr.update or gr.skip): the sidebar, or `gr.skip()` when unchanged — otherwise the file list cannot be scrolled during a run.
    '''

    markup = conversation_panel_markup(state)
    if markup == state.last_panel_markup:
        return gr.skip()
    state.last_panel_markup = markup
    return gr.update(value=markup)


def invalidate_panel_cache(state: UIState) -> None:
    '''Force the next render to send the sidebar, whatever it looks like.

    Parameters:
    ---------
    state (UIState): the state whose cached sidebar to discard.
    '''

    state.last_panel_markup = None
    state.last_progress_markup = None


def thread_to_dict(meta: ConversationMeta) -> Dict[str, Any]:
    return {
        "thread_id": meta.thread_id,
        "title": meta.title,
        "created_at": meta.created_at,
        "updated_at": meta.updated_at,
    }


def append_file_paths(prompt: str, state: UIState) -> str:
    '''Tell the agents where the user's uploads landed.

    Parameters:
    ---------
    prompt (str): what the user typed.
    state (UIState): the state holding this turn's uploads.

    Returns:
    ----------
    prompt (str): the message with the upload paths appended, so the agents know where the files landed.
    '''

    files = state.uploaded_files
    if not files:
        return prompt
    if len(files) == 1:
        return f"{prompt}\n\nUploaded file: {files[0].path}"
    return prompt + "\n\nUploaded files:\n" + "\n".join(f"- {file.path}" for file in files)


__all__ = [
    "append_file_paths",
    "conversation_panel_markup",
    "conversation_panel_update",
    "invalidate_panel_cache",
    "thread_to_dict",
]
