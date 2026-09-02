'''Entry point: mount the UI on FastAPI and serve it.

Everything else lives in its own module — `app/ui/layout.py` builds the widgets,
`app/run_controller.py` drives a run, `app/session.py` handles conversations. This
file only wires them to a server, which is why it is short.
'''

from __future__ import annotations

import inspect
from contextlib import asynccontextmanager

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.config import APP_TITLE, DATABASE_PATH, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, logger
from app.downloads import FILES_ROUTER
from app.ui.layout import build_demo
from backend.db import (
    ConversationRepository,
    close_checkpointer,
    close_connection,
    get_checkpointer,
    import_legacy_conversations,
)
from backend.utils.retention import retention_worker

__all__ = ["build_demo", "create_fastapi_app", "launch"]


@asynccontextmanager
async def _lifespan(_: FastAPI):
    '''Open the database, verify the schema, carry old conversations forward.

    Doing this at startup rather than lazily means an unwritable database directory
    fails on boot with one clear error, instead of surfacing as an empty sidebar
    later. The order matters: the checkpointer creates its own tables, and the
    legacy import reads them to recover conversations that predate the
    `conversations` table.

    Parameters:
    ---------
    _ (FastAPI): the app, which this hook does not need.

    Returns:
    ----------
    lifespan (async generator): yields once the database is ready and the retention worker started.
    '''

    await ConversationRepository().ensure_schema()
    await get_checkpointer()
    await import_legacy_conversations()
    await retention_worker.start()
    logger.info("%s ready on %s:%s (database: %s)", APP_TITLE, GRADIO_SERVER_NAME, GRADIO_SERVER_PORT, DATABASE_PATH)
    try:
        yield
    finally:
        await retention_worker.stop()
        await close_checkpointer()
        await close_connection()


def create_fastapi_app() -> FastAPI:
    demo = build_demo()
    application = FastAPI(title=APP_TITLE, lifespan=_lifespan)
    application.include_router(FILES_ROUTER)

    mount_kwargs = {"path": "/"}
    if "footer_links" in inspect.signature(gr.mount_gradio_app).parameters:
        mount_kwargs["footer_links"] = ["api", "gradio"]
    return gr.mount_gradio_app(application, demo, **mount_kwargs)


def launch() -> None:
    uvicorn.run(
        create_fastapi_app(),
        host=GRADIO_SERVER_NAME,
        port=GRADIO_SERVER_PORT,
        log_level="info",
    )
