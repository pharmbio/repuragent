'''The widget tree and every event wiring.

One list — `workspace_outputs` — is the output shape of every handler that can change
the workspace, because `projection.render` returns exactly that tuple. Before this
each handler named its own outputs and they drifted apart, so a handler could resolve
the spinner without ever updating the buttons.
'''

from __future__ import annotations

import gradio as gr

from app.config import (
    APP_TITLE,
    FILE_LIST_REFRESH_INTERVAL_SECONDS,
    UI_CONCURRENCY_LIMIT,
    UI_QUEUE_MAX_SIZE,
)
from app.run_controller import (
    on_approve_plan,
    on_request_changes,
    on_send_message,
    on_stop_run,
)
from app.session import (
    on_app_load,
    on_clear_files,
    on_conversation_action,
    on_extract_learning,
    on_files_uploaded,
    on_new_task,
    on_periodic_file_refresh,
    on_toggle_learning,
)
from app.ui.assets import (
    HEADER_LINKS_HTML,
    intro_markdown,
    logo_html,
    partner_logos_html,
    title_html,
)
from app.ui.scripts import CONVERSATION_SCRIPT
from app.ui.theme import APP_CSS, REPURAGENT_THEME


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title=APP_TITLE,
        theme=REPURAGENT_THEME,
        css=APP_CSS,
        head=CONVERSATION_SCRIPT,
    ) as demo:
        state = gr.State()

        # --- Header ---------------------------------------------------------
        with gr.Row(elem_id="app-header"):
            logo_markup = logo_html()
            if logo_markup:
                with gr.Column(scale=0, min_width=96):
                    gr.HTML(logo_markup, elem_id="app-logo")
            with gr.Column(scale=1):
                gr.HTML(title_html(), elem_id="app-title")
            with gr.Column(scale=0, min_width=240, elem_id="header-links-column"):
                gr.HTML(HEADER_LINKS_HTML, elem_id="header-links")

        partner_panel = partner_logos_html()
        if partner_panel:
            gr.HTML(partner_panel, elem_id="partner-logos-panel")

        with gr.Row(elem_id="layout-row"):
            # --- Sidebar ----------------------------------------------------
            with gr.Column(scale=1, min_width=280, elem_id="sidebar-column"):
                # There is no sign-in, so "New task" is live from the first frame.
                new_task_btn = gr.Button("New task", elem_id="new-task-button")

                conversation_list = gr.HTML(
                    value="", elem_id="conversation-list", min_height=10, container=False
                )
                # The sidebar is one HTML block, so a click on a card has no Gradio
                # event of its own; the injected script writes actions here instead.
                conversation_action_bus = gr.Textbox(
                    value="", show_label=False, elem_id="conversation-action-bus"
                )
                file_refresh_timer = gr.Timer(
                    value=FILE_LIST_REFRESH_INTERVAL_SECONDS, active=True, render=False
                )
                file_upload = gr.File(
                    label="Upload data",
                    file_count="multiple",
                    file_types=["file"],
                    elem_id="file-upload-panel",
                )
                clear_files_btn = gr.Button("Clear uploads", elem_id="clear-files-button")

                with gr.Column(elem_id="learning-controls"):
                    use_learning = gr.Checkbox(
                        label="Plan from past tasks",
                        value=True,
                        info="Give the planner precedent from earlier successful runs.",
                    )
                    extract_btn = gr.Button(
                        "Remember this plan", elem_id="extract-learning-button"
                    )
                    learning_status = gr.Markdown(value="", elem_id="learning-status")

            # --- Workspace --------------------------------------------------
            with gr.Column(scale=4, elem_id="conversation-column"):
                gr.Markdown(intro_markdown(), elem_id="intro-text")
                chatbot = gr.Chatbot(
                    label="Conversation", height=560, type="messages", elem_id="chatbot-panel"
                )

                # The live plan. Directly under the transcript because "which step
                # are we on" is the question a long run raises most often, and it
                # used to be answerable only by expanding a collapsed tool result.
                progress_panel = gr.HTML(
                    value="", visible=False, elem_id="progress-panel", container=False
                )

                # The approval gate. Hidden until the graph actually pauses.
                approval_banner = gr.HTML(
                    value="", visible=False, elem_id="approval-banner", container=False
                )
                with gr.Row(elem_id="approval-actions-row"):
                    approve_btn = gr.Button(
                        "✓ Approve plan", variant="primary", visible=False, elem_id="approve-button"
                    )
                    request_changes_btn = gr.Button(
                        "✎ Request changes",
                        variant="secondary",
                        visible=False,
                        elem_id="request-changes-button",
                    )

                user_input = gr.Textbox(
                    label="Your request",
                    lines=3,
                    placeholder="e.g. Find repurposing candidates for acute myeloid leukemia and rank them by hERG risk",
                    elem_id="user-input",
                )
                with gr.Row(elem_id="input-actions-row"):
                    with gr.Column(scale=9):
                        send_btn = gr.Button("Send", variant="primary", elem_id="send-button")
                    with gr.Column(scale=1, min_width=120):
                        stop_btn = gr.Button(
                            "Stop", variant="secondary", interactive=False, elem_id="stop-button"
                        )

        # --- Wiring ---------------------------------------------------------
        # One output shape for every handler that can change the workspace; see
        # `app/ui/projection.py::render`.
        workspace_outputs = [
            state,
            chatbot,
            user_input,
            conversation_list,
            approval_banner,
            approve_btn,
            request_changes_btn,
            send_btn,
            stop_btn,
            progress_panel,
        ]

        demo.load(on_app_load, inputs=None, outputs=workspace_outputs + [conversation_action_bus])

        conversation_action_bus.change(
            on_conversation_action,
            inputs=[conversation_action_bus, state],
            outputs=workspace_outputs + [conversation_action_bus],
        )

        file_refresh_timer.tick(
            on_periodic_file_refresh,
            inputs=[state],
            outputs=[state, conversation_list, progress_panel],
            trigger_mode="always_last",
        )

        new_task_btn.click(on_new_task, inputs=state, outputs=workspace_outputs)

        file_upload.upload(
            on_files_uploaded, inputs=[file_upload, state], outputs=[state, conversation_list]
        )
        clear_files_btn.click(on_clear_files, inputs=state, outputs=[state, conversation_list])

        send_btn.click(on_send_message, inputs=[user_input, state], outputs=workspace_outputs)
        user_input.submit(on_send_message, inputs=[user_input, state], outputs=workspace_outputs)

        approve_btn.click(on_approve_plan, inputs=state, outputs=workspace_outputs)
        request_changes_btn.click(on_request_changes, inputs=state, outputs=workspace_outputs)
        stop_btn.click(on_stop_run, inputs=state, outputs=workspace_outputs)

        use_learning.change(on_toggle_learning, inputs=[use_learning, state], outputs=state)
        extract_btn.click(on_extract_learning, inputs=state, outputs=learning_status)

    return demo.queue(
        max_size=UI_QUEUE_MAX_SIZE, default_concurrency_limit=UI_CONCURRENCY_LIMIT
    )


__all__ = ["build_demo"]
