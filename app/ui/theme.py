'''Palette and stylesheet.

The brand colours are Repuragent's fern green. The structure is built around the
three tiers a run produces — the plan under review, quiet working narration, and the
deliverable as a document — because rendering them identically made the answer as
hard to find as a stray tool call.

The theme is pinned to light. A dark scheme was never designed for these components,
and the browser's preference used to leak through into half-styled panels.
'''

from __future__ import annotations

import gradio as gr
from gradio.themes.utils import colors

PRIMARY_FERN = colors.Color(
    c50="#dbeee5",
    c100="#cfe3d9",
    c200="#bad4c7",
    c300="#9fc3b2",
    c400="#78a78f",
    c500="#3f7f6e",
    c600="#1f5c55",
    c700="#184842",
    c800="#10322d",
    c900="#0a211f",
    c950="#05110f",
    name="repuragent_primary_green",
)

SECONDARY_SAGE = colors.Color(
    c50="#edf6f2",
    c100="#dfeee6",
    c200="#c8dfd3",
    c300="#b2d1c0",
    c400="#95bfa9",
    c500="#79ad92",
    c600="#5e967c",
    c700="#4b7761",
    c800="#365646",
    c900="#233a2f",
    c950="#142019",
    name="repuragent_secondary_green",
)

REPURAGENT_THEME = gr.themes.Default(
    primary_hue=PRIMARY_FERN,
    secondary_hue=SECONDARY_SAGE,
    neutral_hue=colors.gray,
).set(
    color_accent="*primary_600",
    color_accent_soft="#dbeee5",
    color_accent_soft_dark="*primary_700",
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_500",
    button_primary_text_color="#f6fbf8",
    button_primary_text_color_hover="#f6fbf8",
)


APP_CSS = """
    :root {
        color-scheme: light;
        --font-ui: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        --font-editorial: "Palatino Linotype", Palatino, "Book Antiqua", Georgia, serif;
        --font-mono: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;

        /* One type scale for the whole app. Before this there were thirty
           hand-picked sizes between 0.64rem and 1.4rem, a dozen of them within a
           hair of each other, which is what made neighbouring elements look
           mismatched rather than deliberately different. Six tiers, each with a
           job: */
        --fs-eyebrow: 0.72rem;   /* uppercase micro-labels, tags, tab names     */
        --fs-meta: 0.8rem;       /* notes, durations, badges, captions          */
        --fs-ui-sm: 0.88rem;     /* compact UI: tool lines, plan steps, sidebar */
        --fs-ui: 0.95rem;        /* standard UI copy and controls               */
        --fs-body: 1.02rem;      /* the reading tier: agent prose and the user  */
        --fs-lead: 1.12rem;      /* a lead line or a heading inside prose       */
        --fs-title: 1.35rem;     /* the report's title                          */
        --fs-display: clamp(2.5rem, 4vw, 3.4rem);  /* unchanged: the app title */
        --lh-tight: 1.3;    /* headings and one-line labels */
        --lh-ui: 1.5;       /* compact UI text             */
        --lh-copy: 1.6;     /* short paragraphs in a card  */
        --lh-body: 1.7;     /* the reading tier            */
        --page-bg: #f7f9f8;
        --surface-bg: #ffffff;
        --surface-muted: #fbfcfb;
        --surface-tint: #f1f6f3;
        --field-bg: #ffffff;
        --text-main: #16211d;
        --text-soft: #5c6b64;
        --border-subtle: #d8e2dc;
        --border-strong: #b9c9c1;
        --brand: #1f5c55;
        --brand-strong: #184842;
        --brand-soft: #dbeee5;
        --focus-color: #f0b429;
        --ok-color: #2f8f4e;
        --error-color: #b3382b;
        --partner-card-width: 220px;
        --partner-card-gap: 1.15rem;
    }
    html, body, .gradio-container {
        color-scheme: light !important;
        font-family: var(--font-ui);
        background: var(--page-bg);
        color: var(--text-main);
    }
    .gradio-container *, .gradio-container *::before, .gradio-container *::after { box-sizing: border-box; }
    .gradio-container {
        max-width: none;
        width: 100vw;
        margin: 0 auto !important;
        padding: 1.5rem 1.25rem 2rem;
    }
    .gradio-container a { color: var(--brand); text-underline-offset: 0.14em; }
    .gradio-container a:hover, .gradio-container a:focus { color: var(--brand-strong); }
    .gradio-container button, .gradio-container input, .gradio-container textarea,
    .gradio-container label, .gradio-container .tabs, .gradio-container .tabitem {
        font-family: var(--font-ui) !important;
    }
    .gradio-container button:focus-visible, .gradio-container input:focus-visible,
    .gradio-container textarea:focus-visible, .gradio-container [role="tab"]:focus-visible,
    .conversation-card__delete:focus-visible, .partner-logo-card:focus-visible {
        outline: 3px solid var(--focus-color);
        outline-offset: 2px;
    }

    /* ---- Header ------------------------------------------------------------ */
    #app-header {
        align-items: center;
        gap: 1rem;
        margin-bottom: 0.8rem;
        padding: 0 0 0.9rem;
        border-bottom: 1px solid var(--border-strong);
    }
    #app-logo { display: flex; align-items: center; justify-content: center; padding: 0 !important; }
    #app-logo .app-logo-img { width: 84px; height: 84px; object-fit: contain; display: block; }
    #app-title { margin: 0 !important; padding: 0 !important; display: flex; align-items: center; }
    #app-title .app-title-text {
        font-family: var(--font-editorial);
        font-size: var(--fs-display);
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 0.95;
        margin: 0;
        color: var(--text-main);
    }
    #app-title .app-tagline {
        font-family: var(--font-ui);
        font-size: var(--fs-ui-sm);
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--text-soft);
        margin-top: 0.35rem;
    }
    #header-links-column {
        margin-left: auto;
        padding: 0 !important;
        display: flex;
        justify-content: flex-end;
        align-items: center;
        overflow: visible !important;
    }
    #header-links {
        display: flex;
        gap: 0.5rem;
        align-items: center;
        font-weight: 600;
        font-size: var(--fs-ui-sm);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        overflow: visible !important;
        white-space: nowrap;
    }
    #header-links .header-link { color: var(--text-main); text-decoration: none; transition: color 0.2s ease; }
    #header-links .header-link-divider { color: var(--border-strong); font-weight: 400; padding: 0 0.9rem; user-select: none; }
    #header-links .header-link:hover, #header-links .header-link:focus { color: var(--brand); text-decoration: underline; }

    /* ---- Partner strip ----------------------------------------------------- */
    #partner-logos-panel {
        width: 100%;
        margin: 0 auto 0.85rem;
        padding: 0.4rem 0 0.1rem;
        border-top: 1px solid var(--border-subtle);
        border-bottom: 1px solid var(--border-subtle);
        background: linear-gradient(180deg, #fbfcfb 0%, var(--page-bg) 100%);
    }
    #partner-logos-panel .partner-slider { width: 100%; margin: 0; }
    .partner-slider__viewport { overflow: hidden; width: 100%; }
    .partner-slider__track {
        display: flex;
        gap: var(--partner-card-gap);
        padding: 0.25rem;
        will-change: transform;
        transition: transform 0.4s ease;
    }
    .partner-logo-card {
        background: var(--surface-bg);
        border: 1px solid var(--border-subtle);
        padding: 1rem 1.7rem;
        min-height: 104px;
        min-width: var(--partner-card-width);
        width: var(--partner-card-width);
        max-width: var(--partner-card-width);
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.15s ease, border-color 0.15s ease;
        flex: 0 0 var(--partner-card-width);
    }
    .partner-logo-card:hover, .partner-logo-card:focus-visible { background: var(--surface-tint); border-color: var(--brand); }
    .partner-logo-card img {
        max-height: 68px;
        max-width: calc(var(--partner-card-width) - 20px);
        width: auto; height: auto; object-fit: contain;
    }
    .partner-logo-card--xl {
        min-width: calc(var(--partner-card-width) + 60px);
        width: calc(var(--partner-card-width) + 60px);
        max-width: calc(var(--partner-card-width) + 60px);
    }
    .partner-logo-card--xl img { max-height: 88px; max-width: calc(var(--partner-card-width) + 20px); }
    .partner-slider__dots { display: flex; justify-content: center; gap: 0.45rem; margin-top: 0.5rem; }
    .partner-slider__dot {
        width: 9px; height: 9px; border-radius: 999px;
        background: var(--border-strong); border: 0; cursor: pointer; transition: all 0.2s ease;
    }
    .partner-slider__dot.is-active { background: var(--brand); }

    /* ---- Layout ------------------------------------------------------------ */
    #layout-row { width: 100%; gap: 1.5rem; align-items: flex-start; }
    #layout-row > div { min-width: 0; }
    #conversation-column {
        display: flex; flex-direction: column; gap: 0.55rem;
        padding: 1rem 1rem 1.1rem;
        background: var(--page-bg);
        border: 1px solid transparent;
        flex: 1 1 0 !important; width: auto !important; min-width: 0 !important;
    }
    #sidebar-column {
        display: flex; flex-direction: column; gap: 0.75rem;
        position: sticky; top: 1rem; align-self: flex-start;
        min-width: 312px; width: 312px !important; flex: 0 0 312px !important; max-width: 312px;
    }
    #sidebar-column > div {
        background: var(--page-bg); border: 1px solid transparent;
        box-shadow: none; padding: 0.2rem;
    }
    #conversation-column > div { background: transparent; border: 0; box-shadow: none; }
    #intro-text { margin: 0 0 0.18rem 0 !important; padding-bottom: 0.3rem; width: 100%; border-bottom: 1px solid var(--border-subtle); }
    #intro-text img {
        width: 100%; max-width: 100%; max-height: 320px; height: auto;
        display: block; margin: 0 auto;
        border: 1px solid var(--border-subtle); object-fit: contain; background: #fff;
    }

    /* ---- Sidebar controls -------------------------------------------------- */
    /* Text fields only. A checkbox must be excluded: gradio paints its tick with a
       `background-image` (`--checkbox-check`) on `:checked`, and `background` here
       is a SHORTHAND with `!important`, so it reset `background-image` to none in
       every state. The box then looked identical checked and unchecked — the
       episodic-memory toggle flipped its state invisibly and read as a dead
       control, including at startup where its value is True. Set only
       `background-color` on a checkbox, never `background`. */
    #sidebar-column input:not([type="checkbox"]):not([type="radio"]),
    #sidebar-column textarea, #conversation-column textarea {
        border-radius: 0 !important; border: 1px solid var(--border-strong) !important;
        background: var(--field-bg) !important; color: var(--text-main) !important;
    }
    #sidebar-column input[type="checkbox"] {
        width: 1.05rem; height: 1.05rem; flex: 0 0 auto; cursor: pointer;
        border: 1px solid var(--border-strong) !important; border-radius: 0 !important;
        background-color: var(--field-bg) !important;
    }
    #sidebar-column input[type="checkbox"]:hover { border-color: var(--brand) !important; }
    #sidebar-column input[type="checkbox"]:checked {
        background-color: var(--brand) !important; border-color: var(--brand) !important;
    }
    /* The label of a checkbox is a control, not a field caption: the uppercase
       soft-grey treatment below made an enabled toggle look disabled. Overriding
       on the span works because a declaration on the element always beats an
       inherited one, `!important` or not. */
    #sidebar-column input[type="checkbox"] + span {
        font-size: var(--fs-ui-sm); font-weight: 600; letter-spacing: 0;
        text-transform: none; color: var(--text-main);
    }
    #sidebar-column input:focus, #sidebar-column textarea:focus, #conversation-column textarea:focus {
        border-color: var(--brand) !important; box-shadow: none !important;
    }
    #sidebar-column input:-webkit-autofill, #conversation-column textarea:-webkit-autofill {
        -webkit-text-fill-color: var(--text-main) !important;
        -webkit-box-shadow: 0 0 0 1000px var(--field-bg) inset !important;
        transition: background-color 9999s ease-out 0s;
    }
    #sidebar-column label, #conversation-column label {
        font-size: var(--fs-meta) !important; font-weight: 700 !important; letter-spacing: 0.04em;
        text-transform: uppercase; color: var(--text-soft) !important;
    }
    #sidebar-column button, #conversation-column button {
        border-radius: 0 !important; box-shadow: none !important;
        font-weight: 700 !important; letter-spacing: 0.02em;
    }
    #logout-button, #new-task-button, #clear-files-button, #extract-learning-button { min-height: 42px; }
    #new-task-button button:disabled, #send-button button:disabled, #stop-button button:disabled {
        background: #eef1ef !important; color: #93a29b !important;
        border-color: var(--border-subtle) !important; opacity: 1 !important;
    }
    #stop-button button, #sidebar-column button.secondary { border: 1px solid var(--border-strong) !important; }
    #learning-controls { padding-top: 0.35rem; border-top: 1px solid var(--border-subtle); }
    #learning-status p { font-size: var(--fs-meta) !important; color: var(--text-soft) !important; margin: 0.35rem 0 0 !important; }
    #file-upload-panel, #file-upload-panel > div, #file-upload-panel > div > div {
        background: transparent !important; border: 0 !important; padding: 0 !important;
        box-shadow: none !important; width: 100% !important;
    }
    #file-upload-panel .wrap {
        border: 1px dashed var(--border-strong) !important;
        background: linear-gradient(180deg, #ffffff 0%, var(--surface-muted) 100%) !important;
        min-height: 140px; padding: 0.85rem !important; width: 100% !important; max-width: 100% !important;
    }
    #file-upload-panel .or, #file-upload-panel .hint { color: var(--text-soft) !important; }
    #file-upload-panel .label-wrap, #file-upload-panel .label-wrap label { background: transparent !important; box-shadow: none !important; }
    #conversation-action-bus { display: none !important; }
    #input-actions-row {
        margin-top: 0.15rem; gap: 0.65rem; padding-top: 0.35rem;
        border-top: 1px solid var(--border-subtle); align-items: stretch;
    }
    #send-button, #stop-button { width: 100%; }
    #send-button button, #stop-button button { min-height: 46px; }
    #stop-button button { background: #fff !important; }
    #user-input, #user-input > div, #user-input > div > div {
        border: 0 !important; background: transparent !important;
        padding: 0 !important; box-shadow: none !important;
    }
    #user-input textarea {
        min-height: 104px !important; line-height: var(--lh-copy) !important;
        padding: 0.8rem 0.9rem !important; border: 1px solid var(--border-subtle) !important;
    }

    /* ---- Transcript -------------------------------------------------------- */
    /* Type in here is not simply what we declare. The transcript sits inside a
       gradio `.prose`, and a finished agent block is a gradio "thought", whose
       stylesheet contains a rule of the form `.content * { font-size:
       var(--text-sm) }` — a universal descendant selector, 12px — plus per
       element rules on `.prose p` and `.prose li`. So a size set on a container
       of ours reached the paragraphs we name and nothing else: measured, a plan
       rendered 16.6px prose with 12px bold labels, which is what made it look
       broken rather than small. Two defences, both needed:

         1. `--text-sm` is redefined for this subtree, so gradio's own rule
            resolves to our compact size instead of 12px. That catches every
            element we never thought to name.
         2. Everything we do name states its size and family explicitly, because
            `#chatbot-panel .prose p` outranks anything a container of ours is
            inherited from.

       Two traps, both of which cost an afternoon:

       - `:is()` must not be used in this stylesheet. Gradio prefixes every
         selector with `.gradio-container ... .contain`, and it rewrites the
         arguments inside `:is()` as it goes — which raises that list's
         specificity from a bare tag to a class chain, so a generic
         `:is(h1, h2, h3)` rule silently started outranking the specific
         `--report h1` rule below it. Write the lists out.
       - CSS comments do not nest. An inner marker inside this block ends it
         early and the prose that follows is parsed as CSS, which swallows the
         next rules whole.

       `[data-testid]` is `user` / `bot` in gradio 5; the rules here used to say
       `assistant`, which matched nothing. */
    #chatbot-panel {
        font-size: var(--fs-body); line-height: var(--lh-body);
        --text-sm: var(--fs-ui-sm);
        --chatbot-text-size: var(--fs-body);
        border-radius: 0 !important;
        background: var(--surface-bg) !important;
        border: 1px solid var(--border-subtle) !important;
    }
    #intro-text, #intro-text * { font-family: var(--font-editorial) !important; }
    #chatbot-panel .prose, #chatbot-panel .prose p, #chatbot-panel .prose li {
        font-size: inherit !important; line-height: inherit !important; color: var(--text-main) !important;
    }
    /* Agent prose is the editorial serif; the user's own words are the UI sans,
       so the two voices are distinguishable before they are read. */
    #chatbot-panel, #chatbot-panel .message, #chatbot-panel .prose,
    #chatbot-panel .prose p, #chatbot-panel .prose li,
    #chatbot-panel .agent-message-section,
    #chatbot-panel .agent-message-section p,
    #chatbot-panel .agent-message-section li,
    #chatbot-panel .agent-message-section strong,
    #chatbot-panel .agent-message-section b,
    #chatbot-panel .agent-message-section em,
    #chatbot-panel .agent-message-section i,
    #chatbot-panel .agent-message-section a,
    #chatbot-panel .agent-message-section blockquote {
        font-family: var(--font-editorial) !important;
    }
    #chatbot-panel [data-testid="user"],
    #chatbot-panel [data-testid="user"] p,
    #chatbot-panel [data-testid="user"] li,
    #chatbot-panel [data-testid="user"] strong,
    #chatbot-panel [data-testid="user"] em {
        font-size: var(--fs-body) !important; line-height: var(--lh-body) !important;
        font-family: var(--font-ui) !important;
    }
    #chatbot-panel [data-testid="chatbot-avatar"], #chatbot-panel .message, #chatbot-panel .message-row { border-radius: 0 !important; }
    #chatbot-panel .bubble-wrap, #chatbot-panel .message-wrap { padding-left: 0.15rem !important; padding-right: 0.15rem !important; }
    #chatbot-panel [data-testid="bot"] { background: #fff !important; }
    #chatbot-panel .bubble.user-row {
        background: var(--surface-tint) !important;
        border: 1px solid var(--border-subtle) !important;
        box-shadow: none !important; padding: 0.65rem 0.9rem !important;
    }
    #chatbot-panel .bubble.user-row .message.user, #chatbot-panel .bubble.user-row .message.user > div {
        background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important;
    }
    #chatbot-panel code, #chatbot-panel pre, .tool-code-block pre { font-family: var(--font-mono) !important; }

    /* The block's title bar is gradio's thought header. At `--text-sm` in the
       editorial serif it read as a footnote under its own content; it is a label,
       so it is sans, and it names the agent, so it is not the smallest thing on
       the screen. */
    #chatbot-panel .thought-group > .title .md,
    #chatbot-panel .thought-group > .title .md p {
        font-size: var(--fs-ui-sm) !important; font-family: var(--font-ui) !important;
        font-weight: 600; color: var(--text-main);
    }
    #chatbot-panel .thought-group > .title .duration {
        font-size: var(--fs-meta) !important; font-family: var(--font-ui) !important;
        color: var(--text-soft);
    }

    /* ---- The three tiers of agent output ----------------------------------- */
    /* All three read at one size. What separates them is family, colour and
       frame; scale was doing that job badly, because a 0.93 / 0.96 / 1.02
       difference reads as inconsistency rather than as hierarchy. */
    #chatbot-panel .agent-message-section {
        font-size: var(--fs-body); line-height: var(--lh-body);
    }
    #chatbot-panel .agent-message-section p,
    #chatbot-panel .agent-message-section ul,
    #chatbot-panel .agent-message-section ol,
    #chatbot-panel .agent-message-section li,
    #chatbot-panel .agent-message-section strong,
    #chatbot-panel .agent-message-section b,
    #chatbot-panel .agent-message-section em,
    #chatbot-panel .agent-message-section i,
    #chatbot-panel .agent-message-section a,
    #chatbot-panel .agent-message-section span,
    #chatbot-panel .agent-message-section small,
    #chatbot-panel .agent-message-section blockquote,
    #chatbot-panel .agent-message-section dt,
    #chatbot-panel .agent-message-section dd {
        font-size: inherit;
    }
    #chatbot-panel .agent-message-section h1,
    #chatbot-panel .agent-message-section h2,
    #chatbot-panel .agent-message-section h3,
    #chatbot-panel .agent-message-section h4 {
        font-size: var(--fs-lead); line-height: var(--lh-tight);
        margin: 1.15rem 0 0.5rem; font-weight: 700;
    }
    #chatbot-panel .agent-message-section code {
        font-size: 0.9em; background: var(--surface-tint); padding: 0.08em 0.35em;
    }
    /* The plan template leaves a blank line between steps, so each step is its
       own paragraph — this margin is the gap that makes the breakdown scannable.
       Inside a step the lines are `<br>`s, because markdown-it runs with
       `breaks: True`, so leading is the only thing separating `Agent:` from
       `Details:`. */
    #chatbot-panel .agent-message-section p { margin: 0 0 0.8rem; }
    #chatbot-panel .agent-message-section ul,
    #chatbot-panel .agent-message-section ol { margin: 0.5rem 0 0.8rem 1.3rem; }
    #chatbot-panel .agent-message-section li { margin: 0 0 0.4rem; }
    #chatbot-panel .agent-message-section > :first-child { margin-top: 0; }
    #chatbot-panel .agent-message-section > :last-child { margin-bottom: 0; }
    .agent-message-inline { margin: 0.9rem 0; white-space: pre-wrap; word-break: break-word; }

    /* Working narration: quieter than the deliverable, still readable. */
    .agent-message-section--activity { color: var(--text-main); }

    /* The plan under review. */
    .agent-block-content--plan { border-left: 3px solid var(--brand); padding-left: 0.9rem; }
    #chatbot-panel .agent-message-section--plan h1,
    #chatbot-panel .agent-message-section--plan h2,
    #chatbot-panel .agent-message-section--plan h3 { font-family: var(--font-ui) !important; }
    .agent-message-section--plan strong { color: var(--brand-strong); }

    /* The deliverable. Reads as a document, not another chat bubble. */
    .agent-block-content--report {
        background: var(--surface-bg);
        border: 1px solid var(--border-subtle);
        border-top: 3px solid var(--brand);
        padding: 1.15rem 1.35rem 1.05rem;
        margin: 0.3rem 0;
    }
    #chatbot-panel .agent-message-section--report,
    #chatbot-panel .agent-message-section--report p,
    #chatbot-panel .agent-message-section--report li,
    #chatbot-panel .agent-message-section--report strong,
    #chatbot-panel .agent-message-section--report b,
    #chatbot-panel .agent-message-section--report em,
    #chatbot-panel .agent-message-section--report a,
    #chatbot-panel .agent-message-section--report th,
    #chatbot-panel .agent-message-section--report td,
    #chatbot-panel .agent-message-section--report h2,
    #chatbot-panel .agent-message-section--report h3 {
        font-family: var(--font-ui) !important;
    }
    #chatbot-panel .agent-message-section--report h1 {
        font-family: var(--font-editorial) !important;
        font-size: var(--fs-title) !important; line-height: var(--lh-tight);
        margin: 0 0 0.75rem; padding-bottom: 0.5rem; border-bottom: 1px solid var(--border-subtle);
    }
    #chatbot-panel .agent-message-section--report h2 {
        font-size: var(--fs-eyebrow) !important; font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-soft);
        margin: 1.3rem 0 0.45rem;
    }
    #chatbot-panel .agent-message-section--report h2:first-of-type { margin-top: 0.2rem; }
    #chatbot-panel .agent-message-section--report table {
        border-collapse: collapse; width: 100%; margin: 0.7rem 0;
    }
    #chatbot-panel .agent-message-section--report th,
    #chatbot-panel .agent-message-section--report td {
        border: 1px solid var(--border-subtle); padding: 0.45rem 0.6rem; text-align: left;
        font-size: var(--fs-ui-sm); line-height: var(--lh-ui);
    }
    #chatbot-panel .agent-message-section--report th { background: var(--surface-tint); font-weight: 700; }

    /* ---- One line per tool call ------------------------------------------- */
    .tool-entry {
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--border-strong);
        background: var(--surface-muted);
        margin: 0.3rem 0;
    }
    .tool-entry > summary {
        display: flex; align-items: center; gap: 0.55rem;
        padding: 0.42rem 0.7rem; cursor: pointer;
        font-family: var(--font-ui); font-size: var(--fs-ui-sm); color: var(--text-main); list-style: none;
    }
    .tool-entry > summary::-webkit-details-marker { display: none; }
    .tool-entry__label { flex: 1 1 auto; font-size: var(--fs-ui-sm); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .tool-entry__note { flex: 0 0 auto; font-size: var(--fs-meta); color: var(--text-soft); white-space: nowrap; max-width: 45%; overflow: hidden; text-overflow: ellipsis; }
    .tool-entry__mark { flex: 0 0 auto; font-size: var(--fs-meta); font-weight: 700; width: 0.9rem; text-align: center; }
    .tool-entry__mark--ok { color: var(--ok-color); }
    .tool-entry__mark--error { color: var(--error-color); }
    .tool-entry__mark--running {
        border-radius: 50%; width: 0.5rem; height: 0.5rem;
        background: var(--brand); animation: repuragent-pulse 1.4s ease-in-out infinite;
    }
    .tool-entry--error { border-left-color: var(--error-color); background: #fdf4f2; }
    .tool-entry--ok { border-left-color: #cfe0d6; }
    .tool-entry__body { padding: 0 0.7rem 0.6rem; }
    .tool-entry__body:empty { padding: 0; }
    .tool-recovery { font-family: var(--font-ui); font-size: var(--fs-ui-sm); line-height: var(--lh-ui); color: var(--text-soft); margin: 0.5rem 0 0; }
    .tool-code-block {
        background: var(--surface-tint); border: 1px solid var(--border-subtle);
        padding: 0.85rem 1rem; margin-top: 0.6rem; overflow-x: auto;
    }
    .tool-code-label {
        font-family: var(--font-ui); font-size: var(--fs-eyebrow); letter-spacing: 0.1em;
        font-weight: 700; color: var(--text-soft); margin-bottom: 0.45rem;
    }
    .tool-code-block pre { margin: 0; font-size: var(--fs-ui-sm); line-height: var(--lh-copy); background: transparent; white-space: pre; }

    /* ---- Cards ------------------------------------------------------------- */
    .agent-error-card {
        border: 1px solid #dcb0ab; border-left: 4px solid var(--error-color);
        background: #fdf4f2; padding: 0.9rem 1.05rem; margin: 0.9rem 0;
    }
    .agent-error-card__title { font-family: var(--font-ui); font-size: var(--fs-ui); font-weight: 700; color: #8f2d22; margin-bottom: 0.35rem; }
    .agent-error-card__title::before { content: "\\26A0"; margin-right: 0.5rem; }
    .agent-error-card__message { font-family: var(--font-ui); font-size: var(--fs-ui); line-height: var(--lh-copy); }
    .agent-error-card__detail { margin-top: 0.6rem; }
    .agent-error-card__detail summary {
        font-family: var(--font-ui); font-size: var(--fs-eyebrow); font-weight: 700;
        letter-spacing: 0.08em; text-transform: uppercase; color: var(--text-soft); cursor: pointer;
    }
    .agent-error-card__detail pre {
        margin: 0.55rem 0 0; font-family: var(--font-mono); font-size: var(--fs-ui-sm); line-height: var(--lh-ui);
        background: #fff; border: 1px solid #e6cac6; padding: 0.7rem 0.85rem;
        overflow-x: auto; white-space: pre-wrap; color: #6b2b23;
    }
    .agent-notice-card {
        border: 1px solid var(--border-subtle); border-left: 4px solid var(--text-soft);
        background: var(--surface-tint); padding: 0.9rem 1.05rem; margin: 0.9rem 0;
    }
    .agent-notice-card__title { font-family: var(--font-ui); font-weight: 700; margin-bottom: 0.35rem; }
    .agent-notice-card__message { font-family: var(--font-ui); font-size: var(--fs-ui); line-height: var(--lh-copy); color: var(--text-soft); }

    /* ---- The live plan ----------------------------------------------------- */
    #progress-panel { margin: 0.55rem 0 0; }
    .plan-panel { border: 1px solid var(--border-subtle); border-left: 3px solid var(--brand); background: var(--surface-bg); }
    .plan-panel__summary { padding: 0.7rem 0.9rem; cursor: pointer; list-style: none; }
    .plan-panel__summary::-webkit-details-marker { display: none; }
    .plan-panel__head { display: flex; align-items: baseline; gap: 0.6rem; }
    .plan-panel__title {
        font-family: var(--font-ui); font-size: var(--fs-eyebrow); font-weight: 700;
        letter-spacing: 0.1em; text-transform: uppercase; color: var(--text-soft);
    }
    .plan-panel__count { margin-left: auto; font-family: var(--font-ui); font-size: var(--fs-meta); font-weight: 700; color: var(--brand-strong); }
    .plan-panel__goal { font-family: var(--font-ui); font-size: var(--fs-ui); margin-top: 0.25rem; }
    .plan-panel__caption { font-family: var(--font-ui); font-size: var(--fs-meta); color: var(--text-soft); margin-top: 0.15rem; }
    .plan-panel__live {
        display: flex; align-items: center; gap: 0.4rem; margin-top: 0.3rem;
        font-family: var(--font-ui); font-size: var(--fs-meta); color: var(--brand-strong);
    }
    .plan-panel__pulse {
        flex: 0 0 auto; width: 0.5rem; height: 0.5rem; border-radius: 50%;
        background: var(--brand); animation: repuragent-pulse 1.4s ease-in-out infinite;
    }
    .plan-panel__bar {
        margin-top: 0.5rem; height: 4px; background: var(--surface-tint);
        border: 1px solid var(--border-subtle); overflow: hidden;
    }
    .plan-panel__bar > span { display: block; height: 100%; background: var(--brand); transition: width 0.3s ease; }
    .plan-panel__conditions { margin-top: 0.5rem; font-family: var(--font-ui); font-size: var(--fs-meta); color: var(--text-soft); }
    .plan-panel__conditions-title { font-weight: 700; text-transform: uppercase; letter-spacing: 0.07em; font-size: var(--fs-eyebrow); }
    .plan-panel__conditions ul { margin: 0.2rem 0 0 1rem; }
    .plan-panel__steps { list-style: none; margin: 0; padding: 0.2rem 0.9rem 0.8rem; border-top: 1px solid var(--border-subtle); }
    .plan-step { display: flex; gap: 0.55rem; padding: 0.28rem 0; font-family: var(--font-ui); font-size: var(--fs-ui-sm); line-height: var(--lh-ui); }
    .plan-step__mark { flex: 0 0 1rem; text-align: center; font-weight: 700; color: var(--border-strong); }
    .plan-step__body { display: flex; flex-direction: column; min-width: 0; }
    .plan-step__note { font-size: var(--fs-meta); color: var(--text-soft); }
    .plan-step--done .plan-step__mark { color: var(--ok-color); }
    .plan-step--done .plan-step__title { color: var(--text-soft); }
    .plan-step--active .plan-step__mark { color: var(--brand); }
    .plan-step--active .plan-step__title { font-weight: 700; }
    .plan-step--blocked .plan-step__mark { color: var(--error-color); }
    .plan-step--blocked .plan-step__title { color: #8f2d22; }
    .plan-step--skipped .plan-step__title { color: var(--text-soft); text-decoration: line-through; }

    /* ---- The approval gate ------------------------------------------------- */
    /* Deliberately the loudest thing on the page while it is up: the run is
       stopped until the user acts, and the previous build gave no sign of that. */
    #approval-banner { margin: 0.75rem 0 0; }
    .approval-panel {
        border: 1px solid var(--brand); border-left: 4px solid var(--brand);
        background: var(--brand-soft); padding: 0.95rem 1.1rem;
    }
    .approval-panel__title {
        font-family: var(--font-ui); font-weight: 700; font-size: var(--fs-ui);
        color: var(--brand-strong); display: flex; align-items: center; gap: 0.5rem;
    }
    .approval-panel__icon { font-size: var(--fs-lead); }
    .approval-panel__message { font-family: var(--font-ui); font-size: var(--fs-ui); line-height: var(--lh-copy); margin-top: 0.4rem; }
    .approval-panel__hint { font-family: var(--font-ui); font-size: var(--fs-ui-sm); line-height: var(--lh-ui); color: #3c554d; margin-top: 0.45rem; }
    #approval-actions-row { gap: 0.6rem; margin: 0.6rem 0 0; }

    /* ---- Sidebar conversations -------------------------------------------- */
    #conversation-list {
        margin-top: 0.5rem; font-family: var(--font-ui); width: 100%; display: block;
        background: transparent !important; border: 0 !important; box-shadow: none !important; padding: 0 !important;
    }
    #conversation-list > div { background: transparent !important; border: 0 !important; box-shadow: none !important; }
    #conversation-list, #conversation-list > div, #conversation-list-root { width: 100%; box-sizing: border-box; }
    #conversation-list-root { border: 1px solid var(--border-subtle); background: var(--surface-bg); overflow: hidden; }
    .conversation-list__header {
        font-weight: 700; padding: 0.8rem 0.95rem; border-bottom: 1px solid var(--border-subtle);
        text-transform: uppercase; letter-spacing: 0.08em; font-size: var(--fs-eyebrow);
        color: var(--text-soft); background: var(--surface-muted);
    }
    details.conversation-card { border-bottom: 1px solid #eceeed; }
    details.conversation-card:last-child { border-bottom: none; }
    details.conversation-card summary {
        list-style: none; padding: 0.8rem 0.95rem; cursor: pointer;
        background: transparent; transition: background 0.2s ease;
    }
    details.conversation-card summary::-webkit-details-marker { display: none; }
    details.conversation-card.is-active summary { background: var(--brand-soft); }
    .conversation-card__title-row { display: flex; align-items: flex-start; gap: 0.5rem; }
    .conversation-card__title { font-size: var(--fs-ui-sm); font-weight: 600; flex: 1; line-height: var(--lh-tight); min-width: 0; }
    .conversation-card__chevron {
        width: 11px; height: 11px; border-right: 2px solid currentColor; border-bottom: 2px solid currentColor;
        transform: rotate(45deg); transition: transform 0.2s ease; margin-top: 0.3rem; flex: 0 0 11px;
    }
    details.conversation-card[open] .conversation-card__chevron { transform: rotate(-135deg); }
    .conversation-card__tag {
        flex: 0 0 auto; font-size: var(--fs-eyebrow); font-weight: 700; letter-spacing: 0.08em;
        text-transform: uppercase; color: var(--brand-strong); background: var(--surface-tint);
        border: 1px solid var(--border-subtle); padding: 0.1rem 0.32rem; align-self: center;
    }
    .conversation-card__delete {
        border: 1px solid var(--border-strong); padding: 0; font-size: var(--fs-meta);
        background: var(--surface-bg); cursor: pointer; color: var(--text-soft);
        transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
        width: 1.9rem; min-width: 1.9rem; height: 1.9rem;
        display: inline-flex; align-items: center; justify-content: center; flex: 0 0 auto;
    }
    .conversation-card__delete:hover { background: var(--surface-muted); color: var(--error-color); border-color: var(--error-color); }
    .conversation-card__body { background: var(--surface-muted); padding: 0.55rem 0.95rem 0.9rem; border-top: 1px solid #eceeed; }
    .conversation-card__files-container { max-height: 260px; overflow-y: auto; padding-right: 0.25rem; }
    .conversation-card__files { list-style: none; margin: 0; padding: 0; }
    .conversation-card__file-item { font-size: var(--fs-meta); margin-bottom: 0.15rem; }
    .conversation-card__file-name { font-weight: 500; }
    .conversation-card__file-link { font-weight: 600; color: var(--brand); text-decoration: none; word-break: break-all; }
    .conversation-card__file-link:hover, .conversation-card__file-link:focus { text-decoration: underline; }
    .conversation-card__file-more, .conversation-card__empty { font-size: var(--fs-meta); color: var(--text-soft); margin: 0; }
    .conversation-card__thumb { margin: 0.3rem 0 0.45rem; }
    .conversation-card__thumb img {
        max-width: 100%; height: auto; display: block;
        border: 1px solid var(--border-subtle); background: #fff;
    }
    .conversation-card__badge { margin-left: auto; margin-right: 0.2rem; font-size: var(--fs-eyebrow); line-height: 1; flex: 0 0 auto; align-self: center; }
    .conversation-card__badge--running { color: var(--brand); animation: repuragent-pulse 1.4s ease-in-out infinite; }
    .conversation-card__badge--updated { color: var(--ok-color); }

    @keyframes repuragent-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.25; } }
    @media (prefers-reduced-motion: reduce) {
        .conversation-card__badge--running, .tool-entry__mark--running,
        .plan-panel__pulse { animation: none; }
    }

    footer {
        border-top: 1px solid var(--border-subtle);
        margin-top: 1rem !important; padding-top: 0.85rem !important; color: var(--text-soft) !important;
    }

    @media (max-width: 900px) {
        .gradio-container { width: 100vw; padding: 1rem; }
        #app-header { gap: 0.75rem; }
        #header-links { font-size: var(--fs-meta); letter-spacing: 0.05em; }
        #layout-row { gap: 1rem; }
        #sidebar-column { position: static; min-width: 0; width: 100% !important; flex: 1 1 auto !important; max-width: none; }
        #conversation-column { padding: 0.8rem; }
        #chatbot-panel { font-size: var(--fs-body); line-height: var(--lh-copy); }
    }
    """

__all__ = ["APP_CSS", "PRIMARY_FERN", "REPURAGENT_THEME", "SECONDARY_SAGE"]
