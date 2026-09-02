'''Turning raw tool traffic into something a scientist can skim.

A repurposing run makes dozens of tool calls. Rendered literally they are a wall of
identical grey boxes labelled `getDrugsforProteins`, and the reader has to open
every one to find out what happened — including whether it failed. Here each call
becomes one line saying what was done and how it went, with the raw detail one
click away.

Two calls are dropped entirely: `plan_update` and `plan_status` exist to move the
plan file, and the plan is shown live in its own panel, so echoing them into the
transcript would only compete with the panel that shows the same thing.
'''

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Tools whose entire purpose is bookkeeping the progress panel already shows.
SUPPRESSED_TOOLS = {"plan_update", "plan_status"}

MAX_INLINE_RESULT_CHARS = 4000

# The specialist each handoff targets, for a readable delegation line.
_AGENT_LABELS = {
    "research_agent": "Research Agent",
    "prediction_agent": "Prediction Agent",
    "data_agent": "Data Agent",
}

# ADMET endpoints, keyed by tool name, so a call reads as the endpoint rather than
# as a class name.
_ADMET_LABELS = {
    "CYP3A4_classifier": "CYP3A4 inhibition",
    "CYP2C19_classifier": "CYP2C19 inhibition",
    "CYP2D6_classifier": "CYP2D6 inhibition",
    "CYP1A2_classifier": "CYP1A2 inhibition",
    "CYP2C9_classifier": "CYP2C9 inhibition",
    "hERG_classifier": "hERG inhibition",
    "AMES_classifier": "Ames mutagenicity",
    "PGP_classifier": "P-gp inhibition",
    "PAMPA_classifier": "PAMPA permeability",
    "BBB_classifier": "BBB penetration",
    "Solubility_regressor": "aqueous solubility",
    "Lipophilicity_regressor": "lipophilicity (logP)",
}

_KG_LABELS = {
    "search_disease_id": "Resolved disease identifier",
    "create_knowledge_graph": "Built knowledge graph",
    "extract_drugs_from_kg": "Extracted drugs from the graph",
    "extract_proteins_from_kg": "Extracted proteins from the graph",
    "extract_pathways_from_kg": "Extracted pathways from the graph",
    "extract_mechanism_of_actions_from_kg": "Extracted mechanisms from the graph",
    "extract_side_effects_from_kg": "Extracted side effects from the graph",
    "getDrugsforProteins": "Found drugs for proteins",
    "getDrugsforPathways": "Found drugs for pathways",
    "getDrugsforMechanisms": "Found drugs for mechanisms",
}


@dataclass
class ToolView:
    '''How one tool call should appear in the transcript.'''

    label: str
    status: str = "running"  # running | ok | error
    note: str = ""  # short outcome shown on the summary line


def _short_path(value: Any, *, keep: int = 2) -> str:
    '''Trailing path segments: enough to identify a file, short enough to scan.

    Parameters:
    ---------
    value (Any): a path, or anything stringifiable.
    keep (int): how many trailing segments to keep.

    Returns:
    ----------
    text (str): enough of the path to identify the file, short enough to scan.
    '''

    text = str(value or "").strip()
    if not text:
        return ""
    parts = [part for part in Path(text).parts if part not in ("/", "\\")]
    return "/".join(parts[-keep:]) if len(parts) > keep else text


def _first_meaningful_line(code: str) -> str:
    '''A comment or import that hints at what a snippet is for.

    Parameters:
    ---------
    code (str): the snippet the data agent ran.

    Returns:
    ----------
    line (str): a leading comment or import that hints at what the snippet is for.
    '''

    for raw in (code or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            return line.lstrip("#").strip()
        return line
    return ""


def _looks_like_path(text: str) -> bool:
    '''A bare filesystem path, as several tools return.

    Parameters:
    ---------
    text (str): a tool result.

    Returns:
    ----------
    is_path (boolean): True when it is a bare filesystem path, as several tools return.
    '''

    candidate = text.strip()
    return (
        len(candidate) < 400
        and ("/" in candidate or "\\" in candidate)
        and " " not in candidate
        and bool(Path(candidate).suffix)
    )


def _describe_compounds(value: Any) -> str:
    '''How many structures a prediction call was given.

    Parameters:
    ---------
    value (Any): the `smiles_input` a prediction call was given.

    Returns:
    ----------
    text (str): how many structures it covers.
    '''

    if isinstance(value, list):
        return f"{len(value)} compounds"
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        if Path(text).suffix.lower() in {".csv", ".tsv"}:
            return _short_path(text)
        count = len([item for item in text.split(",") if item.strip()])
        return f"{count} compounds" if count > 1 else text[:40]
    return ""


def _parse_args(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        # LangChain sometimes nests the real arguments.
        for key in ("args", "arguments"):
            inner = value.get(key)
            if isinstance(inner, (dict, str)):
                return _parse_args(inner)
    return value


def describe_call(tool_name: Optional[str], args: Any) -> ToolView:
    '''The summary line for a call, before its result is known.

    Parameters:
    ---------
    tool_name (str): the tool being called.
    args (Any): its arguments.

    Returns:
    ----------
    view (ToolView): the one-line summary shown before the result is known.
    '''

    name = (tool_name or "tool").strip()
    parsed = _parse_args(args)
    values = parsed if isinstance(parsed, dict) else {}

    if name.startswith("transfer_to_"):
        target = name.removeprefix("transfer_to_")
        label = f"Delegated to {_AGENT_LABELS.get(target, target.replace('_', ' ').title())}"
        # The brief's objective is the readable half; `task` is the pre-brief argument
        # name, kept so conversations recorded before the structured brief still render.
        objective = str(values.get("objective") or values.get("task") or "").strip()
        if objective:
            label += f" — {objective[:80]}" + ("…" if len(objective) > 80 else "")
        return ToolView(label=label)

    if name in _ADMET_LABELS:
        subject = _describe_compounds(values.get("smiles_input"))
        label = f"Predicted {_ADMET_LABELS[name]}"
        return ToolView(label=f"{label} · {subject}" if subject else label)

    if name == "predict_repurposedrugs":
        subject = _describe_compounds(values.get("query"))
        return ToolView(label="Predicted new indications" + (f" · {subject}" if subject else ""))

    if name in _KG_LABELS:
        detail = ""
        for key in ("disease_name", "disease_id", "kg_path", "pathway_names", "protein_list"):
            value = values.get(key)
            if value:
                detail = _short_path(value) if key == "kg_path" else str(value)[:60]
                break
        return ToolView(label=_KG_LABELS[name] + (f" · {detail}" if detail else ""))

    if name == "literature_search_litsense":
        query = str(values.get("query") or "").strip()
        label = "Searched literature" if name == "literature_search_litsense" else "Searched the literature in depth"
        return ToolView(label=f"{label} — {query[:70]}" if query else label)

    if name == "protocol_search_sop":
        query = str(values.get("query") or "").strip()
        return ToolView(label=f"Searched SOPs — {query[:70]}" if query else "Searched SOPs")

    if name == "annotate_chemicals":
        return ToolView(
            label="Annotated compounds" + (f" · {_short_path(values.get('input_file'))}" if values.get("input_file") else ""),
        )

    if name == "read_files":
        target = _short_path(values.get("file_path"))
        offset, limit = values.get("offset"), values.get("limit")
        span = ""
        if offset or limit:
            start = int(offset or 1)
            span = f" (lines {start}–{start + int(limit) - 1})" if limit else f" (from line {start})"
        return ToolView(label=f"Read {target or 'a file'}{span}")

    if name == "python_executor":
        hint = _first_meaningful_line(str(values.get("code") or ""))
        label = "Ran Python"
        if hint:
            label = f"Ran Python — {hint[:70]}" + ("…" if len(hint) > 70 else "")
        return ToolView(label=label)

    if name == "reset_python_state":
        return ToolView(label="Reset the Python session")

    return ToolView(label=name.replace("_", " ").strip().capitalize() or "Tool call")


def coerce_result(result: Any) -> Any:
    '''Recover a tool's failure envelope from its serialized form.

    LangChain serializes a dict-returning tool into `ToolMessage.content`, so
    `python_executor`'s `{"ok": false, ...}` arrives as a **string**. Without
    decoding it every failure would render with a success tick. The decode is
    deliberately narrow, so a run that legitimately prints JSON is left alone.

    Parameters:
    ---------
    result (Any): a tool result, possibly a serialized failure envelope.

    Returns:
    ----------
    result (any): the recovered value, so a failure is not rendered as ordinary output.
    '''

    if isinstance(result, (dict, list)):
        return result
    if not isinstance(result, str):
        return result
    text = result.strip()
    if not text.startswith("{") or "error_type" not in text[:200]:
        return result
    for parse in (json.loads, ast.literal_eval):
        try:
            decoded = parse(text)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            continue
        if isinstance(decoded, dict) and "ok" in decoded:
            return decoded
    return result


def describe_result(tool_name: Optional[str], result: Any) -> Tuple[str, str]:
    '''`(status, note)` for a finished call, so failure is visible unexpanded.

    Parameters:
    ---------
    tool_name (str): the tool that finished.
    result (Any): what it returned.

    Returns:
    ----------
    summary (tuple): `(status, note)`, so a failure is visible without expanding the entry.
    '''

    name = (tool_name or "").strip()
    result = coerce_result(result)

    if isinstance(result, dict) and result.get("ok") is False:
        kind = str(result.get("error_type") or "Error")
        lines = str(result.get("error") or "").strip().splitlines()
        return "error", f"{kind}: {lines[0][:120]}" if lines else kind

    if isinstance(result, dict):
        # A single-compound prediction comes back as a row.
        for key, value in result.items():
            if key.endswith(("_inhibition", "_mutagenic", "_permeability", "_penetration")):
                return "ok", f"{key.replace('_', ' ')} = {value}"
        if "logS" in result:
            return "ok", f"logS = {result['logS']}"
        if "logP" in result:
            return "ok", f"logP = {result['logP']}"
        return "ok", f"{len(result)} field(s)"

    if isinstance(result, str):
        stripped = result.strip()
        if stripped.startswith("Error:") or stripped.startswith("Plan not updated:"):
            return "error", stripped[:140]
        if "[warning]" in stripped:
            fragment = stripped.split("[warning]", 1)[1].strip()
            return "ok", fragment[:110]
        if name == "python_executor":
            if "[figures]" in stripped:
                count = stripped.count("\n- ")
                return "ok", f"wrote {count} figure{'s' if count != 1 else ''}"
            if not stripped:
                return "ok", "no output"
            return "ok", f"{len(stripped.splitlines())} line(s) of output"
        if name == "read_files":
            if "preview" in stripped[:200].lower():
                return "ok", "preview (file is large)"
            return "ok", f"{len(stripped.splitlines())} line(s)"
        if name.startswith("transfer_to_"):
            return "ok", ""
        lines = stripped.splitlines()
        if len(lines) == 1:
            # A single line is almost always the answer itself — a path, a count, a
            # resolved identifier — so show it rather than counting it.
            single = lines[0].strip()
            if _looks_like_path(single):
                return "ok", _short_path(single)
            return "ok", single[:110] + ("…" if len(single) > 110 else "")
        return "ok", f"{len(lines)} line(s)"

    return "ok", ""


def _clip(text: str) -> str:
    if len(text) <= MAX_INLINE_RESULT_CHARS:
        return text
    head = int(MAX_INLINE_RESULT_CHARS * 0.7)
    tail = MAX_INLINE_RESULT_CHARS - head
    omitted = len(text) - head - tail
    return f"{text[:head]}\n\n… [{omitted:,} characters hidden] …\n\n{text[-tail:]}"


def _code_block(content: str, *, language: str) -> str:
    return (
        "<div class='tool-code-block'>"
        f"<div class='tool-code-label'>{escape(language.upper())}</div>"
        f"<pre>{escape(content)}</pre>"
        "</div>"
    )


def render_call_body(tool_name: Optional[str], args: Any) -> str:
    '''The expandable detail for a call: the code, or the arguments.

    Parameters:
    ---------
    tool_name (str): the tool being called.
    args (Any): its arguments.

    Returns:
    ----------
    html (str): the expandable detail — the code for `python_executor`, otherwise the arguments.
    '''

    name = (tool_name or "").strip()
    parsed = _parse_args(args)

    if name == "python_executor" and isinstance(parsed, dict):
        code = parsed.get("code")
        if isinstance(code, str) and code.strip():
            return _code_block(code.rstrip("\n"), language="python")

    if isinstance(parsed, (dict, list)):
        if not parsed:
            return ""
        return _code_block(json.dumps(parsed, indent=2, default=str), language="json")
    return _code_block(str(parsed), language="text") if parsed else ""


def render_result_body(tool_name: Optional[str], result: Any) -> str:
    '''The expandable detail for a result, bounded so one dump cannot fill the page.

    Parameters:
    ---------
    tool_name (str): the tool that finished.
    result (Any): what it returned.

    Returns:
    ----------
    html (str): the expandable detail, bounded so one dump cannot fill the page.
    '''

    result = coerce_result(result)
    if isinstance(result, dict) and result.get("ok") is False:
        parts = []
        error = str(result.get("error") or "").strip()
        if error:
            parts.append(_code_block(_clip(error), language="text"))
        recovery = str(result.get("recovery") or "").strip()
        if recovery:
            parts.append(f"<p class='tool-recovery'>{escape(recovery)}</p>")
        return "".join(parts)

    if isinstance(result, (dict, list)):
        return _code_block(_clip(json.dumps(result, indent=2, default=str)), language="json")

    text = str(result or "")
    return _code_block(_clip(text), language="text") if text.strip() else ""


STATUS_MARK = {"running": "", "ok": "✓", "error": "✕"}


def render_tool_entry(view: ToolView, *, call_body: str = "", result_body: str = "") -> str:
    '''One collapsible line: what was done, and how it went.

    Parameters:
    ---------
    view (ToolView): the call's summary.
    call_body (str): the expandable call detail.
    result_body (str): the expandable result detail.

    Returns:
    ----------
    html (str): one collapsible line — what was done, and how it went.
    '''

    mark = STATUS_MARK.get(view.status, "")
    note = f"<span class='tool-entry__note'>{escape(view.note)}</span>" if view.note else ""
    mark_html = (
        f"<span class='tool-entry__mark tool-entry__mark--{view.status}'>{mark}</span>"
        if mark
        else "<span class='tool-entry__mark tool-entry__mark--running'></span>"
    )
    body = "".join(part for part in (call_body, result_body) if part)
    # Every entry starts collapsed, failures included. What went wrong is on the
    # summary line — the ✕ and the error text — so the default hides nothing a
    # reader needs to triage; only the trace and the recovery hint are behind the
    # click. Auto-expanding failures was tried and removed: a run that recovers
    # from several errors unfolded into a wall of stack traces.
    return (
        f"<details class='tool-entry tool-entry--{view.status}'>"
        "<summary>"
        f"<span class='tool-entry__label'>{escape(view.label)}</span>"
        f"{note}{mark_html}"
        "</summary>"
        f"<div class='tool-entry__body'>{body}</div>"
        "</details>"
    )


def call_metadata(tool_name: Optional[str], args: Any) -> Dict[str, Any]:
    '''Everything the timeline stores for one call, snapshot-safe.

    Parameters:
    ---------
    tool_name (str): the tool being called.
    args (Any): its arguments.

    Returns:
    ----------
    metadata (dict): everything the timeline stores for the call, JSON-safe so it survives a snapshot.
    '''

    view = describe_call(tool_name, args)
    return {
        "label": view.label,
        "status": "running",
        "note": "",
        "call_body": render_call_body(tool_name, args),
        "result_body": "",
    }


__all__ = [
    "SUPPRESSED_TOOLS",
    "ToolView",
    "call_metadata",
    "coerce_result",
    "describe_call",
    "describe_result",
    "render_call_body",
    "render_result_body",
    "render_tool_entry",
]
