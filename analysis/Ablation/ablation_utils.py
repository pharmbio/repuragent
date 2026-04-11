from __future__ import annotations

import json
from statistics import mean
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field
import sys
from pathlib import Path
import numpy as np

CURRENT_DIR = Path.cwd().resolve()
PROJECT_ROOT = CURRENT_DIR
while PROJECT_ROOT != PROJECT_ROOT.parent and not ((PROJECT_ROOT / "app").is_dir() and (PROJECT_ROOT / "backend").is_dir()):
    PROJECT_ROOT = PROJECT_ROOT.parent

if not ((PROJECT_ROOT / "app").is_dir() and (PROJECT_ROOT / "backend").is_dir()):
    raise RuntimeError(f"Could not locate project root from {CURRENT_DIR}")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from persistence.memory.episodic_memory.episodic_learning import get_orchestrator
from backend.utils.research_tools import literature_search_pubmed, protocol_search_sop
from core.prompts.prompts import PLANNING_SYSTEM_PROMPT_ver3
from langchain.chat_models import init_chat_model
from app.config import OPENAI_API_KEY

from itertools import combinations
from scipy.stats import kruskal, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import pandas as pd


# ======================= CONFIG ======================= 

PLANNER_MODEL_NAME = "gpt-4o"
JUDGE_MODEL_NAME = "gpt-4o"
assert OPENAI_API_KEY, "OPENAI_API_KEY is required to execute this ablation notebook."

planner_llm = init_chat_model(
    PLANNER_MODEL_NAME,
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

judge_llm = init_chat_model(
    JUDGE_MODEL_NAME,
    model_provider="openai",
    api_key=OPENAI_API_KEY,
    temperature=0,
)

orchestrator = get_orchestrator()

PROTOCOL_CACHE = {}
LITERATURE_CACHE = {}
EPISODIC_CACHE = {}

def truncate_text(text: str, max_chars: int) -> str:
    text = text.strip()
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated for prompt efficiency]"


def get_protocol_context(task: dict) -> dict:
    task_id = task["task_id"]
    if task_id not in PROTOCOL_CACHE:
        text = protocol_search_sop.invoke({"query": task["protocol_query"]})
        PROTOCOL_CACHE[task_id] = {
            "query": task["protocol_query"],
            "text": truncate_text(text, max_chars=3500),
        }
    return PROTOCOL_CACHE[task_id]


def get_literature_context(task: dict) -> dict:
    task_id = task["task_id"]
    if task_id not in LITERATURE_CACHE:
        text = literature_search_pubmed.invoke({"query": task["literature_query"], "limit": 4})
        LITERATURE_CACHE[task_id] = {
            "query": task["literature_query"],
            "text": truncate_text(text, max_chars=3000),
        }
    return LITERATURE_CACHE[task_id]


def get_episodic_examples(task: dict, max_examples: int = 2) -> list[dict]:
    task_id = task["task_id"]
    if task_id not in EPISODIC_CACHE:
        ctx = orchestrator.get_episodic_context(task["request"])
        EPISODIC_CACHE[task_id] = ctx.get("examples", [])[:max_examples]
    return EPISODIC_CACHE[task_id]


def format_episodic_examples(task: dict) -> str:
    examples = get_episodic_examples(task)
    if not examples:
        return ""
    blocks = []
    for example in examples:
        decomposition = example.get("final_decomposition") or example.get("initial_decomposition") or ""
        blocks.append(
            f"Input: {example.get('task', '')}\n```\n📋 BREAKDOWN: {decomposition}\n\n📋 Note for success: {example.get('notes', '')}\n```"
        )
    return "\n\nRelevant past planning examples:\n" + "\n\n".join(blocks)



def build_planner_prompt(task: dict, use_protocol: int, use_literature: int, use_episodic: int):
    context_sections = []
    retrieval_meta = {
        "protocol_query": None,
        "literature_query": None,
        "episodic_examples_used": 0,
    }

    if use_protocol:
        protocol_ctx = get_protocol_context(task)
        retrieval_meta.update(
            {
                "protocol_query": protocol_ctx["query"],
            }
        )
        context_sections.append("SOP / protocol context retrieved for planning:\n" + protocol_ctx["text"])

    if use_literature:
        literature_ctx = get_literature_context(task)
        retrieval_meta.update(
            {
                "literature_query": literature_ctx["query"],
            }
        )
        context_sections.append("Scientific literature context retrieved for planning:\n" + literature_ctx["text"])

    episodic_block = ""
    if use_episodic:
        examples = get_episodic_examples(task)
        retrieval_meta["episodic_examples_used"] = len(examples)
        episodic_block = format_episodic_examples(task)

    context_text = "\n\n".join(context_sections) if context_sections else " "

    prompt = f"""{PLANNING_SYSTEM_PROMPT_ver3}

# OUTPUT ADAPTATION FOR REPRODUCIBLE EVALUATION
Return the plan as a structured object instead of free text.
- goal: concise restatement of the planning objective
- steps: ordered executable sub-tasks
- note_for_success: critical success factors

Each step must include:
- step_number
- title
- objective
- assigned_agent
- expected_output

The ordered steps should correspond to the same BREAKDOWN logic that the live planner would present, and note_for_success should capture the same content that would appear after `📋 Note for success:`.

# RETRIEVED CONTEXT FOR THIS CONDITION
{context_text}

{episodic_block}

# USER REQUEST
{task['request']}
"""
    return prompt, retrieval_meta, context_text + episodic_block




# ======================= Plan parsing and judement======================= 
class PlanStep(BaseModel):
    step_number: int = Field(..., ge=1, description="1-based step order in the plan.")
    title: str = Field(..., min_length=3, description="Short title for the step.")
    objective: str = Field(..., min_length=10, description="What this step accomplishes.")
    assigned_agent: str = Field(
        ...,
        min_length=2,
        description="Best-fit agent or role for this step, such as data_agent, research_agent, prediction_agent, or report_agent.",
    )
    expected_output: str = Field(
        ...,
        min_length=5,
        description="Concrete artifact or result expected from the step.",
    )


class PlanningPlan(BaseModel):
    goal: str = Field(..., min_length=10, description="Restatement of the planning objective.")
    steps: list[PlanStep] = Field(
        ...,
        min_length=4,
        description="Ordered actionable plan steps.",
    )
    note_for_success: str = Field(
        ...,
        min_length=10,
        description="Critical success factors or cautions for executing the plan well.",
    )


class JudgementScore(BaseModel):
    score: int = Field(..., ge=1, le=5)
    rationale: str = Field(..., min_length=10)


class PlanJudgement(BaseModel):
    structure_completeness: JudgementScore
    task_alignment: JudgementScore
    executability: JudgementScore
    scientific_rigor: JudgementScore
    context_use: JudgementScore
    specificity: JudgementScore
    overall_quality: JudgementScore
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)

    def composite_score(self) -> float:
        components = [
            self.structure_completeness.score,
            self.task_alignment.score,
            self.executability.score,
            self.scientific_rigor.score,
            self.context_use.score,
            self.specificity.score,
        ]
        return mean(components) / 5.0


def _coerce_pydantic(model_cls: type[BaseModel], payload: Any) -> BaseModel:
    if isinstance(payload, model_cls):
        return payload
    if isinstance(payload, dict):
        return model_cls.model_validate(payload)
    if isinstance(payload, str):
        return model_cls.model_validate_json(payload)
    if hasattr(payload, "content"):
        content = getattr(payload, "content")
        if isinstance(content, str):
            return model_cls.model_validate_json(content)
        if isinstance(content, dict):
            return model_cls.model_validate(content)
    raise TypeError(f"Cannot coerce payload of type {type(payload)!r} into {model_cls.__name__}")


def render_plan_markdown(plan: PlanningPlan) -> str:
    lines = ["📋 BREAKDOWN:"]
    for step in plan.steps:
        lines.append(
            f"{step.step_number}. [{step.assigned_agent}] {step.title}: {step.objective} "
            f"Output: {step.expected_output}"
        )
    lines.append("")
    lines.append(f"📋 Note for success: {plan.note_for_success}")
    lines.append("")
    lines.append(
        "Please review this plan. You can ask for changes, provide additional requirements, or tell me if you approve."
    )
    return "\n".join(lines)


def plan_to_json(plan: PlanningPlan) -> str:
    return json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)


def judgement_to_json(judgement: PlanJudgement) -> str:
    return json.dumps(judgement.model_dump(mode="json"), ensure_ascii=False)


def run_structured_planner(llm, prompt: str) -> PlanningPlan:
    structured_llm = llm.with_structured_output(PlanningPlan)
    result = structured_llm.invoke(prompt)
    return _coerce_pydantic(PlanningPlan, result)


def _format_reference_steps(reference_steps: Iterable[str]) -> str:
    return "\n".join(f"- {step}" for step in reference_steps)


def _format_component_flags(use_protocol: int, use_literature: int, use_episodic: int) -> str:
    return "\n".join(
        [
            f"- protocol_search_sop enabled: {'yes' if use_protocol else 'no'}",
            f"- literature_search_pubmed enabled: {'yes' if use_literature else 'no'}",
            f"- episodic_memory enabled: {'yes' if use_episodic else 'no'}",
        ]
    )


def judge_plan(
    llm,
    *,
    task_request: str,
    reference_steps: Iterable[str],
    plan: PlanningPlan,
    available_context: str,
    use_protocol: int,
    use_literature: int,
    use_episodic: int,
) -> PlanJudgement:
    prompt = f"""You are a strict reviewer evaluating the quality of an initial scientific planning output for a drug-repurposing workflow.

Score each criterion from 1 to 5.
- 1 = poor / seriously deficient
- 2 = weak
- 3 = adequate
- 4 = strong
- 5 = excellent

Use the full scale. Be critical, not polite.

Evaluation criteria:
1. structure_completeness: Is the plan coherent, ordered, and cover the task end-to-end?
2. task_alignment: How well it actually match the reference workflow?
3. executability: Are the steps concrete enough to execute with realistic handoffs and outputs?
4. scientific_rigor: Does the plan reflect sound scientific reasoning for drug repurposing?
5. context_use: Given the enabled components and provided context, does the plan use that context appropriately and productively?
6. specificity: Is the plan specific and non-generic rather than boilerplate?
7. overall_quality: Your holistic view after considering all criteria above.


## Rubrics

### 1. `structure_completeness`
**Anchors:**
- **1** — No discernible order; major phases (e.g. target ID, candidate selection, validation) are missing or jumbled
- **3** — Has a recognizable structure but skips one or more necessary phases, or ordering is inconsistent
- **5** — Phases are complete, sequenced correctly, and clearly connected from problem framing to final output

---

### 2. `task_alignment`
**Anchors:**
- **1** — Plan addresses a different problem or ignores core workflow requirements
- **3** — Covers the main objective but misses or misframes specific workflow steps or constraints
- **5** — Every step maps cleanly to the reference workflow; scope and framing are fully consistent

---

### 3. `executability`
**Anchors:**
- **1** — Steps are vague actions ("analyze data," "review literature") with no named tools, methods, or outputs
- **3** — Steps name outputs but leave tool choice or handoff conditions underspecified
- **5** — Each step names a tool or method, specifies a concrete output artifact, and clearly defines the condition for passing to the next step

---

### 4. `scientific_rigor`
**Anchors:**
- **1** — Scientifically unsound: wrong assumptions, inappropriate methods for the domain, or missing essential validation logic
- **3** — Scientifically reasonable overall, but with at least one notable methodological gap or unjustified assumption
- **5** — Reflects current best practices: appropriate target/indication rationale, sound prioritization logic, realistic validation strategy

---

### 5. `context_use`
**Anchors:**
- **1** — Context ignored entirely, or hallucinated sources/evidence introduced that were not provided
- **3** — Context is acknowledged but used superficially or inconsistently
- **5** — Context is tightly integrated: specific provided details inform decisions and are cited accurately; nothing unsupported is invented

---

### 6. `specificity`
**Anchors:**
- **1** — Fully generic: could describe any drug-repurposing project with no modification; no disease, target, or dataset specifics
- **3** — Some specific elements, but key sections read as templated filler
- **5** — Highly specific throughout: disease, target rationale, data sources, thresholds, and methods are named and tailored to this request

---

### 7. `overall_quality`
**Anchors:**
- **1** — Plan is not usable; would require complete rework
- **3** — Plan is a reasonable starting point but needs significant revision before execution
- **5** — Plan could be handed directly to an expert team and executed with minimal clarification

Important judging rule:
- Do NOT penalize the plan for not using protocol, literature, or episodic context when that component was disabled.
- When no external context was provided, score context_use based on whether the plan stays appropriately grounded in the request without inventing unsupported evidence.

Task request:
{task_request}

Reference workflow elements:
{_format_reference_steps(reference_steps)}

Enabled planning components:
{_format_component_flags(use_protocol, use_literature, use_episodic)}

Provided retrieved context for the planner:
{available_context}

Structured plan to evaluate:
{plan.model_dump_json(indent=2)}
"""
    structured_llm = llm.with_structured_output(PlanJudgement)
    result = structured_llm.invoke(prompt)
    return _coerce_pydantic(PlanJudgement, result)



#  ======================= Main function ======================= 

def run_condition(task: dict, use_protocol: int, use_literature: int, use_episodic: int) -> dict:
    prompt, retrieval_meta, judge_context = build_planner_prompt(task, use_protocol, use_literature, use_episodic)

    plan = run_structured_planner(planner_llm, prompt)

    judgement = judge_plan(
        judge_llm,
        task_request=task["request"],
        reference_steps=task["reference_steps"],
        plan=plan,
        available_context=judge_context,
        use_protocol=use_protocol,
        use_literature=use_literature,
        use_episodic=use_episodic,
    )

    return {
        "plan": plan,
        "plan_text": render_plan_markdown(plan),
        "plan_json": plan_to_json(plan),
        "judgement": judgement,
        "judgement_json": judgement_to_json(judgement),
        "quality_score": judgement.composite_score(),
        **retrieval_meta,
    }



def nonparametric_config_tests(
    df: pd.DataFrame,
    group_col: str = "config_label",
    metric_col: str = "quality_score",
    p_adjust: str = "holm",
):
    """
    Non-parametric testing for ablation configurations:
      1) Kruskal-Wallis across all configs
      2) Pairwise Mann-Whitney U for each config pair
      3) Holm correction across all pairwise p-values
    """
    if group_col not in df.columns:
        raise ValueError(f"Missing grouping column: {group_col}")
    if metric_col not in df.columns:
        raise ValueError(f"Missing metric column: {metric_col}")

    groups = sorted(df[group_col].dropna().unique().tolist())
    grouped_values = [
        df.loc[df[group_col] == g, metric_col].dropna().values
        for g in groups
    ]

    _, kw_p = kruskal(*grouped_values)
    kw_df = pd.DataFrame(
        [{
            "metric": metric_col,
            "n_groups": len(groups),
            "kruskal_p": float(kw_p),
        }]
    )

    pairs, pvals = [], []
    for g1, g2 in combinations(groups, 2):
        x = df.loc[df[group_col] == g1, metric_col].dropna().values
        y = df.loc[df[group_col] == g2, metric_col].dropna().values
        _, p = mannwhitneyu(x, y, alternative="two-sided")
        pairs.append((g1, g2, len(x), len(y), float(np.median(x)), float(np.median(y))))
        pvals.append(p)

    pvals = np.asarray(pvals, dtype=float)
    reject, p_adj, _, _ = multipletests(pvals, method=p_adjust)

    pairwise_df = pd.DataFrame(
        [
            {
                "config_a": g1,
                "config_b": g2,
                "n_a": n1,
                "n_b": n2,
                "median_a": med1,
                "median_b": med2,
                "median_diff_a_minus_b": med1 - med2,
                "p_raw": float(pr),
                "p_adj": float(pa),
                "significant": bool(rj),
                "p_adjust": p_adjust,
            }
            for (g1, g2, n1, n2, med1, med2), pr, pa, rj in zip(pairs, pvals, p_adj, reject)
        ]
    ).sort_values(["p_adj", "p_raw", "config_a", "config_b"]).reset_index(drop=True)

    return kw_df, pairwise_df