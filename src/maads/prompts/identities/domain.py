"""Domain Knowledge Expert — task formatting (persona in backstories/domain.md)."""
from __future__ import annotations

import json
from typing import Any

from maads.prompts.loader import load_agent_prompts
from maads.rag import retrieve_for_state
from maads.state import CrispDMState

DOMAIN_UNDERSTANDING_TASK = """For the dataset "{dataset_name}" with prediction target "{target}",
evaluation metric "{metric}", and ML task type "{ml_task}", produce CRISP-DM 1.1
Determine Business Objectives only.

Work ONLY from these inputs:
  - Feature schema & statistics: {feature_schema}
  - Retrieved domain notes / data dictionary: {domain_corpus}

Do all of the following:
  1. State the business objective in one or two plain sentences.
  2. Define a measurable success criterion tied to "{metric}". If no target
     threshold is provided, set target_value to null and explain the direction.

Rules:
  - Ground every confirmed domain claim in either the domain corpus or
    the feature schema.
  - Anything not directly supported must go under "assumptions" or
    "open_questions".
  - Do not invent columns, files, thresholds, metrics, target meanings, or
    business context.
  - Do not write code or recommend a final model.
  - Do not produce situation_assessment, data_mining_goal, feature_hints, or
    loop_a_recommendation here — those belong to later substeps."""

DOMAIN_UNDERSTANDING_SCHEMA_HINT = """{
  "business_objectives": "string",
  "success_criterion": {
    "metric": "string",
    "target_value": "string|null",
    "direction": "maximize|minimize"
  },
  "assumptions": ["string"],
  "open_questions": ["string"]
}"""

DOMAIN_SITUATION_SCHEMA_HINT = """{
  "situation_assessment": {
    "resources": ["string"],
    "requirements": ["string"],
    "assumptions": ["string"],
    "constraints": ["string"],
    "risks": ["string"],
    "terminology": [{"term": "string", "meaning": "string"}],
    "costs_or_tradeoffs": ["string"],
    "expected_benefits": ["string"]
  },
  "data_description_notes": [
    {"feature": "string", "meaning": "string"}
  ],
  "feature_hints": [
    {
      "feature": "string",
      "rationale": "string",
      "expected_effect": "positive|negative|nonlinear|unknown"
    }
  ],
  "domain_data_quality_flags": [
    {"feature": "string", "risk": "string"}
  ],
  "assumptions": ["string"],
  "open_questions": ["string"]
}"""


def domain_identity(dataset_name: str) -> dict[str, str]:
    """Return role/goal/backstory for a specific dataset."""
    base = load_agent_prompts()["domain"]
    name = dataset_name or "this dataset"
    return {
        "role": base["role"].format(dataset_name=name),
        "goal": base["goal"].format(dataset_name=name),
        "backstory": base["backstory"],
    }


def _feature_schema(state: CrispDMState) -> dict[str, Any]:
    """Config-only schema hints; DU reports live in the scaffold state_view."""
    cfg = state.config
    return {
        "target_column": cfg.target_column,
        "id_column": cfg.id_column,
        "problem_type": cfg.problem_type,
        "evaluation_metric": cfg.evaluation_metric,
        "config_feature_hints": cfg.feature_hints,
    }


def _domain_corpus(state: CrispDMState, passages: list[str]) -> dict[str, Any]:
    cfg = state.config
    return {
        "problem_statement": cfg.problem_statement,
        "kaggle_competition": cfg.kaggle_competition,
        "success_criterion": cfg.success_criterion.model_dump(),
        "retrieved_passages": passages,
    }


def _rag_block(passages: list[str]) -> str:
    if not passages:
        return "(no passages retrieved)"
    return "\n".join(f"- {p}" for p in passages)


def format_domain_understanding_task(state: CrispDMState) -> tuple[str, str]:
    """Build the domain-understanding instruction and JSON schema hint."""
    cfg = state.config
    passages = retrieve_for_state(state)
    instruction = DOMAIN_UNDERSTANDING_TASK.format(
        dataset_name=cfg.case_id,
        target=cfg.target_column,
        metric=cfg.evaluation_metric,
        ml_task=cfg.problem_type,
        feature_schema=json.dumps(_feature_schema(state), indent=2, default=str),
        domain_corpus=json.dumps(_domain_corpus(state, passages), indent=2, default=str),
    )
    return instruction, DOMAIN_UNDERSTANDING_SCHEMA_HINT


DOMAIN_SITUATION_TASK = """CRISP-DM 1.2 Assess Situation for "{dataset_name}".

Expand the situation assessment: resources, requirements, assumptions, constraints,
risks, terminology, costs/tradeoffs, and expected benefits. Also provide concise
feature gloss notes, feature_hints (with expected_effect), and domain data-quality
flags grounded in the schema and retrieved passages.

Retrieved domain passages:
{retrieved_passages}

Do not restate business_objectives or success_criterion unless you must correct them.
Output strict JSON matching the situation schema hint."""

DOMAIN_REFINE_GOALS_TASK = """CRISP-DM 1.3 Determine Data Mining Goals for "{dataset_name}".

Given the data quality report, feature_hints (including na_means_absent), and current
business understanding in the state view, refine data_mining_goals and success criteria
only if quality evidence warrants a change. If the existing success criterion remains
valid, omit success_criterion or echo it unchanged. High missingness on na_means_absent
columns means feature absence, not data corruption — do not recommend Loop A for those
alone. Recommend Loop A only when quality blockers contradict goals or require a
fundamental rethink.

Retrieved domain passages:
{retrieved_passages}

Output JSON with keys: data_mining_goal,
success_criterion (metric, target_value, direction) only when refining,
loop_a_recommendation. Optional: domain_data_quality_flags when refining quality semantics."""


def format_domain_situation_task(state: CrispDMState) -> tuple[str, str]:
    cfg = state.config
    passages = retrieve_for_state(state)
    instruction = DOMAIN_SITUATION_TASK.format(
        dataset_name=cfg.case_id,
        retrieved_passages=_rag_block(passages),
    )
    return instruction, DOMAIN_SITUATION_SCHEMA_HINT


def format_domain_refine_goals_task(state: CrispDMState) -> tuple[str, str]:
    cfg = state.config
    passages = retrieve_for_state(state)
    instruction = DOMAIN_REFINE_GOALS_TASK.format(
        dataset_name=cfg.case_id,
        retrieved_passages=_rag_block(passages),
    )
    hint = """{"data_mining_goal": "string", "success_criterion": {"metric": "string",
    "target_value": "string|null", "direction": "maximize|minimize"},
    "loop_a_recommendation": {"should_trigger": true, "reason": "string"}}"""
    return instruction, hint
