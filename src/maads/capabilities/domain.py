"""Domain Expert capabilities — CRISP-DM-independent."""
from __future__ import annotations

from typing import Any

from maads.deltas import StateDelta
from maads.state import CrispDMState


def format_success_criterion(sc: dict, cfg) -> str:
    metric = sc.get("metric") or cfg.evaluation_metric
    target = sc.get("target_value")
    direction = sc.get("direction") or "maximize"
    if target is not None and str(target).lower() not in {"null", "none"}:
        return f"{metric} {direction} {target}"
    threshold = cfg.success_criterion.threshold
    return f"{metric} {direction} (threshold {threshold})"


def apply_understanding(data: dict, state: CrispDMState) -> StateDelta:
    """Map domain_understanding_task JSON into BusinessUnderstanding fields."""
    cfg = state.config
    sit = data.get("situation_assessment") or {}
    sc = data.get("success_criterion") or {}

    state.bu.background = cfg.problem_statement
    state.bu.business_objectives = (
        data.get("business_objectives")
        or f"Predict {cfg.target_column} as accurately as possible."
    )
    state.bu.business_success_criteria = format_success_criterion(sc, cfg)

    state.bu.inventory_of_resources = {
        "resources": sit.get("resources", []),
        "data": cfg.data.model_dump(),
        "domain_artifacts": {
            "data_description_notes": data.get("data_description_notes", []),
            "feature_hints": data.get("feature_hints", []),
            "domain_data_quality_flags": data.get("domain_data_quality_flags", []),
            "loop_a_recommendation": data.get("loop_a_recommendation"),
        },
    }
    state.bu.requirements_assumptions_constraints = {
        "requirements": sit.get("requirements", []),
        "assumptions": sit.get("assumptions", []) + data.get("assumptions", []),
        "constraints": sit.get("constraints", []),
        "open_questions": data.get("open_questions", []),
        "metric": cfg.evaluation_metric,
        "problem_type": cfg.problem_type,
    }
    state.bu.risks_and_contingencies = sit.get("risks", [])
    state.bu.terminology = {
        t["term"]: t["meaning"]
        for t in sit.get("terminology", [])
        if isinstance(t, dict) and t.get("term")
    }
    state.bu.costs_and_benefits = {
        "costs_or_tradeoffs": sit.get("costs_or_tradeoffs", []),
        "expected_benefits": sit.get("expected_benefits", []),
    }

    state.bu.data_mining_goals = (
        data.get("data_mining_goal")
        or f"Train a {cfg.problem_type} model for {cfg.target_column}."
    )
    state.bu.data_mining_success_criteria = format_success_criterion(sc, cfg)

    return StateDelta([
        "bu.background",
        "bu.business_objectives",
        "bu.business_success_criteria",
        "bu.inventory_of_resources",
        "bu.requirements_assumptions_constraints",
        "bu.risks_and_contingencies",
        "bu.terminology",
        "bu.costs_and_benefits",
        "bu.data_mining_goals",
        "bu.data_mining_success_criteria",
    ])


def apply_situation(data: dict, state: CrispDMState) -> StateDelta:
    sit = data.get("situation_assessment") or data
    fields: list[str] = []
    if sit:
        inv = dict(state.bu.inventory_of_resources or {})
        inv["situation_1_2"] = sit
        artifacts = dict(inv.get("domain_artifacts") or {})
        for key in (
            "data_description_notes",
            "feature_hints",
            "domain_data_quality_flags",
        ):
            if data.get(key) is not None:
                artifacts[key] = data[key]
        if artifacts:
            inv["domain_artifacts"] = artifacts
        state.bu.inventory_of_resources = inv
        if sit.get("risks"):
            state.bu.risks_and_contingencies = sit.get("risks", [])
            fields.append("bu.risks_and_contingencies")
        fields.append("bu.inventory_of_resources")
        # Merge situation assumptions into requirements block when present.
        rac = dict(state.bu.requirements_assumptions_constraints or {})
        if sit.get("requirements") is not None:
            rac["requirements"] = sit.get("requirements", [])
        if sit.get("assumptions") is not None or data.get("assumptions") is not None:
            rac["assumptions"] = list(sit.get("assumptions") or []) + list(
                data.get("assumptions") or []
            )
        if sit.get("constraints") is not None:
            rac["constraints"] = sit.get("constraints", [])
        if data.get("open_questions") is not None:
            rac["open_questions"] = data.get("open_questions", [])
        if rac != (state.bu.requirements_assumptions_constraints or {}):
            state.bu.requirements_assumptions_constraints = rac
            fields.append("bu.requirements_assumptions_constraints")
        if sit.get("terminology"):
            state.bu.terminology = {
                t["term"]: t["meaning"]
                for t in sit.get("terminology", [])
                if isinstance(t, dict) and t.get("term")
            }
            fields.append("bu.terminology")
        if sit.get("costs_or_tradeoffs") is not None or sit.get("expected_benefits") is not None:
            state.bu.costs_and_benefits = {
                "costs_or_tradeoffs": sit.get("costs_or_tradeoffs", []),
                "expected_benefits": sit.get("expected_benefits", []),
            }
            fields.append("bu.costs_and_benefits")
        return StateDelta(fields or ["bu.inventory_of_resources"])
    return StateDelta(notes="1.2 situation assessment empty")


def apply_refine_goals(data: dict, state: CrispDMState) -> StateDelta:
    if data.get("data_mining_goal"):
        state.bu.data_mining_goals = data["data_mining_goal"]
    sc = data.get("success_criterion") or {}
    if sc:
        state.bu.data_mining_success_criteria = format_success_criterion(sc, state.config)
    loop_rec = data.get("loop_a_recommendation")
    domain_flags = data.get("domain_data_quality_flags")
    if loop_rec is not None or domain_flags is not None:
        inv = dict(state.bu.inventory_of_resources or {})
        artifacts = dict(inv.get("domain_artifacts") or {})
        if loop_rec is not None:
            artifacts["loop_a_recommendation"] = loop_rec
        if domain_flags is not None:
            artifacts["domain_data_quality_flags"] = domain_flags
        inv["domain_artifacts"] = artifacts
        state.bu.inventory_of_resources = inv
    fields = ["bu.data_mining_goals", "bu.data_mining_success_criteria"]
    if loop_rec is not None or domain_flags is not None:
        fields.append("bu.inventory_of_resources")
    return StateDelta(fields)
