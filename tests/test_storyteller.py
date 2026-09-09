"""Tests for storyteller capabilities."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maads.capabilities.storyteller import apply_response, render_final_report_step
from maads.config import load_case_config
from maads.paths import resolve_path
from maads.reports.final_report import build_story_spec_from_bundle, render_final_report_md
from maads.state import CrispDMState, EvaluationBundle, ModelRun
from maads.output_contracts import minimal_storyteller_output, validate_agent_output
from maads.prompts.identities.storyteller import format_storyteller_task


@pytest.fixture
def state_with_bundle() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    state = CrispDMState.from_config(cfg)
    bundle = EvaluationBundle(
        problem_type="binary_classification",
        metrics={"accuracy": 0.82, "balanced_accuracy": 0.81},
        confusion_matrix=[[10, 2], [3, 5]],
        class_labels=dict(cfg.class_labels),
        cv={"mean": 0.80, "std": 0.02, "n_folds": 5},
        figures=["figures/confusion_matrix.png"],
        warnings=["moderate class imbalance detected"],
    )
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=0.80,
        cv_std=0.02,
        assessment="selected",
        evaluation_bundle=bundle,
    )
    state.ev.assessment_of_dm_results = {"meets": True, "cv_score": 0.80, "threshold": 0.77}
    return state


def test_storyteller_output_contract():
    payload = minimal_storyteller_output("6.2")
    assert not validate_agent_output("storyteller", payload, substep="6.2")


def test_format_storyteller_task_includes_assignment_id(
    state_with_bundle: CrispDMState, tmp_path: Path,
):
    state_with_bundle.substep = "6.2"
    instruction, hint = format_storyteller_task(state_with_bundle, tmp_path)
    assert '"assignment_id": "6.2"' in instruction
    assert 'assignment_id must be exactly "6.2"' in hint


def test_apply_response_writes_story_spec(state_with_bundle: CrispDMState, tmp_path: Path):
    data = minimal_storyteller_output("6.2")
    delta = apply_response(data, state_with_bundle, "6.2", tmp_path)
    assert not delta.failed
    assert state_with_bundle.dep.story_spec_path
    spec = json.loads(Path(state_with_bundle.dep.story_spec_path).read_text())
    assert spec.get("storytelling_summary")


def test_render_final_report(state_with_bundle: CrispDMState, tmp_path: Path):
    apply_response(minimal_storyteller_output("6.2"), state_with_bundle, "6.2", tmp_path)
    delta = render_final_report_step(state_with_bundle, tmp_path)
    assert not delta.failed
    assert state_with_bundle.dep.final_report_path
    text = Path(state_with_bundle.dep.final_report_path).read_text()
    assert "Not survived" in text or "Executive Summary" in text
    assert "accuracy" in text.lower()


def test_apply_response_fails_without_bundle(tmp_path: Path):
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    state = CrispDMState.from_config(cfg)
    state.md.chosen_model = ModelRun(technique="logistic_regression", cv_score=0.7)
    delta = apply_response({}, state, "6.2", tmp_path)
    assert delta.failed


def test_build_story_spec_from_bundle(state_with_bundle: CrispDMState):
    spec = build_story_spec_from_bundle(state_with_bundle)
    assert spec["detected_problem_type"] == "binary_classification"
    assert spec["interpretations"]


def test_render_md_includes_metrics(state_with_bundle: CrispDMState):
    spec = build_story_spec_from_bundle(state_with_bundle)
    md = render_final_report_md(state_with_bundle, spec)
    assert "0.8200" in md or "0.82" in md
