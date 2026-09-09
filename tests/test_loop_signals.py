"""Real loop signals reach the PM: data-quality blockers and validator findings."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maads.agents import DataEngineerAgent, DataScientistAgent
from maads.config import load_case_config
from maads.deltas import Plan
from maads.flow.phase_runner import apply_loop, validate_phase_exit
from maads.output_contracts import minimal_agent_output
from maads.paths import resolve_path
from maads.state import CrispDMState, ModelRun, Phase
from maads.testing.flow_harness import make_run_context


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")
    monkeypatch.setattr("maads.agents.run_json_task", lambda *a, **k: {})


@pytest.fixture
def state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


def test_documented_cabin_missingness_reaches_pm_quality_gate(
    state: CrispDMState, tmp_path: Path,
):
    state.substep = "2.4"
    de = DataEngineerAgent(artifact_dir=tmp_path)
    delta = de.act(state)
    assert not delta.failed, delta.notes
    report = state.du.data_quality_report or {}
    blockers = report.get("blockers") or []
    tolerable = report.get("tolerable") or []
    assert not any("Cabin" in b for b in blockers)
    assert any("Cabin" in t for t in tolerable)
    pm_view = state.view_for("pm")
    gate = pm_view["quality_gate"]
    assert "Cabin" in gate["na_means_absent"]
    assert "Cabin" in gate["high_missing"]
    assert "quality_gate" in pm_view


def test_validator_findings_populate_and_reach_pm(state: CrispDMState, tmp_path: Path):
    ctx = make_run_context(state, tmp_path)
    state.dp.dataset = {"train": "/no/such/train.parquet", "test": "/no/such/test.parquet"}
    state.dp.derived_attributes = {"items": ["FamilySize"]}

    validate_phase_exit(state, Phase.DATA_PREPARATION)

    assert state.validator_findings
    pm_view = state.view_for("pm")
    assert pm_view["validator_findings"]


def test_fire_loop_tolerates_stringy_phase(tmp_path: Path, state: CrispDMState):
    """Regression: PMs often return loop_to_phase as a string; Phase() is an IntEnum."""
    ctx = make_run_context(state, tmp_path)
    state.phase = Phase.MODELING
    apply_loop(
        ctx,
        Plan(action="loop_back", loop_to_phase=3, loop_label="B", reason="stringy target"),
    )
    assert state.phase == Phase.DATA_PREPARATION
    assert state.loop_history[-1].to_phase == 3


def test_fire_loop_clears_validator_findings(tmp_path: Path, state: CrispDMState):
    ctx = make_run_context(state, tmp_path)
    state.validator_findings = ["some deficit"]
    state.ev.assessment_of_dm_results = {"meets": False, "cv_score": 0.70}
    state.phase = Phase.MODELING
    apply_loop(
        ctx,
        Plan(
            action="loop_back",
            loop_to_phase=int(Phase.DATA_PREPARATION),
            loop_label="B",
            reason="addressing prep deficit",
        ),
    )
    assert state.validator_findings == []
    assert state.ev.assessment_of_dm_results is None
    assert state.loop_history and state.loop_history[-1].label == "B"


def test_degraded_flags_reach_pm_view(state: CrispDMState):
    state.record_degraded("data_engineer@3.2: baseline fallback")
    pm_view = state.view_for("pm")
    assert pm_view["degraded_flags"]


def test_suggested_action_loop_b_on_validator_findings(state: CrispDMState):
    state.validator_findings = ["missing artifact"]
    state.substep = "5.1"
    suggested = state._suggested_pm_action()
    assert suggested is not None
    assert suggested["loop_label"] == "B"


def test_business_goal_met_unknown_at_5_1_before_assessment(state: CrispDMState):
    thr = state.config.success_criterion.threshold
    state.substep = "5.1"
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=thr + 0.05,
        description="strong model",
    )
    state.ev.assessment_of_dm_results = None
    pm_view = state.view_for("pm")
    assert pm_view["business_goal_met"] is None
    assert state._suggested_pm_action() is None


def test_business_goal_met_unknown_at_5_1_with_stale_assessment(state: CrispDMState):
    state.substep = "5.1"
    state.ev.assessment_of_dm_results = {
        "meets": False,
        "cv_score": 0.70,
        "threshold": 0.77,
    }
    pm_view = state.view_for("pm")
    assert pm_view["business_goal_met"] is None
    assert state._suggested_pm_action() is None


def test_loop_c_not_suggested_after_already_fired(state: CrispDMState):
    state.substep = "5.2"
    state.ev.assessment_of_dm_results = {
        "meets": False,
        "cv_score": 0.70,
        "threshold": 0.77,
    }
    state.record_loop("C", 5, 1, "first Loop C")
    assert state._suggested_pm_action() is None


def test_business_goal_met_derived_from_cv_at_5_2_without_assessment(state: CrispDMState):
    thr = state.config.success_criterion.threshold
    state.substep = "5.2"
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=thr + 0.05,
        description="strong model",
    )
    state.ev.assessment_of_dm_results = None
    pm_view = state.view_for("pm")
    assert pm_view["business_goal_met"] is True


def test_ds_5_1_sets_assessment_from_chosen_model(tmp_path: Path, state: CrispDMState):
    thr = state.config.success_criterion.threshold
    state.phase = Phase.EVALUATION
    state.substep = "5.1"
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=thr + 0.05,
        description="strong model",
    )
    ds = DataScientistAgent(artifact_dir=tmp_path)
    payload = minimal_agent_output("data_scientist", "5.1", summary="DS 5.1")
    with patch.object(ds._crew, "kickoff_substep", return_value=payload):
        ds.act(state)
    assessment = state.ev.assessment_of_dm_results or {}
    assert assessment.get("meets") is True
    assert assessment.get("cv_score") == thr + 0.05


def test_ds_5_1_rmse_log_meets_on_minimize(tmp_path: Path):
    cfg = load_case_config(resolve_path("configs/house_prices.yaml"))
    state = CrispDMState.from_config(cfg)
    thr = cfg.success_criterion.threshold
    state.phase = Phase.EVALUATION
    state.substep = "5.1"
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=thr - 0.02,
        description="strong model",
    )
    ds = DataScientistAgent(artifact_dir=tmp_path)
    payload = minimal_agent_output(
        "data_scientist",
        "5.1",
        summary="DS 5.1",
        state_updates={
            "ev": {
                "assessment_of_dm_results": {
                    "metric": "rmse_log",
                    "achieved_score": thr - 0.02,
                    "success_criterion_met": True,
                    "threshold": thr,
                },
            },
        },
    )
    with patch.object(ds._crew, "kickoff_substep", return_value=payload):
        ds.act(state)
    assessment = state.ev.assessment_of_dm_results or {}
    assert assessment.get("meets") is True
    assert assessment.get("success_criterion_met") is True
    assert assessment.get("cv_score") == thr - 0.02


def test_inspect_dataset_reports_column_diff(state: CrispDMState, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    describe_code = '''```python
import pandas as pd, json
df = pd.read_csv(TRAIN_CSV)
print(json.dumps({
    "n_rows": int(len(df)), "n_cols": int(df.shape[1]),
    "columns": list(df.columns),
    "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    "missing": {c: int(df[c].isna().sum()) for c in df.columns},
}))
```'''
    monkeypatch.setattr("maads.crew.run_text_task", lambda *a, **k: describe_code)
    de = DataEngineerAgent(artifact_dir=tmp_path)
    state.substep = "2.2"
    delta = de.act(state)
    assert not delta.failed, delta.notes
    assert state.du.data_description_report
