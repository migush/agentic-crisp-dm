"""Tests for phase advance semantics and related fixes."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maads.agents import Plan
from maads.config import load_case_config
from maads.token_budget import TokenBudgetExceeded, check_after_spend
from maads.flow import phase_runner as pr
from maads.flow.phase_runner import PM_DECISION_SUBSTEPS, advance_substep
from maads.paths import repo_root, resolve_path
from maads.state import CrispDMState, Phase
from maads.testing.fake_llm import fake_llm_response
from maads.testing.flow_harness import make_flow, make_run_context


@pytest.fixture
def titanic_state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


def test_advance_recovers_desynced_phase_substep(titanic_state: CrispDMState, tmp_path: Path):
    """Phase/substep mismatch must not crash advance_substep."""
    titanic_state.phase = Phase.DATA_PREPARATION
    titanic_state.substep = "4.1"
    artifact_dir = tmp_path / "artifacts" / "titanic"
    artifact_dir.mkdir(parents=True)
    ctx = make_run_context(titanic_state, artifact_dir)

    advance_substep(ctx)

    assert titanic_state.phase == Phase.MODELING
    assert titanic_state.substep == "4.2"


@patch("maads.agents.run_json_task")
def test_advance_dispatches_current_substep_not_pm_target(
    mock_llm, titanic_state: CrispDMState, tmp_path: Path,
):
    """PM target_substep on advance must not skip the current substep."""
    mock_llm.side_effect = fake_llm_response
    artifact_dir = tmp_path / "artifacts" / "titanic"
    artifact_dir.mkdir(parents=True)
    dispatched: list[str] = []
    orig = pr.run_substep

    def track(ctx, substep: str) -> bool:
        dispatched.append(substep)
        return orig(ctx, substep)

    with patch.object(pr, "run_substep", side_effect=track):
        flow = make_flow(titanic_state, artifact_dir)
        plans = iter([
            Plan(action="advance", target_substep="1.2", reason="PM tried to skip"),
            Plan(action="halt", reason="stop at phase 2 entry"),
        ])
        flow._pm.plan = lambda _state: next(plans)  # type: ignore[method-assign]
        flow.run()

    assert dispatched == ["1.1", "1.2", "1.3", "1.4"]
    assert any("ignored PM target_substep" in e.message for e in titanic_state.log)


def test_substep_1_4_requires_data_mining_goals(titanic_state: CrispDMState):
    assert titanic_state.substep_prereqs_satisfied("1.4") is False
    titanic_state.bu.data_mining_goals = "goal"
    assert titanic_state.substep_prereqs_satisfied("1.4") is True


def test_load_case_config_resolves_data_paths_against_repo_root():
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    root = repo_root()
    assert Path(cfg.data.train_csv).is_absolute()
    assert str(root) in cfg.data.train_csv


def test_resolve_path_from_src_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    src = tmp_path / "src"
    src.mkdir()
    monkeypatch.chdir(src)
    p = resolve_path("data/titanic/train.csv")
    assert p == repo_root() / "data/titanic/train.csv"


def test_token_budget_raises_when_exceeded(
    titanic_state: CrispDMState, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "100")
    titanic_state.token_spend["pm"] = 100
    with pytest.raises(TokenBudgetExceeded, match="token cap"):
        check_after_spend(titanic_state, "pm")


@patch("maads.agents.run_json_task")
def test_pm_plan_halts_on_llm_failure(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    from maads.agents import ProjectManagerAgent
    from maads.crew import CrewKickoffError

    mock_llm.side_effect = CrewKickoffError("timeout")
    pm = ProjectManagerAgent(artifact_dir=tmp_path)
    plan = pm.plan(titanic_state)
    assert plan.action == "halt"
    assert "PM LLM call failed" in plan.reason


@patch("maads.agents.run_json_task")
def test_phase1_domain_llm_substeps_run_in_order(
    mock_llm, titanic_state: CrispDMState, tmp_path: Path,
):
    """Integration: mocked LLM fills domain substeps; flow visits 1.1 before 1.2."""

    def _fake_llm(agent_name, instruction, state, schema_hint="", **kwargs):
        if state.substep == "1.1":
            return {
                "business_objectives": "obj",
                "situation_assessment": {
                    "resources": [],
                    "requirements": [],
                    "assumptions": [],
                    "constraints": [],
                    "risks": [],
                    "terminology": [],
                    "costs_or_tradeoffs": [],
                    "expected_benefits": [],
                },
                "data_mining_goal": "goal",
                "success_criterion": {
                    "metric": "accuracy",
                    "target_value": "0.77",
                    "direction": "maximize",
                },
                "data_description_notes": [],
                "feature_hints": [],
                "domain_data_quality_flags": [],
                "loop_a_recommendation": {"should_trigger": False, "reason": ""},
                "assumptions": [],
                "open_questions": [],
            }
        if agent_name == "pm":
            return {"action": "advance", "target_substep": None, "reason": "ok"}
        return {}

    mock_llm.side_effect = _fake_llm

    artifact_dir = tmp_path / "artifacts" / "titanic"
    artifact_dir.mkdir(parents=True)
    dispatched: list[str] = []
    orig = pr.run_substep

    def track(ctx, substep: str) -> bool:
        dispatched.append(substep)
        return orig(ctx, substep)

    with patch.object(pr, "run_substep", side_effect=track):
        flow = make_flow(titanic_state, artifact_dir)
        plans = []
        for _ in range(30):
            plans.append(Plan(action="advance", reason="ok"))
        plans.append(Plan(action="halt", reason="stop"))
        plan_iter = iter(plans)
        flow._pm.plan = lambda _s: next(plan_iter)  # type: ignore[method-assign]
        flow.run()

    assert dispatched.index("1.1") < dispatched.index("1.2")
    assert "1.1" in dispatched
    assert "1.3" in dispatched
    assert titanic_state.bu.business_objectives == "obj"
    assert titanic_state.bu.data_mining_goals == "goal"


def test_pm_decision_substeps_cover_phase_boundaries_and_loops():
    """PM LLM should fire at phase entry and loop checkpoints, not every substep."""
    assert "1.1" in PM_DECISION_SUBSTEPS
    assert "3.1" not in PM_DECISION_SUBSTEPS  # Loop A only at checkpoint_3_1
    assert "5.1" in PM_DECISION_SUBSTEPS
    assert "5.2" in PM_DECISION_SUBSTEPS
    assert "1.2" not in PM_DECISION_SUBSTEPS
    assert "2.2" not in PM_DECISION_SUBSTEPS
    assert "4.3" not in PM_DECISION_SUBSTEPS


def test_mechanical_advance_skips_pm_llm_between_decision_points(
    titanic_state: CrispDMState, tmp_path: Path,
):
    """Mid-phase substeps must not call the PM LLM."""
    artifact_dir = tmp_path / "artifacts" / "titanic"
    artifact_dir.mkdir(parents=True)
    pm_calls: list[str] = []
    dispatched: list[str] = []
    orig = pr.run_substep

    def track_plan(state: CrispDMState) -> Plan:
        pm_calls.append(state.substep)
        if state.substep == "2.1":
            return Plan(action="halt", reason="stop")
        return Plan(action="advance", reason="ok")

    def track(ctx, substep: str) -> bool:
        dispatched.append(substep)
        return True

    with patch.object(pr, "run_substep", side_effect=track):
        flow = make_flow(titanic_state, artifact_dir)
        flow._pm.plan = track_plan  # type: ignore[method-assign]
        flow.run()

    assert pm_calls == ["1.1", "2.1"]
    assert dispatched.index("1.1") < dispatched.index("1.2") < dispatched.index("1.4")
    assert any("mechanical advance" in e.message for e in titanic_state.log)


def test_pm_llm_called_at_each_decision_substep(
    titanic_state: CrispDMState, tmp_path: Path,
):
    """A full 24-substep walk consults the PM only at decision checkpoints."""
    artifact_dir = tmp_path / "artifacts" / "titanic"
    artifact_dir.mkdir(parents=True)
    pm_calls: list[str] = []

    def track_plan(state: CrispDMState) -> Plan:
        pm_calls.append(state.substep)
        return Plan(action="advance", reason="ok")

    with patch.object(pr, "run_substep", lambda *_a, **_k: True):
        flow = make_flow(titanic_state, artifact_dir)
        flow._pm.plan = track_plan  # type: ignore[method-assign]
        flow.run()

    assert set(pm_calls) == set(PM_DECISION_SUBSTEPS) | {"3.1"}
    # Loop A is decided only at checkpoint_3_1 (phase-2 exit), not again at 3.1 entry.
    assert pm_calls.count("3.1") == 1
    assert "3.1" not in PM_DECISION_SUBSTEPS
    assert len(pm_calls) == len(PM_DECISION_SUBSTEPS) + 1
