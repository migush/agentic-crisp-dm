"""Path-coverage scenarios: mocked LLM, real sklearn snippets, no hour-long runs."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maads.agents import Plan, ProjectManagerAgent
from maads.config import load_case_config
from maads.crew import CrewKickoffError
from maads.flow import phase_runner as pr
from maads.paths import resolve_path
from maads.state import SUBSTEPS, CrispDMState, ModelRun, Phase
from maads.token_budget import HALT_REASON
from maads.testing.fake_llm import fake_llm_response
from maads.testing.flow_harness import make_flow, make_run_context


@pytest.fixture
def titanic_state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


@pytest.fixture(autouse=True)
def fast_run(monkeypatch: pytest.MonkeyPatch):
    """Disable trace I/O during coverage runs."""
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


def _artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts" / "titanic"
    d.mkdir(parents=True)
    return d


def _all_substeps() -> list[str]:
    out: list[str] = []
    for phase in Phase:
        out.extend(SUBSTEPS[phase])
    return out


@patch("maads.agents.run_json_task")
def test_happy_path_all_substeps(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    dispatched: list[str] = []
    orig = pr.run_substep

    def track(ctx, substep: str) -> bool:
        dispatched.append(substep)
        return orig(ctx, substep)

    with patch.object(pr, "run_substep", side_effect=track):
        flow = make_flow(titanic_state, artifact_dir)
        plans = [Plan(action="advance", reason="ok") for _ in range(60)]
        plan_iter = iter(plans)
        flow._pm.plan = lambda _s: next(plan_iter, Plan(action="halt", reason="done"))  # type: ignore[method-assign]
        flow.run()

    assert titanic_state.dep.submission_path
    assert titanic_state.dep.final_report_path
    assert titanic_state.dep.experience_documentation
    assert set(dispatched) == set(_all_substeps())
    assert dispatched == _all_substeps()


@patch("maads.agents.run_json_task")
def test_loop_back_fires_and_records_history(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)

    plans = iter([
        Plan(action="advance", reason="run 1.1"),
        Plan(action="advance", reason="run 2.1"),
        Plan(
            action="loop_back",
            loop_label="B",
            loop_to_phase=3,
            target_substep="3.1",
            reason="revisit prep",
        ),
        Plan(action="halt", reason="stop after loop"),
    ])
    flow._pm.plan = lambda _s: next(plans)  # type: ignore[method-assign]
    flow.run()

    assert titanic_state.loop_history
    assert titanic_state.loop_history[0].label == "B"
    # Loop B returns to phase 3; without a PM gate at 3.1 entry the phase runs,
    # then the next decision point (4.1) receives the halt plan.
    assert titanic_state.halted
    assert titanic_state.halt_reason == "stop after loop"
    assert titanic_state.phase == Phase.MODELING
    assert titanic_state.substep == "4.1"


@patch("maads.agents.run_json_task")
def test_pm_halt_stops_run(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    flow._pm.plan = lambda _s: Plan(action="halt", reason="user stop")  # type: ignore[method-assign]
    flow.run()

    assert titanic_state.halted
    assert titanic_state.halt_reason == "user stop"


@patch("maads.agents.run_json_task")
def test_dispatch_exception_halts(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)

    def boom(_ctx, _substep: str) -> None:
        raise RuntimeError("snippet failed")

    with patch.object(pr, "run_substep", side_effect=boom):
        flow = make_flow(titanic_state, artifact_dir)
        flow._pm.plan = lambda _s: Plan(action="advance", reason="ok")  # type: ignore[method-assign]
        flow.run()

    assert titanic_state.halted
    assert "dispatch failed" in (titanic_state.halt_reason or "")


@patch("maads.agents.run_json_task")
def test_hard_cap_halts(mock_llm, titanic_state: CrispDMState, tmp_path: Path, monkeypatch):
    mock_llm.side_effect = fake_llm_response
    monkeypatch.setattr(pr, "MAX_PHASE_TRANSITIONS", 0)
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    flow._pm.plan = lambda _s: Plan(action="advance", reason="ok")  # type: ignore[method-assign]
    flow.run()

    assert titanic_state.halted
    assert titanic_state.halt_reason == "hard cap exceeded"


@patch("maads.agents.run_json_task")
def test_token_budget_halts(mock_llm, titanic_state: CrispDMState, tmp_path: Path, monkeypatch):
    mock_llm.side_effect = fake_llm_response
    monkeypatch.setenv("MAX_TOKENS_PER_RUN", "100")
    titanic_state.token_spend["pm"] = 100
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    flow._pm.plan = lambda _s: Plan(action="advance", reason="ok")  # type: ignore[method-assign]
    flow.run()

    assert titanic_state.halted
    assert titanic_state.halt_reason == HALT_REASON


@patch("maads.agents.run_json_task")
def test_prereq_skip_does_not_run_agent(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    titanic_state.substep = "1.4"
    assert titanic_state.substep_prereqs_satisfied("1.4") is False
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    plans = iter([
        Plan(action="advance", reason="try 1.4"),
        Plan(action="halt", reason="stop"),
    ])
    flow._pm.plan = lambda _s: next(plans)  # type: ignore[method-assign]
    flow.run()

    assert not titanic_state.bu.project_plan
    assert any("prereqs not satisfied for 1.4" in e.message for e in titanic_state.log)


@patch("maads.agents.run_json_task")
def test_pm_plan_halts_on_llm_failure(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = CrewKickoffError("timeout")
    pm = ProjectManagerAgent(artifact_dir=_artifact_dir(tmp_path))
    plan = pm.plan(titanic_state)
    assert plan.action == "halt"
    assert "PM LLM call failed" in plan.reason


@patch("maads.agents.run_json_task")
def test_pm_halt_before_6_4_runs_review(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    """PM must not skip 6.4 when submission and report already exist."""
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    titanic_state.phase = Phase.DEPLOYMENT
    titanic_state.substep = "6.4"
    titanic_state.dep.submission_path = str(artifact_dir / "submission.csv")
    titanic_state.dep.final_report_path = str(artifact_dir / "final_report.md")
    run = ModelRun(technique="rf", cv_score=0.82, assessment="ok")
    titanic_state.md.chosen_model = run
    titanic_state.ev.assessment_of_dm_results = {"meets": True, "cv_score": 0.82}
    ctx = make_run_context(titanic_state, artifact_dir)
    ctx.pm.plan = lambda _s: Plan(  # type: ignore[method-assign]
        action="halt",
        reason="Phase 6 completion requirements are met: phase_6_ready is true.",
    )
    pr.run_phase_substeps(ctx, Phase.DEPLOYMENT)

    assert titanic_state.dep.experience_documentation
    assert any("ignored PM halt" in e.message for e in titanic_state.log)
    assert titanic_state.halted
    assert titanic_state.halt_reason == "completed phase 6"


@patch("maads.agents.run_json_task")
def test_loop_blocked_when_cap_reached(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    flow._ctx.inner_loop_count = 3

    plans = iter([
        Plan(
            action="loop_back",
            loop_label="B",
            loop_to_phase=3,
            reason="should block",
        ),
        Plan(action="halt", reason="stop"),
    ])
    flow._pm.plan = lambda _s: next(plans)  # type: ignore[method-assign]
    flow.run()

    assert not titanic_state.loop_history
    assert any("loop blocked" in e.message for e in titanic_state.log)
    assert titanic_state.halted
    assert "recovery budget exhausted" in (titanic_state.halt_reason or "")


@patch("maads.agents.run_json_task")
def test_suggested_action_no_auto_loop_a_on_quality_blockers(mock_llm, titanic_state: CrispDMState):
    """Loop A at 3.1 is semantic (PM agent); no mechanical suggested_action."""
    mock_llm.side_effect = fake_llm_response
    titanic_state.substep = "3.1"
    titanic_state.du.data_quality_report = {"blockers": ["Cabin >70% missing"]}
    suggested = titanic_state._suggested_pm_action()
    assert suggested is None


@patch("maads.agents.run_json_task")
def test_loop_c_suggested_on_unmet_business_goal(mock_llm, tmp_path: Path):
    mock_llm.side_effect = fake_llm_response
    cfg = load_case_config(resolve_path("configs/titanic_loopdemo.yaml"))
    state = CrispDMState.from_config(cfg)
    state.substep = "5.2"
    state.ev.assessment_of_dm_results = {"cv_score": 0.8, "threshold": 0.99, "meets": False}
    suggested = state._suggested_pm_action()
    assert suggested is not None
    assert suggested["action"] == "loop_back"
    assert suggested["loop_label"] == "C"


@patch("maads.agents.run_json_task")
def test_proof_milestone_degraded_then_loop_b(mock_llm, titanic_state: CrispDMState, tmp_path: Path):
    """Degraded prep + validator findings → PM Loop B → rerun completes with submission."""
    mock_llm.side_effect = fake_llm_response
    artifact_dir = _artifact_dir(tmp_path)
    titanic_state.record_degraded("data_engineer@3.2: baseline fallback")
    titanic_state.validator_findings = ["derived FamilySize missing from parquet"]
    titanic_state.substep = "5.1"
    suggested = titanic_state._suggested_pm_action()
    assert suggested and suggested["loop_label"] == "B"

    flow = make_flow(titanic_state, artifact_dir)
    plans = iter([
        Plan(
            action="loop_back",
            loop_label="B",
            loop_to_phase=3,
            target_substep="3.2",
            reason="address degraded prep",
        ),
        *[Plan(action="advance", reason="ok") for _ in range(60)],
        Plan(action="halt", reason="done"),
    ])
    flow._pm.plan = lambda _s: next(plans, Plan(action="halt", reason="done"))  # type: ignore[method-assign]
    flow.run()

    assert titanic_state.loop_history and titanic_state.loop_history[0].label == "B"
    assert titanic_state.dep.submission_path


@patch("maads.agents.run_json_task")
def test_checkpoint_5_1_does_not_fire_loop_c_on_unevaluated_goal(
    mock_llm, titanic_state: CrispDMState, tmp_path: Path,
):
    """PM at 5.1 must not treat missing assessment as business-goal failure."""
    mock_llm.side_effect = fake_llm_response
    thr = titanic_state.config.success_criterion.threshold
    titanic_state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=thr + 0.05,
        description="strong model",
    )
    titanic_state.ev.assessment_of_dm_results = None
    titanic_state.substep = "5.1"
    assert titanic_state.view_for("pm")["business_goal_met"] is None

    artifact_dir = _artifact_dir(tmp_path)
    flow = make_flow(titanic_state, artifact_dir)
    plans = iter([
        Plan(action="advance", reason="ok"),
        *[Plan(action="advance", reason="ok") for _ in range(60)],
        Plan(action="halt", reason="done"),
    ])
    flow._pm.plan = lambda _s: next(plans, Plan(action="halt", reason="done"))  # type: ignore[method-assign]
    flow.run()

    assert not any(le.label == "C" for le in titanic_state.loop_history)
    assessment = titanic_state.ev.assessment_of_dm_results or {}
    assert assessment.get("meets") is True
