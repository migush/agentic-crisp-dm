"""Tests for shared phase_runner primitives."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maads.config import load_case_config
from maads.deltas import Plan, StateDelta
from maads.flow.phase_runner import (
    RunContext,
    advance_substep,
    apply_loop,
    can_fire_loop,
    deployment_review_pending,
    force_halt,
    handle_plan,
    loop_block_reason,
    resolve_loop_back,
    resolve_plan,
    run_substep,
)
from maads.outcome import completion_halt_reason, ml_outcome_deficits
from maads.paths import resolve_path
from maads.state import CrispDMState, ModelRun, Phase


@pytest.fixture
def titanic_state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


def _ctx(state: CrispDMState, tmp_path: Path) -> RunContext:
    pm = MagicMock()
    pm.plan.return_value = Plan(action="advance", reason="test")
    agents = {
        "domain": MagicMock(),
        "pm": pm,
        "data_engineer": MagicMock(),
        "data_scientist": MagicMock(act=MagicMock(return_value=__import__("maads.deltas", fromlist=["StateDelta"]).StateDelta(notes="skip"))),
        "developer": MagicMock(),
    }
    artifact = tmp_path / "artifacts"
    artifact.mkdir(parents=True)
    return RunContext.create(state, artifact, agents, pm)


def test_advance_substep_moves_within_phase(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    assert titanic_state.substep == "1.1"
    assert advance_substep(ctx) is False
    assert titanic_state.substep == "1.2"


def test_resolve_plan_mechanical_outside_checkpoints(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.substep = "1.2"
    plan = resolve_plan(ctx)
    assert plan.action == "advance"
    ctx.pm.plan.assert_not_called()


def test_apply_loop_records_history(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.DATA_PREPARATION
    titanic_state.substep = "3.1"
    apply_loop(
        ctx,
        Plan(action="loop_back", loop_to_phase=1, loop_label="A", reason="quality"),
    )
    assert titanic_state.phase == Phase.BUSINESS_UNDERSTANDING
    assert len(titanic_state.loop_history) == 1


def test_can_fire_loop_respects_inner_cap(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    ctx.inner_loop_count = 3
    plan = Plan(action="loop_back", loop_to_phase=3, loop_label="B", reason="x")
    assert can_fire_loop(ctx, plan) is False


def test_deployment_review_pending(titanic_state: CrispDMState):
    titanic_state.phase = Phase.DEPLOYMENT
    titanic_state.dep.experience_documentation = ""
    assert deployment_review_pending(titanic_state) is True


def test_run_substep_skips_when_prereqs_missing(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.BUSINESS_UNDERSTANDING
    titanic_state.substep = "4.1"
    run_substep(ctx, "4.1")
    ctx.agents["data_scientist"].act.assert_not_called()


def test_force_halt_sets_state(titanic_state: CrispDMState):
    force_halt(titanic_state, "test halt")
    assert titanic_state.halted is True
    assert titanic_state.halt_reason == "test halt"


def test_run_substep_fails_on_agent_failure(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.MODELING
    titanic_state.substep = "4.1"
    titanic_state.dp.dataset = {"train": "x", "test": "y"}
    ctx.agents["data_scientist"].act.return_value = StateDelta(
        notes="DS 4.3 execution failed: boom",
        failed=True,
    )
    assert run_substep(ctx, "4.1") is False


def test_loop_block_reason_inner_cap(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    ctx.inner_loop_count = 3
    plan = Plan(action="loop_back", loop_to_phase=3, loop_label="B", reason="x")
    assert loop_block_reason(ctx, plan) == "inner Loop B budget exhausted"


def test_handle_plan_advances_on_blocked_loop(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    ctx.inner_loop_count = 3
    plan = Plan(action="loop_back", loop_to_phase=3, loop_label="B", reason="retry prep")
    route = handle_plan(ctx, plan)
    assert route is None
    assert not titanic_state.halted
    assert any("loop blocked" in e.message for e in titanic_state.log)
    assert any("advancing instead" in e.message for e in titanic_state.log)


def test_handle_plan_ignores_loop_c_at_5_1(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.EVALUATION
    titanic_state.substep = "5.1"
    plan = Plan(
        action="loop_back",
        loop_to_phase=1,
        loop_label="C",
        target_substep="1.3",
        reason="stale business_goal_met",
    )
    assert resolve_loop_back(ctx, plan) == "continue"
    assert titanic_state.phase == Phase.EVALUATION
    assert not titanic_state.loop_history
    assert any("premature Loop C" in e.message for e in titanic_state.log)


def test_handle_plan_advances_when_phase_1_visit_cap_blocks_loop_c(
    titanic_state: CrispDMState, tmp_path: Path,
):
    ctx = _ctx(titanic_state, tmp_path)
    ctx.phase_visits[1] = 3
    titanic_state.phase = Phase.EVALUATION
    titanic_state.substep = "5.2"
    plan = Plan(
        action="loop_back",
        loop_to_phase=1,
        loop_label="C",
        target_substep="1.3",
        reason="business goal not met",
    )
    route = handle_plan(ctx, plan)
    assert route is None
    assert not titanic_state.halted
    assert not titanic_state.loop_history
    assert any("phase 1 visit cap exhausted" in e.message for e in titanic_state.log)


def test_handle_plan_blocks_second_loop_c(titanic_state: CrispDMState, tmp_path: Path):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.EVALUATION
    titanic_state.substep = "5.2"
    titanic_state.record_loop("C", 5, 1, "first Loop C")
    plan = Plan(
        action="loop_back",
        loop_to_phase=1,
        loop_label="C",
        target_substep="1.3",
        reason="business goal still not met",
    )
    route = handle_plan(ctx, plan)
    assert route is None
    assert not titanic_state.halted
    assert len(titanic_state.loop_history) == 1
    assert any("Loop C already fired once" in e.message for e in titanic_state.log)


def test_handle_plan_blocks_loop_a_without_quality_trigger(
    titanic_state: CrispDMState, tmp_path: Path,
):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.DATA_PREPARATION
    titanic_state.substep = "3.1"
    titanic_state.du.data_quality_report = {
        "blockers": [],
        "tolerable": ["Cabin: documented high missingness (77% NA)"],
        "source": "deterministic quality_report",
    }
    plan = Plan(
        action="loop_back",
        loop_to_phase=1,
        loop_label="A",
        target_substep="1.3",
        reason="PM invented Loop A for tolerable Cabin missingness",
    )
    route = handle_plan(ctx, plan)
    assert route is None
    assert not titanic_state.halted
    assert not titanic_state.loop_history
    assert any("no actionable Loop A" in e.message for e in titanic_state.log)


def test_handle_plan_allows_loop_a_with_blockers(
    titanic_state: CrispDMState, tmp_path: Path,
):
    ctx = _ctx(titanic_state, tmp_path)
    titanic_state.phase = Phase.DATA_PREPARATION
    titanic_state.substep = "3.1"
    titanic_state.du.data_quality_report = {
        "blockers": ["missing target column 'Survived'"],
        "tolerable": [],
        "source": "deterministic quality_report",
    }
    plan = Plan(
        action="loop_back",
        loop_to_phase=1,
        loop_label="A",
        target_substep="1.3",
        reason="actionable quality blocker",
    )
    route = handle_plan(ctx, plan)
    assert route == "phase_1"
    assert titanic_state.phase == Phase.BUSINESS_UNDERSTANDING
    assert titanic_state.loop_history[-1].label == "A"


def test_completion_halt_reason_without_ml_success(titanic_state: CrispDMState):
    titanic_state.phase = Phase.DEPLOYMENT
    titanic_state.substep = "6.4"
    titanic_state.halted = True
    reason = completion_halt_reason(titanic_state)
    assert "without ML success" in reason
    assert set(ml_outcome_deficits(titanic_state)) >= {
        "no chosen_model",
        "no submission_path",
        "business success criteria not met",
    }


def test_completion_halt_reason_on_full_success(titanic_state: CrispDMState, tmp_path: Path):
    run = ModelRun(technique="rf", cv_score=0.82, assessment="ok")
    titanic_state.md.chosen_model = run
    titanic_state.dep.submission_path = str(tmp_path / "submission.csv")
    titanic_state.ev.assessment_of_dm_results = {"meets": True}
    assert completion_halt_reason(titanic_state) == "completed phase 6"
