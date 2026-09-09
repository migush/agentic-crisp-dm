"""Shared CRISP-DM orchestration primitives for CrispDMFlow."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from maads.deltas import Plan
from maads.flow.tracing import trace_substep_dispatch, trace_substep_end
from maads.shutdown import INTERRUPT_HALT_REASON, shutdown_requested
from maads.state import (
    SUBSTEPS,
    SUBSTEP_OWNER,
    CrispDMState,
    Phase,
    has_actionable_loop_a_trigger,
)
from maads.run_deadline import HALT_REASON as DEADLINE_HALT_REASON, deadline_exceeded
from maads.token_budget import (
    HALT_REASON,
    TokenBudgetExceeded,
    budget_status,
    over_budget,
    soft_limit_reached,
)
from maads.outcome import completion_halt_reason
from maads.validators import validate_phase_3_artifacts, validate_phase_4_models

MAX_PHASE_TRANSITIONS = 12
MAX_VISITS_PER_PHASE = 3
MAX_INNER_LOOP_ITERATIONS = 3

PM_DECISION_SUBSTEPS = frozenset({
    "1.1",
    "2.1",
    # Loop A at phase-2 exit lives only in checkpoint_3_1 (not again at 3.1 entry).
    "4.1",
    "5.1",
    "5.2",
    "6.1",
    "6.4",
})


class PMAgent(Protocol):
    def plan(self, state: CrispDMState) -> Plan: ...


class AgentLike(Protocol):
    def act(self, state: CrispDMState) -> Any: ...


@dataclass
class RunContext:
    state: CrispDMState
    artifact_dir: Path
    agents: dict[str, AgentLike]
    pm: PMAgent
    n_transitions: int = 0
    phase_visits: dict[int, int] = field(default_factory=dict)
    inner_loop_count: int = 0

    @classmethod
    def create(
        cls,
        state: CrispDMState,
        artifact_dir: Path,
        agents: dict[str, AgentLike],
        pm: PMAgent,
    ) -> RunContext:
        visits = {int(state.phase): 1}
        return cls(
            state=state,
            artifact_dir=artifact_dir,
            agents=agents,
            pm=pm,
            phase_visits=visits,
        )


def check_global_halt(ctx: RunContext) -> str | None:
    if shutdown_requested():
        return INTERRUPT_HALT_REASON
    if deadline_exceeded():
        return DEADLINE_HALT_REASON
    if caps_exceeded(ctx):
        return "hard cap exceeded"
    if over_budget(ctx.state):
        return HALT_REASON
    return None


def resolve_plan(ctx: RunContext, *, force: bool = False) -> Plan:
    """Ask the PM for a plan, or return a mechanical advance.

    Flow checkpoints pass ``force=True`` so Loop A/B/C decisions still consult
    the PM even when the current substep is not in ``PM_DECISION_SUBSTEPS``
    (e.g. checkpoint_3_1 after 3.1 was removed from in-phase decision points).
    """
    substep = ctx.state.substep
    if not force and substep not in PM_DECISION_SUBSTEPS:
        return Plan(
            action="advance",
            reason=f"mechanical advance within phase at {substep}",
        )
    plan = ctx.pm.plan(ctx.state)
    suggested = ctx.state._suggested_pm_action()
    if suggested and plan.action != suggested.get("action"):
        ctx.state.append_log(
            "orchestrator",
            f"PM overrode suggested_action {suggested.get('action')}: "
            f"chose {plan.action} — {plan.reason}",
            level="warn",
        )
    elif suggested:
        ctx.state.append_log(
            "orchestrator",
            f"PM aligned with suggested_action: {suggested.get('action')}",
        )
    return plan


def run_substep(ctx: RunContext, substep: str) -> bool:
    """Run one substep. Return False when dispatch failed or agent reported failure."""
    if not ctx.state.substep_prereqs_satisfied(substep):
        ctx.state.append_log(
            "orchestrator",
            f"prereqs not satisfied for {substep}; skipping",
            level="warn",
        )
        return True
    owner = SUBSTEP_OWNER[substep]
    delta = ctx.agents[owner].act(ctx.state)
    detail = ", ".join(delta.fields_written) or delta.notes
    level = "warn" if delta.failed else "info"
    ctx.state.append_log(owner, f"ran {substep} -> {detail}", level=level)
    return not delta.failed


def execute_substep(ctx: RunContext, substep: str) -> bool:
    """Run one substep and emit trace events for dashboard dispatch edges."""
    trace_substep_dispatch(ctx.state, substep)
    ok = run_substep(ctx, substep)
    if ok:
        trace_substep_end(substep)
    return ok


def apply_loop(
    ctx: RunContext,
    plan: Plan,
    *,
    log_source: str = "orchestrator",
) -> None:
    target_phase = int(plan.loop_to_phase)  # type: ignore[arg-type]
    label = plan.loop_label or "?"
    from_phase = int(ctx.state.phase)
    ctx.state.record_loop(label, from_phase, target_phase, plan.reason)
    from maads.flow.tracing import trace_loop
    from maads.progress import on_loop

    trace_loop(label, from_phase, target_phase, plan.reason)
    on_loop(label, target_phase, plan.reason)
    if label == "B":
        ctx.inner_loop_count += 1
    ctx.state.phase = Phase(target_phase)
    ctx.state.substep = SUBSTEPS[Phase(target_phase)][0]
    ctx.phase_visits[target_phase] = ctx.phase_visits.get(target_phase, 0) + 1
    ctx.state.validator_findings = []
    # Stale 5.1 assessment must not trigger Loop C at the next 5.1 checkpoint.
    ctx.state.ev.assessment_of_dm_results = None
    ctx.state.append_log(
        log_source,
        f"loop {label} -> phase {target_phase}: {plan.reason}",
        level="warn",
    )
    if plan.target_substep:
        allowed = SUBSTEPS[Phase(target_phase)]
        if plan.target_substep in allowed:
            ctx.state.substep = plan.target_substep
        else:
            ctx.state.append_log(
                log_source,
                f"ignored PM target_substep {plan.target_substep} "
                f"for phase {target_phase}",
                level="warn",
            )


def validate_phase_exit(state: CrispDMState, leaving_phase: Phase) -> None:
    if leaving_phase == Phase.DATA_PREPARATION:
        findings = validate_phase_3_artifacts(state)
    elif leaving_phase == Phase.MODELING:
        findings = validate_phase_4_models(state)
    else:
        return
    state.validator_findings = findings
    if findings:
        state.append_log(
            "orchestrator",
            f"validator found {len(findings)} deficit(s) leaving phase "
            f"{int(leaving_phase)}: {'; '.join(findings)}",
            level="warn",
        )


def advance_substep(ctx: RunContext) -> bool:
    """Advance to the next substep / phase. Return True when the run is done."""
    sync_phase_substep(ctx.state)
    phase = ctx.state.phase
    subs = SUBSTEPS[phase]
    i = subs.index(ctx.state.substep)
    if i + 1 < len(subs):
        ctx.state.substep = subs[i + 1]
        return False
    validate_phase_exit(ctx.state, phase)
    ctx.n_transitions += 1
    next_phase = int(phase) + 1
    if next_phase > int(Phase.DEPLOYMENT):
        ctx.state.halted = True
        ctx.state.halt_reason = completion_halt_reason(ctx.state)
        ctx.state.append_log("orchestrator", "run complete: phase 6 finished")
        return True
    ctx.state.phase = Phase(next_phase)
    ctx.state.substep = SUBSTEPS[Phase(next_phase)][0]
    ctx.phase_visits[next_phase] = ctx.phase_visits.get(next_phase, 0) + 1
    return False


def can_fire_loop(ctx: RunContext, plan: Plan) -> bool:
    if caps_exceeded(ctx):
        return False
    if plan.loop_label == "B" and ctx.inner_loop_count >= MAX_INNER_LOOP_ITERATIONS:
        return False
    if plan.loop_label == "C" and any(le.label == "C" for le in ctx.state.loop_history):
        return False
    if plan.loop_label == "A" and not has_actionable_loop_a_trigger(ctx.state):
        return False
    target = plan.loop_to_phase
    if target is None:
        return False
    if ctx.phase_visits.get(target, 0) >= MAX_VISITS_PER_PHASE:
        return False
    return True


def loop_block_reason(ctx: RunContext, plan: Plan) -> str:
    """Explain why ``can_fire_loop`` would reject this plan."""
    if caps_exceeded(ctx):
        return "hard cap exceeded"
    if plan.loop_label == "B" and ctx.inner_loop_count >= MAX_INNER_LOOP_ITERATIONS:
        return "inner Loop B budget exhausted"
    if plan.loop_label == "C" and any(le.label == "C" for le in ctx.state.loop_history):
        return "Loop C already fired once"
    if plan.loop_label == "A" and not has_actionable_loop_a_trigger(ctx.state):
        return "no actionable Loop A quality trigger"
    target = plan.loop_to_phase
    if target is None:
        return "loop_to_phase is missing"
    if ctx.phase_visits.get(target, 0) >= MAX_VISITS_PER_PHASE:
        return f"phase {target} visit cap exhausted"
    return "recovery budget exhausted"


def caps_exceeded(ctx: RunContext) -> bool:
    if ctx.n_transitions >= MAX_PHASE_TRANSITIONS:
        return True
    if any(v > MAX_VISITS_PER_PHASE for v in ctx.phase_visits.values()):
        return True
    if ctx.inner_loop_count > MAX_INNER_LOOP_ITERATIONS:
        return True
    return False


def over_token_budget(state: CrispDMState) -> bool:
    """Back-compat alias for ``token_budget.over_budget``."""
    return over_budget(state)


def maybe_log_soft_limit(ctx: RunContext) -> None:
    """Log once when spend crosses the soft token budget threshold."""
    if not soft_limit_reached(ctx.state):
        return
    for entry in reversed(ctx.state.log):
        if entry.agent == "orchestrator" and "token budget soft limit" in entry.message:
            return
    status = budget_status(ctx.state)
    ctx.state.append_log(
        "orchestrator",
        f"token budget soft limit reached ({status.get('pct')}% of cap); "
        "skipping DEBUG repairs and reducing codegen retries",
        level="warn",
    )


def sync_phase_substep(state: CrispDMState) -> None:
    phase = state.phase
    subs = SUBSTEPS[phase]
    if state.substep in subs:
        return
    try:
        derived = int(state.substep.split(".", 1)[0])
    except (ValueError, AttributeError):
        derived = int(phase)
    target = Phase(derived)
    if target != phase and state.substep in SUBSTEPS.get(target, []):
        state.append_log(
            "orchestrator",
            f"substep {state.substep} out of sync with phase {int(phase)}; "
            f"syncing phase to {derived}",
            level="warn",
        )
        state.phase = target
        return
    state.append_log(
        "orchestrator",
        f"substep {state.substep} invalid for phase {int(phase)}; "
        f"resetting to {subs[0]}",
        level="warn",
    )
    state.substep = subs[0]


def deployment_review_pending(state: CrispDMState) -> bool:
    return (
        state.phase == Phase.DEPLOYMENT
        and not state.dep.experience_documentation
    )


def force_halt(state: CrispDMState, reason: str) -> None:
    state.halted = True
    state.halt_reason = reason
    state.append_log("orchestrator", reason, level="warn")


def handle_plan(
    ctx: RunContext,
    plan: Plan,
) -> str | None:
    """Apply a PM plan. Return a flow route name, or None to continue dispatch."""
    ctx.state.append_log(
        "pm",
        f"plan -> {plan.action} {plan.target_substep or ''}: {plan.reason}",
    )
    if plan.action == "halt":
        if deployment_review_pending(ctx.state):
            ctx.state.append_log(
                "orchestrator",
                "ignored PM halt: substep 6.4 (experience_documentation) not complete",
                level="warn",
            )
            return None
        force_halt(ctx.state, plan.reason or "PM halt")
        return "halt"
    if plan.action == "loop_back":
        route = resolve_loop_back(ctx, plan)
        if route == "continue":
            return None
        return route
    if (
        plan.action == "advance"
        and plan.target_substep
        and plan.target_substep != ctx.state.substep
    ):
        ctx.state.append_log(
            "orchestrator",
            f"ignored PM target_substep {plan.target_substep}; "
            f"running current {ctx.state.substep}",
            level="warn",
        )
    return None


def loop_route_for_phase(phase: int) -> str:
    return {
        1: "phase_1",
        2: "phase_2",
        3: "phase_3",
        4: "phase_4",
        5: "phase_5",
        6: "phase_6",
    }.get(phase, f"phase_{phase}")


def resolve_loop_back(
    ctx: RunContext,
    plan: Plan,
    *,
    log_source: str = "orchestrator",
) -> str:
    """Apply ``loop_back`` or convert it to a continue.

    Loop C is only legal at 5.2 (after Evaluate Results) and at most once.
    Loop A requires actionable quality blockers or an explicit domain
    ``should_trigger`` recommendation. A blocked loop means stop iterating —
    finish the current phase so a submission can still be produced — rather
    than halt with recovery-budget exhaustion.
    """
    if plan.loop_label == "C" and ctx.state.substep == "5.1":
        ctx.state.append_log(
            log_source,
            "ignored premature Loop C at 5.1; Evaluate Results has not run this visit",
            level="warn",
        )
        return "continue"
    if plan.loop_to_phase and can_fire_loop(ctx, plan):
        apply_loop(ctx, plan, log_source=log_source)
        return loop_route_for_phase(int(plan.loop_to_phase))
    reason = loop_block_reason(ctx, plan)
    ctx.state.append_log(
        log_source,
        f"loop blocked by guard: {reason}; advancing instead of exhausting recovery",
        level="warn",
    )
    return "continue"


def run_phase_substeps(ctx: RunContext, phase: Phase) -> str | None:
    """Run substeps for one phase from the current substep through phase end."""
    subs = SUBSTEPS[phase]
    start_idx = subs.index(ctx.state.substep) if ctx.state.substep in subs else 0
    for substep in subs[start_idx:]:
        ctx.state.substep = substep
        reason = check_global_halt(ctx)
        if reason:
            force_halt(ctx.state, reason)
            return "halt"
        maybe_log_soft_limit(ctx)
        if substep in PM_DECISION_SUBSTEPS:
            route = handle_plan(ctx, resolve_plan(ctx))
            if route:
                return route
            if ctx.state.halted:
                return "halt"
        else:
            ctx.state.append_log(
                "orchestrator",
                f"mechanical advance within phase at {substep}",
            )
        try:
            if not execute_substep(ctx, substep):
                force_halt(ctx.state, f"execution failed at {substep}")
                return "halt"
        except TokenBudgetExceeded:
            force_halt(ctx.state, HALT_REASON)
            return "halt"
        except Exception as exc:
            force_halt(ctx.state, f"dispatch failed at {substep}: {exc}")
            return "halt"
        if advance_substep(ctx):
            return "complete"
    return None
