"""PM checkpoint routing helpers for CrispDMFlow."""
from __future__ import annotations

from maads.deltas import Plan
from maads.flow.phase_runner import (
    RunContext,
    apply_loop,
    can_fire_loop,
    deployment_review_pending,
    force_halt,
    loop_block_reason,
    loop_route_for_phase,
    resolve_plan,
)


def route_from_plan(ctx: RunContext, plan: Plan) -> str:
    """Map a PM plan to a flow route label."""
    if plan.action == "halt":
        if deployment_review_pending(ctx.state):
            return "continue"
        force_halt(ctx.state, plan.reason or "PM halt")
        return "halt"
    if plan.action == "loop_back":
        if plan.loop_to_phase and can_fire_loop(ctx, plan):
            apply_loop(ctx, plan, log_source="flow")
            return loop_route_for_phase(int(plan.loop_to_phase))
        reason = loop_block_reason(ctx, plan)
        ctx.state.append_log("flow", f"loop blocked by guard: {reason}", level="warn")
        force_halt(ctx.state, f"recovery budget exhausted: {reason}")
        return "halt"
    return "continue"


def checkpoint_route(ctx: RunContext) -> str:
    """Resolve PM plan at a checkpoint substep and return a flow route."""
    plan = resolve_plan(ctx, force=True)
    ctx.state.append_log(
        "pm",
        f"plan -> {plan.action} {plan.target_substep or ''}: {plan.reason}",
    )
    route = route_from_plan(ctx, plan)
    if route == "continue":
        return "advance"
    return route
