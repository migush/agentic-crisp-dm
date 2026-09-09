"""CrewAI Flow for the CRISP-DM state machine."""
from __future__ import annotations

from pathlib import Path

from crewai.flow.flow import Flow, listen, or_, router, start

from maads.agents import (
    DataEngineerAgent,
    DataScientistAgent,
    DeveloperAgent,
    DomainExpertAgent,
    ProjectManagerAgent,
    StorytellerAgent,
)
from maads.outcome import completion_halt_reason
from maads.run_deadline import start_deadline_clock
from maads.flow import phase_runner as pr
from maads.flow.routers import checkpoint_route
from maads.flow.tracing import install_flow_run_tracer, trace_flow_method
from maads.observability.collector import get_collector
from maads.state import Phase, CrispDMState


class CrispDMFlow(Flow[CrispDMState]):
    """Event-driven CRISP-DM workflow using CrewAI Flow decorators."""

    def __init__(self, state: CrispDMState, artifact_dir: Path) -> None:
        super().__init__(initial_state=state)
        # CrewAI copies BaseModel initial_state into a StateWithId instance.
        # Keep the caller's object authoritative for artifacts, tests, and dashboard.
        object.__setattr__(self, "_state", state)
        self._artifact_dir = artifact_dir
        self._pm = ProjectManagerAgent(artifact_dir=artifact_dir)
        self._agents = {
            "pm": self._pm,
            "domain": DomainExpertAgent(artifact_dir=artifact_dir),
            "data_engineer": DataEngineerAgent(artifact_dir=artifact_dir),
            "data_scientist": DataScientistAgent(artifact_dir=artifact_dir),
            "developer": DeveloperAgent(artifact_dir=artifact_dir),
            "storyteller": StorytellerAgent(artifact_dir=artifact_dir),
        }
        self._ctx = pr.RunContext.create(state, artifact_dir, self._agents, self._pm)
        self._pending_route: str | None = None
        self._last_checkpoint_route = "advance"

    @property
    def artifact_dir(self) -> Path:
        return self._artifact_dir

    def _route_after(self, default_next: str) -> str:
        if self.state.halted:
            return "halt"
        route = self._pending_route
        if route in (None, "advance"):
            return default_next
        return route

    def kickoff(self, inputs=None, **kwargs):  # type: ignore[override]
        start_deadline_clock()
        coll = get_collector()
        tracer = install_flow_run_tracer()
        coll.emit_start(
            "run.start",
            span_key="flow.run",
            name="CrispDMFlow.kickoff",
            source="maads.flow.crisp_dm_flow",
            attributes={
                "case_id": self.state.case_id,
                "phase": int(self.state.phase),
                "substep": self.state.substep,
            },
        )
        try:
            return super().kickoff(inputs=inputs, **kwargs)
        finally:
            tracer.uninstall()
            coll.emit_end(
                "run.end",
                span_key="flow.run",
                attributes={
                    "halt_reason": self.state.halt_reason,
                    "halted": self.state.halted,
                },
            )

    @start()
    @trace_flow_method("initialize")
    def initialize(self) -> None:
        reason = pr.check_global_halt(self._ctx)
        if reason:
            pr.force_halt(self.state, reason)

    @listen(or_(initialize, "phase_1"))
    @trace_flow_method("phase_1")
    def run_phase_1(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        self._pending_route = pr.run_phase_substeps(self._ctx, Phase.BUSINESS_UNDERSTANDING)

    @router(run_phase_1)
    def route_after_phase_1(self) -> str:
        return self._route_after("phase_2")

    @listen("phase_2")
    @trace_flow_method("phase_2")
    def run_phase_2(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        self._pending_route = pr.run_phase_substeps(self._ctx, Phase.DATA_UNDERSTANDING)

    @router(run_phase_2)
    def route_after_phase_2(self) -> str:
        return self._route_after("entry_checkpoint_3_1")

    @listen("entry_checkpoint_3_1")
    @trace_flow_method("checkpoint_3_1")
    def enter_checkpoint_3_1(self) -> None:
        if self.state.halted:
            self._last_checkpoint_route = "halt"
            return
        # Align with checkpoint_5_1: label the decision point before consulting PM.
        self.state.phase = Phase.DATA_PREPARATION
        self.state.substep = "3.1"
        self._last_checkpoint_route = checkpoint_route(self._ctx)

    @router(enter_checkpoint_3_1)
    def route_checkpoint_3_1(self) -> str:
        route = self._last_checkpoint_route
        if route in {"phase_1", "halt"}:
            return route
        return "phase_3"

    @listen("phase_3")
    @trace_flow_method("phase_3")
    def run_phase_3(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        self._pending_route = pr.run_phase_substeps(self._ctx, Phase.DATA_PREPARATION)

    @router(run_phase_3)
    def route_after_phase_3(self) -> str:
        return self._route_after("phase_4")

    @listen("phase_4")
    @trace_flow_method("phase_4")
    def run_phase_4(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        self._pending_route = pr.run_phase_substeps(self._ctx, Phase.MODELING)

    @router(run_phase_4)
    def route_after_phase_4(self) -> str:
        return self._route_after("entry_checkpoint_5_1")

    @listen("entry_checkpoint_5_1")
    @trace_flow_method("checkpoint_5_1")
    def enter_checkpoint_5_1(self) -> None:
        if self.state.halted:
            self._last_checkpoint_route = "halt"
            return
        self.state.phase = Phase.EVALUATION
        self.state.substep = "5.1"
        self._last_checkpoint_route = checkpoint_route(self._ctx)

    @router(enter_checkpoint_5_1)
    def route_checkpoint_5_1(self) -> str:
        route = self._last_checkpoint_route
        if route == "phase_1":
            return "phase_1"
        if route == "phase_3":
            return "phase_3"
        if route == "halt":
            return "halt"
        return "phase_5"

    @listen("phase_5")
    @trace_flow_method("phase_5")
    def run_phase_5(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        pr.execute_substep(self._ctx, "5.1")
        if self.state.halted:
            self._pending_route = "halt"
            return
        pr.advance_substep(self._ctx)
        self.state.substep = "5.2"
        self._pending_route = None

    @router(run_phase_5)
    def route_after_phase_5(self) -> str:
        return self._route_after("entry_checkpoint_5_2")

    @listen("entry_checkpoint_5_2")
    @trace_flow_method("checkpoint_5_2")
    def enter_checkpoint_5_2(self) -> None:
        if self.state.halted:
            self._last_checkpoint_route = "halt"
            return
        self._last_checkpoint_route = checkpoint_route(self._ctx)

    @router(enter_checkpoint_5_2)
    def route_checkpoint_5_2(self) -> str:
        route = self._last_checkpoint_route
        if route == "phase_1":
            return "phase_1"
        if route == "halt":
            return "halt"
        return "phase_5_tail"

    @listen("phase_5_tail")
    @trace_flow_method("phase_5_tail")
    def run_phase_5_tail(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        for substep in ("5.2", "5.3"):
            self.state.substep = substep
            reason = pr.check_global_halt(self._ctx)
            if reason:
                pr.force_halt(self.state, reason)
                self._pending_route = "halt"
                return
            pr.execute_substep(self._ctx, substep)
            if pr.advance_substep(self._ctx):
                self._pending_route = "complete"
                return
        self._pending_route = None

    @router(run_phase_5_tail)
    def route_after_phase_5_tail(self) -> str:
        return self._route_after("phase_6")

    @listen("phase_6")
    @trace_flow_method("phase_6")
    def run_phase_6(self) -> None:
        if self.state.halted:
            self._pending_route = "halt"
            return
        self._pending_route = pr.run_phase_substeps(self._ctx, Phase.DEPLOYMENT)

    @router(run_phase_6)
    def route_after_phase_6(self) -> str:
        if self.state.halted:
            return "halt"
        route = self._pending_route
        if route in (None, "advance"):
            return "complete"
        return route

    @listen(or_("halt", "complete"))
    @trace_flow_method("finish")
    def finish(self) -> CrispDMState:
        if not self.state.halted and int(self.state.phase) == int(Phase.DEPLOYMENT):
            self.state.halted = True
            self.state.halt_reason = (
                self.state.halt_reason or completion_halt_reason(self.state)
            )
        return self._state

    def run(self) -> CrispDMState:
        """Run the flow and return the shared CrispDMState."""
        self.kickoff()
        return self._state
