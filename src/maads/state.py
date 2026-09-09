"""The shared CRISP-DM state.

The state is organised as one nested model per phase, with one field per
output named in the CRISP-DM 1.0 Reference Model "Generic Tasks and Outputs"
diagram. The naming mirrors the spec so an agent's prompt can reference an
output by its canonical name and the team's mental model stays consistent
with the literature.

Two design rules:
    1. Append-only logs. `log`, `loop_history`, and `md.models` only grow.
    2. No agent reads another agent's prompts — they only read state. This
       keeps agents loosely coupled and individually swappable.

When reading or writing state inside an agent prompt, use the
`view_for(agent_name)` helper at the bottom of this file. Sending the
entire state to every LLM call burns the token budget — use
`view_for(agent_name)` instead (see its docstring below).
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import IntEnum
from typing import Any

from pydantic import BaseModel, Field

from maads.config import CaseConfig
from maads.success_criterion import assessment_meets, criterion_direction, score_meets_threshold


class Phase(IntEnum):
    BUSINESS_UNDERSTANDING = 1
    DATA_UNDERSTANDING = 2
    DATA_PREPARATION = 3
    MODELING = 4
    EVALUATION = 5
    DEPLOYMENT = 6


# The 24 generic tasks of CRISP-DM 1.0, in canonical order.
SUBSTEPS: dict[Phase, list[str]] = {
    Phase.BUSINESS_UNDERSTANDING: ["1.1", "1.2", "1.3", "1.4"],
    Phase.DATA_UNDERSTANDING:     ["2.1", "2.2", "2.3", "2.4"],
    Phase.DATA_PREPARATION:       ["3.1", "3.2", "3.3", "3.4", "3.5"],
    Phase.MODELING:               ["4.1", "4.2", "4.3", "4.4"],
    Phase.EVALUATION:             ["5.1", "5.2", "5.3"],
    Phase.DEPLOYMENT:             ["6.1", "6.2", "6.3", "6.4"],
}


SUBSTEP_NAMES: dict[str, str] = {
    "1.1": "Determine Business Objectives",
    "1.2": "Assess Situation",
    "1.3": "Determine Data Mining Goals",
    "1.4": "Produce Project Plan",
    "2.1": "Collect Initial Data",
    "2.2": "Describe Data",
    "2.3": "Explore Data",
    "2.4": "Verify Data Quality",
    "3.1": "Select Data",
    "3.2": "Clean Data",
    "3.3": "Construct Data",
    "3.4": "Integrate Data",
    "3.5": "Format Data",
    "4.1": "Select Modeling Technique",
    "4.2": "Generate Test Design",
    "4.3": "Build Model",
    "4.4": "Assess Model",
    "5.1": "Evaluate Results",
    "5.2": "Review Process",
    "5.3": "Determine Next Steps",
    "6.1": "Build Submission",
    "6.2": "Generate Report Evidence",
    "6.3": "Produce Final Report",
    "6.4": "Review Project",
}


SUBSTEP_OWNER: dict[str, str] = {
    # Phase 1 — Business Understanding
    "1.1": "domain", "1.2": "domain", "1.3": "domain", "1.4": "pm",
    # Phase 2 — Data Understanding
    "2.1": "data_engineer", "2.2": "data_engineer",
    "2.3": "data_scientist",  # data_engineer also contributes
    "2.4": "data_engineer",
    # Phase 3 — Data Preparation
    "3.1": "data_engineer", "3.2": "data_engineer",
    "3.3": "data_engineer", "3.4": "data_engineer", "3.5": "data_engineer",
    # Phase 4 — Modeling
    "4.1": "data_scientist", "4.2": "data_scientist",
    "4.3": "data_scientist", "4.4": "data_scientist",
    # Phase 5 — Evaluation
    "5.1": "data_scientist", "5.2": "pm", "5.3": "pm",
    # Phase 6 — Reporting
    "6.1": "developer", "6.2": "storyteller",
    "6.3": "storyteller", "6.4": "developer",
}


# ── Cross-cutting log + loop types ──────────────────────────────────────────

class LogEntry(BaseModel):
    t: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent: str
    level: str = "info"  # "info" | "warn" | "error"
    message: str
    data: dict[str, Any] | None = None


class LoopEvent(BaseModel):
    t: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    from_phase: int
    to_phase: int
    label: str   # "A" | "B" | "C" | "D" — matching the four loops in the README
    reason: str


class EvaluationBundle(BaseModel):
    problem_type: str
    metrics: dict[str, float] = Field(default_factory=dict)
    confusion_matrix: list[list[int]] | None = None
    class_labels: dict[str, str] = Field(default_factory=dict)
    cv: dict[str, Any] | None = None
    figures: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def coerce_evaluation_bundle(
    raw: dict[str, Any],
    *,
    problem_type: str,
    class_labels: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Normalize sandbox/LLM evaluation_bundle dicts into EvaluationBundle shape."""
    bundle = dict(raw)
    if not bundle.get("problem_type"):
        bundle["problem_type"] = problem_type

    labels = bundle.get("class_labels")
    if not isinstance(labels, dict):
        labels = dict(class_labels or {})
        bundle["class_labels"] = labels

    metrics_in = bundle.get("metrics")
    metrics: dict[str, float] = {}
    cv: dict[str, Any] = dict(bundle.get("cv") or {})

    if isinstance(metrics_in, dict):
        metrics_copy = dict(metrics_in)
        per_class = metrics_copy.pop("per_class", None)
        if isinstance(per_class, dict):
            for lbl_key, scores in per_class.items():
                if not isinstance(scores, dict):
                    continue
                name = labels.get(str(lbl_key), str(lbl_key))
                for metric_name, value in scores.items():
                    if isinstance(value, (int, float)):
                        metrics[f"{metric_name}_{name}"] = float(value)
        for key, value in metrics_copy.items():
            if key == "cv_f1_mean" and isinstance(value, (int, float)):
                cv["mean"] = float(value)
            elif key == "cv_f1_std" and isinstance(value, (int, float)):
                cv["std"] = float(value)
            elif key.startswith("cv_") and isinstance(value, (int, float)):
                cv[key[3:]] = float(value)
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)

    bundle["metrics"] = metrics
    if cv:
        bundle["cv"] = cv

    figures = bundle.get("figures")
    if isinstance(figures, dict):
        bundle["figures"] = [str(path) for path in figures.values()]
    elif isinstance(figures, list):
        bundle["figures"] = [str(path) for path in figures]
    else:
        bundle["figures"] = []

    warnings = bundle.get("warnings")
    bundle["warnings"] = warnings if isinstance(warnings, list) else []

    return bundle


class ModelRun(BaseModel):
    technique: str
    parameter_settings: dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    cv_score: float | None = None
    cv_std: float | None = None
    holdout_score: float | None = None
    assessment: str | None = None
    revised_parameter_settings: dict[str, Any] | None = None
    evaluation_bundle: EvaluationBundle | None = None
    # Absolute path to persisted fitted pipeline (joblib); enables exact 4.4/6.1 reuse.
    artifact_path: str | None = None


# ── Phase 1 — Business Understanding ───────────────────────────────────────

class BusinessUnderstanding(BaseModel):
    # 1.1 Determine Business Objectives
    background: str | None = None
    business_objectives: str | None = None
    business_success_criteria: str | None = None
    # 1.2 Assess Situation
    inventory_of_resources: dict[str, Any] | None = None
    requirements_assumptions_constraints: dict[str, Any] | None = None
    risks_and_contingencies: list[str] = Field(default_factory=list)
    terminology: dict[str, str] = Field(default_factory=dict)
    costs_and_benefits: dict[str, Any] | None = None
    # 1.3 Determine Data Mining Goals
    data_mining_goals: str | None = None
    data_mining_success_criteria: str | None = None
    # 1.4 Produce Project Plan
    project_plan: list[str] = Field(default_factory=list)
    initial_assessment_of_tools_and_techniques: dict[str, Any] | None = None


# ── Phase 2 — Data Understanding ───────────────────────────────────────────

class DataUnderstanding(BaseModel):
    initial_data_collection_report: dict[str, Any] | None = None  # 2.1
    data_description_report: dict[str, Any] | None = None         # 2.2
    data_exploration_report: dict[str, Any] | None = None         # 2.3
    data_quality_report: dict[str, Any] | None = None             # 2.4


# ── Phase 3 — Data Preparation ─────────────────────────────────────────────

class DataPreparation(BaseModel):
    rationale_for_inclusion_exclusion: dict[str, Any] | None = None  # 3.1
    data_cleaning_report: dict[str, Any] | None = None               # 3.2
    derived_attributes: dict[str, Any] | None = None                   # 3.3
    generated_records: dict[str, Any] | None = None                  # 3.3
    merged_data: dict[str, Any] | None = None                        # 3.4
    reformatted_data: dict[str, Any] | None = None                   # 3.5
    # Phase-level outputs
    dataset: dict[str, str] = Field(default_factory=dict)   # role -> path
    dataset_description: str | None = None


# ── Phase 4 — Modeling ─────────────────────────────────────────────────────

class Modeling(BaseModel):
    modeling_technique: str | None = None                       # 4.1
    modeling_assumptions: list[str] = Field(default_factory=list)  # 4.1
    test_design: dict[str, Any] | None = None                   # 4.2
    # 4.3 & 4.4 captured per-run.
    models: list[ModelRun] = Field(default_factory=list)
    chosen_model: ModelRun | None = None


# ── Phase 5 — Evaluation ───────────────────────────────────────────────────

class Evaluation(BaseModel):
    assessment_of_dm_results: dict[str, Any] | None = None      # 5.1
    approved_models: list[ModelRun] = Field(default_factory=list)
    review_of_process: str | None = None                        # 5.2
    list_of_possible_actions: list[str] = Field(default_factory=list)  # 5.3
    decision: str | None = None


# ── Phase 6 — Deployment ───────────────────────────────────────────────────

class Deployment(BaseModel):
    deployment_plan: str | None = None                          # 6.1
    monitoring_and_maintenance_plan: str | None = None          # legacy
    story_spec_path: str | None = None                          # 6.2
    figures_dir: str | None = None                              # 6.2
    final_report_path: str | None = None                        # 6.3
    final_presentation_path: str | None = None                  # 6.3
    experience_documentation: str | None = None                 # 6.4
    submission_path: str | None = None


# ── Top-level state ────────────────────────────────────────────────────────

class CrispDMState(BaseModel):
    case_id: str
    config: CaseConfig

    phase: Phase = Phase.BUSINESS_UNDERSTANDING
    substep: str = "1.1"
    loop_history: list[LoopEvent] = Field(default_factory=list)
    # Deficits found by a state-artifact validator at the last phase transition;
    # a non-empty list is a Loop B signal the PM reads via view_for("pm").
    validator_findings: list[str] = Field(default_factory=list)
    # Append-only signals when authored code fell back to baseline (Loop B trigger).
    degraded_flags: list[str] = Field(default_factory=list)
    halted: bool = False
    halt_reason: str | None = None

    bu: BusinessUnderstanding = Field(default_factory=BusinessUnderstanding)
    du: DataUnderstanding = Field(default_factory=DataUnderstanding)
    dp: DataPreparation = Field(default_factory=DataPreparation)
    md: Modeling = Field(default_factory=Modeling)
    ev: Evaluation = Field(default_factory=Evaluation)
    dep: Deployment = Field(default_factory=Deployment)

    log: list[LogEntry] = Field(default_factory=list)
    token_spend: dict[str, int] = Field(default_factory=dict)
    token_spend_by_provider: dict[str, int] = Field(default_factory=dict)
    # Run-level input/output split, needed for accurate $ cost (input/output
    # prices differ per model). Optional at the call site for backward
    # compatibility — see add_tokens().
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    @classmethod
    def from_config(cls, config: CaseConfig) -> "CrispDMState":
        return cls(case_id=config.case_id, config=config)

    # ── Mutators agents should use ────────────────────────────────────────

    def append_log(self, agent: str, message: str,
                   level: str = "info", data: dict | None = None) -> None:
        self.log.append(LogEntry(agent=agent, level=level,
                                 message=message, data=data))

    def record_loop(self, label: str, from_phase: int,
                    to_phase: int, reason: str) -> None:
        self.loop_history.append(LoopEvent(
            label=label, from_phase=from_phase, to_phase=to_phase, reason=reason
        ))

    def add_tokens(self, agent: str, n_tokens: int, *, provider: str = "unknown",
                   n_input: int | None = None, n_output: int | None = None) -> None:
        self.token_spend[agent] = self.token_spend.get(agent, 0) + n_tokens
        if n_input is not None:
            self.total_input_tokens += n_input
        if n_output is not None:
            self.total_output_tokens += n_output
        self.token_spend_by_provider[provider] = (
            self.token_spend_by_provider.get(provider, 0) + n_tokens
        )

    def record_degraded(self, flag: str) -> None:
        """Append a degraded-run signal for the PM (Loop B)."""
        if flag and flag not in self.degraded_flags:
            self.degraded_flags.append(flag)

    def resolved_target(self) -> str:
        """Configured target, or the one Data Understanding recorded when the user left it blank."""
        configured = (self.config.target_column or "").strip()
        if configured:
            return configured
        report = self.du.data_exploration_report or {}
        inferred = report.get("target")
        if isinstance(inferred, str) and inferred.strip():
            return inferred.strip()
        return ""

    def adopt_target_if_blank(self, candidate: str | None) -> bool:
        """Persist an evidence-backed target into config when the user omitted one."""
        if (self.config.target_column or "").strip():
            return False
        name = (candidate or "").strip()
        if not name:
            return False
        self.config = self.config.model_copy(update={"target_column": name})
        return True

    def adopt_id_if_blank(self, candidate: str | None) -> bool:
        """Persist an evidence-backed identifier when the user omitted one."""
        if (self.config.id_column or "").strip():
            return False
        name = (candidate or "").strip()
        if not name:
            return False
        self.config = self.config.model_copy(update={"id_column": name})
        return True

    # ── Prerequisite checks the orchestrator uses ─────────────────────────

    def substep_prereqs_satisfied(self, substep: str) -> bool:
        """Return True iff the prerequisites for `substep` are filled.

        This is the anti-phase-jumping rule. Extend it as agents produce
        more outputs.
        """
        if substep == "1.3" and self.bu.business_objectives is None:
            return False
        if substep == "1.4" and self.bu.data_mining_goals is None:
            return False
        if substep == "2.3" and self.du.data_description_report is None:
            return False
        if substep == "3.1" and self.du.data_quality_report is None:
            return False
        if substep == "4.1" and not self.dp.dataset:
            return False
        if substep == "4.4" and not self.md.models:
            return False
        if substep == "5.1" and not self.md.models:
            return False
        if substep == "6.1" and self.md.chosen_model is None:
            return False
        if substep == "6.2":
            cm = self.md.chosen_model
            if cm is None or cm.evaluation_bundle is None:
                return False
        if substep == "6.3" and self.dep.story_spec_path is None:
            return False
        return True

    # ── Token-economy views (see view_for docstring) ──────────────────────

    def _suggested_pm_action(self) -> dict[str, Any] | None:
        """Deterministic loop hint for the PM at decision substeps (advisory)."""
        sub = self.substep
        if sub == "5.1":
            if self.validator_findings:
                return {
                    "action": "loop_back",
                    "loop_label": "B",
                    "loop_to_phase": 3,
                    "target_substep": "3.2",
                    "reason": f"validator findings: {self.validator_findings[:3]}",
                }
            if self.degraded_flags:
                return {
                    "action": "loop_back",
                    "loop_label": "B",
                    "loop_to_phase": 3,
                    "target_substep": "3.2",
                    "reason": f"degraded runs: {self.degraded_flags[:3]}",
                }
        if sub == "5.2":
            assessment = self.ev.assessment_of_dm_results or {}
            if assessment and not assessment_meets(assessment):
                loop_a_count = sum(1 for le in self.loop_history if le.label == "A")
                if loop_a_count >= 2:
                    return {
                        "action": "halt",
                        "reason": "business goal not met and Loop A already fired twice",
                    }
                loop_c_count = sum(1 for le in self.loop_history if le.label == "C")
                if loop_c_count >= 1:
                    return None
                return {
                    "action": "loop_back",
                    "loop_label": "C",
                    "loop_to_phase": 1,
                    "target_substep": "1.3",
                    "reason": "business success criteria not met",
                }
        return None

    def view_for(self, agent_name: str) -> dict[str, Any]:
        """Minimal slice of state to send to an agent's prompt.

        DO NOT serialise the whole state object into a prompt. Always
        send the smallest view that lets the agent do its substep.
        """
        base = {
            "case_id": self.case_id,
            "phase": int(self.phase),
            "substep": self.substep,
            "substep_name": SUBSTEP_NAMES.get(self.substep, "?"),
            "config": {
                "problem_statement": self.config.problem_statement,
                "problem_type": self.config.problem_type,
                "target_column": self.config.target_column,
                "evaluation_metric": self.config.evaluation_metric,
            },
        }
        if agent_name == "pm":
            base["loop_history"] = [le.model_dump() for le in self.loop_history]
            base["recent_log"] = _trim_log(self.log, n=8)
            base["latest_quality_blockers"] = _quality_blockers(self.du.data_quality_report)
            base["quality_gate"] = _quality_gate_view(self)
            base["latest_model_assessment"] = _latest_model_assessment(self.md)
            base["outputs_status"] = _pm_outputs_status(self)
            base["validator_findings"] = list(self.validator_findings)
            base["degraded_flags"] = list(self.degraded_flags)
            base["suggested_action"] = self._suggested_pm_action()
            assessment = self.ev.assessment_of_dm_results
            if self.substep == "5.1":
                # Checkpoint 5.1 runs before Evaluate Results this visit.
                base["business_goal_met"] = None
            elif assessment is None:
                cv = self.md.chosen_model.cv_score if self.md.chosen_model else None
                sc = self.config.success_criterion
                dir_ = criterion_direction(sc.metric, sc.direction)
                base["business_goal_met"] = (
                    score_meets_threshold(cv, sc.threshold, direction=dir_)
                    if cv is not None
                    else None
                )
            else:
                base["business_goal_met"] = assessment_meets(assessment)
        elif agent_name == "domain":
            base["bu"] = self.bu.model_dump(exclude_none=True)
            base["du_so_far"] = self.du.model_dump(exclude_none=True)
            base["feature_hints"] = self.config.feature_hints
        elif agent_name == "data_engineer":
            base.update(self._de_phase_view())
        elif agent_name == "data_scientist":
            base["data_mining_goals"] = self.bu.data_mining_goals
            base["dataset"] = self.dp.dataset
            base["data_description_report"] = self.du.data_description_report
            base["recent_models"] = [m.model_dump() for m in self.md.models[-3:]]
            base["test_design"] = self.md.test_design
        elif agent_name == "developer":
            base["chosen_model"] = (
                self.md.chosen_model.model_dump() if self.md.chosen_model else None
            )
            base["dataset"] = self.dp.dataset
            base["sample_submission_csv"] = self.config.data.sample_submission_csv
            base["data_description_report"] = self.du.data_description_report
        elif agent_name == "storyteller":
            cm = self.md.chosen_model
            base["chosen_model"] = cm.model_dump() if cm else None
            base["evaluation_bundle"] = (
                cm.evaluation_bundle.model_dump()
                if cm and cm.evaluation_bundle
                else None
            )
            base["business_objectives"] = self.bu.business_objectives
            base["data_mining_goals"] = self.bu.data_mining_goals
            base["data_description_report"] = self.du.data_description_report
            base["data_exploration_report"] = self.du.data_exploration_report
            base["test_design"] = self.md.test_design
            base["assessment_of_dm_results"] = self.ev.assessment_of_dm_results
            base["loop_history"] = [le.model_dump() for le in self.loop_history]
            base["degraded_flags"] = list(self.degraded_flags)
            base["class_labels"] = dict(self.config.class_labels)
            base["submission_path"] = self.dep.submission_path
        return base

    def _de_phase_view(self) -> dict[str, Any]:
        """Substep-scoped DU/DP slices for the Data Engineer (token discipline)."""
        du = self.du.model_dump(exclude_none=True)
        dp = self.dp.model_dump(exclude_none=True)
        # Never dump reformatted payloads into prompts.
        dp.pop("reformatted_data", None)

        du_keys: tuple[str, ...] = ()
        dp_keys: tuple[str, ...] = ()
        sub = self.substep
        if sub == "2.4":
            du_keys = ("data_description_report",)
        elif sub == "3.1":
            du_keys = (
                "data_quality_report",
                "data_description_report",
                "data_exploration_report",
            )
        elif sub == "3.2":
            du_keys = ("data_quality_report", "data_description_report")
            dp_keys = ("rationale_for_inclusion_exclusion",)
        elif sub == "3.3":
            du_keys = ("data_description_report",)
            dp_keys = ("rationale_for_inclusion_exclusion", "data_cleaning_report")
        elif sub == "3.4":
            dp_keys = (
                "rationale_for_inclusion_exclusion",
                "derived_attributes",
                "generated_records",
            )
        elif sub == "3.5":
            dp_keys = (
                "rationale_for_inclusion_exclusion",
                "data_cleaning_report",
                "derived_attributes",
                "generated_records",
                "merged_data",
            )
        # 2.1 / 2.2: no prior DU/DP reports

        out: dict[str, Any] = {
            "raw_data_paths": self.config.data.model_dump(),
            "data_mining_goals": self.bu.data_mining_goals,
            "du_so_far": {k: du[k] for k in du_keys if k in du and du[k]},
            "dp_so_far": {k: dp[k] for k in dp_keys if k in dp and dp[k]},
        }
        if sub.startswith("2."):
            out["feature_hints"] = self.config.feature_hints
        return out


def next_substep(state: "CrispDMState") -> str | None:
    """Return the substep after `state.substep`, or None past 6.4."""
    phase = state.phase
    subs = SUBSTEPS[phase]
    i = subs.index(state.substep)
    if i + 1 < len(subs):
        return subs[i + 1]
    next_phase = int(phase) + 1
    if next_phase > int(Phase.DEPLOYMENT):
        return None
    return SUBSTEPS[Phase(next_phase)][0]


def _pm_outputs_status(state: "CrispDMState") -> dict[str, bool]:
    bu, du, dp, md, ev, dep = state.bu, state.du, state.dp, state.md, state.ev, state.dep
    phase_4_model_ok = any(
        m.cv_score is not None and m.assessment for m in md.models
    )
    return {
        "phase_1_ready": bool(
            bu.business_objectives
            and bu.business_success_criteria
            and bu.data_mining_goals
            and bu.project_plan
        ),
        "phase_2_ready": bool(
            du.data_description_report
            and du.data_exploration_report
            and du.data_quality_report
        ),
        "phase_3_ready": bool(dp.dataset),
        "phase_4_ready": bool(phase_4_model_ok and md.chosen_model),
        "phase_5_ready": bool(
            ev.assessment_of_dm_results and ev.review_of_process and ev.decision
        ),
        "phase_6_ready": bool(
            dep.submission_path
            and dep.final_report_path
            and dep.experience_documentation
        ),
    }


# ── Helpers used by view_for ───────────────────────────────────────────────

def _trim_log(log: list[LogEntry], n: int = 8) -> list[dict]:
    """Return the last n log entries with messages truncated."""
    out = []
    for e in log[-n:]:
        out.append({
            "agent": e.agent,
            "level": e.level,
            "message": e.message[:200],
        })
    return out


def _quality_blockers(report: dict | None) -> list[str]:
    if not report:
        return []
    return list(report.get("blockers", []))


def _domain_artifacts(bu: BusinessUnderstanding) -> dict[str, Any]:
    inv = bu.inventory_of_resources or {}
    artifacts = inv.get("domain_artifacts") or {}
    return artifacts if isinstance(artifacts, dict) else {}


def has_actionable_loop_a_trigger(state: "CrispDMState") -> bool:
    """True when Loop A has a concrete quality or domain trigger.

    Documented missingness (``high_missing`` / ``na_means_absent``) lives in
    the quality report's tolerable list — empty blockers alone are not enough.
    """
    if _quality_blockers(state.du.data_quality_report):
        return True
    rec = _domain_artifacts(state.bu).get("loop_a_recommendation") or {}
    return bool(isinstance(rec, dict) and rec.get("should_trigger"))


def _quality_gate_view(state: "CrispDMState") -> dict[str, Any]:
    """Context for PM semantic Loop A at 3.1."""
    fh = state.config.feature_hints or {}
    artifacts = _domain_artifacts(state.bu)
    return {
        "data_quality_report": state.du.data_quality_report,
        "na_means_absent": list(fh.get("na_means_absent") or []),
        "high_missing": list(fh.get("high_missing") or []),
        "domain_data_quality_flags": artifacts.get("domain_data_quality_flags") or [],
        "loop_a_recommendation": artifacts.get("loop_a_recommendation"),
        "data_mining_goals": state.bu.data_mining_goals,
    }


def _latest_model_assessment(md: Modeling) -> dict | None:
    if not md.models:
        return None
    last = md.models[-1]
    return {
        "technique": last.technique,
        "cv_score": last.cv_score,
        "assessment": last.assessment,
    }
