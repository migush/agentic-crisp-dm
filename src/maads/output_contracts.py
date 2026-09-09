"""Agent output contracts — schema validation beyond syntactic JSON parsing."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

# Specialist agents share this status set (no FIXED — that is Developer-only).
SpecialistStatus = Literal[
    "COMPLETED",
    "PARTIAL",
    "REVISION_REQUIRED",
    "BLOCKED",
    "HANDOFF_REQUIRED",
]

DeveloperStatus = Literal[
    "COMPLETED",
    "PARTIAL",
    "FIXED",
    "REVISION_REQUIRED",
    "BLOCKED",
    "STUCK",
    "HANDOFF_REQUIRED",
]

PmAction = Literal["advance", "loop_back", "halt"]

_ARRAY_FIELDS_SPECIALIST = (
    "evidence",
    "decisions",
    "operations",
    "quality_findings",
    "validations",
    "artifacts",
    "assumptions",
    "risks",
    "blockers",
    "handoffs",
)

_ARRAY_FIELDS_DS_EXTRA = ("model_runs", "leakage_checks", "diagnostics")

_DICT_LIST_FIELDS = frozenset({
    "evidence",
    "decisions",
    "operations",
    "quality_findings",
    "validations",
    "artifacts",
    "assumptions",
    "risks",
    "blockers",
    "handoffs",
    "model_runs",
    "leakage_checks",
    "diagnostics",
})

_SCHEMA_SHAPE_NOTES = """
Shape rules for specialist agent output:
- assignment_id must be exactly the CRISP-DM substep id (e.g. "6.2"), never a
  run-id, agent name, or "run:substep:agent" composite.
- assumptions, risks, blockers, handoffs, evidence, decisions: arrays of OBJECTS
  (e.g. [{"statement": "..."}]), NOT bare strings.
- Put plain-text modeling assumptions in state_updates.md.modeling_assumptions
  when the substep requests them; use top-level assumptions for structured evidence.
- List fields must be JSON arrays [], never empty strings.
"""

_SUBSTEP_SCHEMA_EXAMPLES: dict[tuple[str, str], str] = {
    ("data_scientist", "4.1"): (
        'Example for 4.1 (fragment): "state_updates": {"md": {"modeling_technique": '
        '"lightgbm", "modeling_assumptions": ["text is primary signal; TF-IDF baseline first"]}}, '
        '"assumptions": [{"statement": "Start with linear baseline", "basis": "feature_hints"}], '
        '"risks": []'
    ),
}


class CompletionEvidenceBase(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_contract_valid: bool
    required_outputs_present: bool
    execution_succeeded: bool
    safe_for_downstream_use: bool


class DataEngineerCompletionEvidence(CompletionEvidenceBase):
    artifacts_verified: bool
    leakage_checks_passed: bool
    reproducibility_checks_passed: bool


class LoopSignal(BaseModel):
    model_config = ConfigDict(extra="allow")

    recommended: bool
    contour: str
    reason: str | None = None
    evidence_ids: list[str] = Field(default_factory=list)


class DataEngineerStateUpdates(BaseModel):
    model_config = ConfigDict(extra="allow")

    du: dict[str, Any] = Field(default_factory=dict)
    dp: dict[str, Any] = Field(default_factory=dict)


class DataEngineerOutput(BaseModel):
    assignment_id: str
    agent: Literal["data_engineer"]
    status: SpecialistStatus
    summary: str
    state_updates: DataEngineerStateUpdates
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    operations: list[dict[str, Any]] = Field(default_factory=list)
    quality_findings: list[dict[str, Any]] = Field(default_factory=list)
    validations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    loop_signal: LoopSignal
    completion_evidence: DataEngineerCompletionEvidence


class DataScientistStateUpdates(BaseModel):
    model_config = ConfigDict(extra="allow")

    du: dict[str, Any] = Field(default_factory=dict)
    md: dict[str, Any] = Field(default_factory=dict)
    ev: dict[str, Any] = Field(default_factory=dict)


class DataScientistCompletionEvidence(CompletionEvidenceBase):
    baseline_established: bool
    leakage_checks_passed: bool
    uncertainty_reported: bool
    evaluated_against_success_criterion: bool | None = None


class DataScientistOutput(BaseModel):
    assignment_id: str
    agent: Literal["data_scientist"]
    status: SpecialistStatus
    summary: str
    state_updates: DataScientistStateUpdates
    model_runs: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    validations: list[dict[str, Any]] = Field(default_factory=list)
    leakage_checks: list[dict[str, Any]] = Field(default_factory=list)
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    loop_signal: LoopSignal
    completion_evidence: DataScientistCompletionEvidence


class StorytellerStateUpdates(BaseModel):
    model_config = ConfigDict(extra="allow")

    dep: dict[str, Any] = Field(default_factory=dict)


class StorytellerCompletionEvidence(CompletionEvidenceBase):
    evaluation_bundle_present: bool
    figures_generated: bool
    report_rendered: bool | None = None


class StorytellerOutput(BaseModel):
    assignment_id: str
    agent: Literal["storyteller"]
    status: SpecialistStatus
    summary: str
    story_spec: dict[str, Any] = Field(default_factory=dict)
    state_updates: StorytellerStateUpdates = Field(default_factory=StorytellerStateUpdates)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    loop_signal: LoopSignal
    completion_evidence: StorytellerCompletionEvidence


class DeveloperStateUpdates(BaseModel):
    model_config = ConfigDict(extra="allow")

    dep: dict[str, Any] = Field(default_factory=dict)


class DeveloperDiagnosis(BaseModel):
    model_config = ConfigDict(extra="allow")

    error_class: str
    root_cause: str | None = None
    offending_columns: list[str] = Field(default_factory=list)
    smallest_fix: str | None = None


class DeveloperCompletionEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    input_contract_valid: bool
    required_outputs_present: bool
    execution_succeeded: bool
    artifacts_verified: bool
    submission_schema_matches_template: bool | None = None
    reproducibility_checks_passed: bool
    safe_for_downstream_use: bool


class DeveloperOutput(BaseModel):
    assignment_id: str
    agent: Literal["developer"]
    mode: str | None = None
    status: DeveloperStatus
    summary: str
    state_updates: DeveloperStateUpdates = Field(default_factory=DeveloperStateUpdates)
    diagnosis: DeveloperDiagnosis | None = None
    fix_attempts: list[dict[str, Any]] = Field(default_factory=list)
    operations: list[dict[str, Any]] = Field(default_factory=list)
    validations: list[dict[str, Any]] = Field(default_factory=list)
    artifacts: list[dict[str, Any]] = Field(default_factory=list)
    assumptions: list[dict[str, Any]] = Field(default_factory=list)
    risks: list[dict[str, Any]] = Field(default_factory=list)
    blockers: list[dict[str, Any]] = Field(default_factory=list)
    handoffs: list[dict[str, Any]] = Field(default_factory=list)
    loop_signal: LoopSignal | None = None
    completion_evidence: DeveloperCompletionEvidence | None = None


class DomainSituationAssessment(BaseModel):
    model_config = ConfigDict(extra="allow")

    resources: list[str] = Field(default_factory=list)
    requirements: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class DomainOutput(BaseModel):
    business_objectives: str
    situation_assessment: DomainSituationAssessment
    data_mining_goal: str
    success_criterion: dict[str, Any]
    data_description_notes: list[dict[str, Any]] = Field(default_factory=list)
    feature_hints: list[dict[str, Any]] = Field(default_factory=list)
    domain_data_quality_flags: list[dict[str, Any]] = Field(default_factory=list)
    loop_a_recommendation: dict[str, Any]
    assumptions: list[str] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


class PmDirectiveOutput(BaseModel):
    action: PmAction
    reason: str = ""
    target_substep: str | None = None
    loop_label: str | None = None
    loop_to_phase: int | str | None = None

    @field_validator("loop_to_phase", mode="before")
    @classmethod
    def _coerce_loop_phase(cls, value: Any) -> int | str | None:
        if value is None or isinstance(value, bool):
            return None
        return value


_AGENT_MODELS: dict[str, type[BaseModel]] = {
    "data_engineer": DataEngineerOutput,
    "data_scientist": DataScientistOutput,
    "storyteller": StorytellerOutput,
    "developer": DeveloperOutput,
    "domain": DomainOutput,
    "pm": PmDirectiveOutput,
}

_SPECIALIST_AGENTS = frozenset({"data_engineer", "data_scientist", "storyteller"})


def _format_validation_errors(exc: ValidationError) -> list[str]:
    return [
        f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}"
        for err in exc.errors()
    ]


def _coerce_dict_list_item(item: Any) -> dict[str, Any]:
    """Wrap a bare string as a structured list item."""
    if isinstance(item, dict):
        return item
    return {"statement": str(item)}


def _coerce_dict_list(value: Any) -> list[dict[str, Any]]:
    """Normalize a field that must be a list of objects."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        return [{"statement": value.strip()}]
    if not isinstance(value, list):
        return [{"statement": str(value)}]
    return [_coerce_dict_list_item(item) for item in value]


def normalize_agent_output(
    agent_name: str,
    data: dict[str, Any],
    *,
    substep: str | None = None,
) -> dict[str, Any]:
    """Apply deterministic shape fixes in place before validation (idempotent)."""
    if agent_name not in _SPECIALIST_AGENTS and agent_name != "developer":
        return data

    if substep and agent_name in _SPECIALIST_AGENTS:
        aid = str(data.get("assignment_id") or "")
        if not aid.startswith("debug-"):
            data["assignment_id"] = substep

    array_fields = list(_ARRAY_FIELDS_SPECIALIST)
    if agent_name == "data_scientist":
        array_fields.extend(_ARRAY_FIELDS_DS_EXTRA)

    for name in array_fields:
        value = data.get(name)
        if value is None:
            continue
        if isinstance(value, str):
            data[name] = [] if not value.strip() else [value]
            value = data[name]
        if name in _DICT_LIST_FIELDS:
            data[name] = _coerce_dict_list(value)
        elif isinstance(value, list):
            data[name] = value

    su = data.get("state_updates")
    if isinstance(su, str):
        data["state_updates"] = {}
    elif isinstance(su, dict):
        for section in su.values():
            if not isinstance(section, dict):
                continue
            for key, val in list(section.items()):
                if key.endswith("_assumptions") and isinstance(val, list):
                    section[key] = [
                        str(item) if not isinstance(item, str) else item
                        for item in val
                    ]

    return data


def _check_array_fields(data: dict[str, Any], fields: tuple[str, ...]) -> list[str]:
    errors: list[str] = []
    for name in fields:
        value = data.get(name)
        if value is None:
            continue
        if isinstance(value, str):
            errors.append(f"{name}: expected array, got empty string")
        elif not isinstance(value, list):
            errors.append(f"{name}: expected array, got {type(value).__name__}")
    return errors


def _check_state_updates(data: dict[str, Any]) -> list[str]:
    su = data.get("state_updates")
    if su is None:
        return []
    if isinstance(su, str):
        return ["state_updates: expected object, got empty string"]
    if not isinstance(su, dict):
        return [f"state_updates: expected object, got {type(su).__name__}"]
    return []


def _check_debug_wrapper(data: dict[str, Any], agent_name: str) -> list[str]:
    """Reject Developer DEBUG meta-responses masquerading as upstream agent output."""
    if agent_name == "developer":
        return []
    errors: list[str] = []
    assignment_id = str(data.get("assignment_id", ""))
    status = str(data.get("status", ""))
    agent = str(data.get("agent", ""))
    if assignment_id.startswith("debug-"):
        errors.append(
            f"assignment_id: debug wrapper '{assignment_id}' is not valid for {agent_name}"
        )
    if status == "FIXED":
        errors.append(f"status: FIXED is reserved for developer DEBUG responses")
    if agent and agent != agent_name:
        errors.append(f"agent: expected '{agent_name}', got '{agent}'")
    if "mode" in data and data.get("mode") == "DEBUG":
        errors.append("mode: DEBUG metadata must not appear in specialist agent output")
    if "diagnosis" in data:
        errors.append("diagnosis: developer-only field present in specialist output")
    return errors


def _minimal_de_response(substep: str = "2.4") -> dict[str, Any]:
    return {
        "assignment_id": substep,
        "agent": "data_engineer",
        "status": "COMPLETED",
        "summary": "Data quality verified.",
        "state_updates": {
            "du": {"data_quality_report": {"blockers": [], "tolerable": []}},
            "dp": {},
        },
        "evidence": [],
        "decisions": [],
        "operations": [],
        "quality_findings": [],
        "validations": [],
        "artifacts": [],
        "assumptions": [],
        "risks": [],
        "blockers": [],
        "handoffs": [],
        "loop_signal": {
            "recommended": False,
            "contour": "NONE",
            "reason": None,
            "evidence_ids": [],
        },
        "completion_evidence": {
            "input_contract_valid": True,
            "required_outputs_present": True,
            "execution_succeeded": True,
            "artifacts_verified": True,
            "leakage_checks_passed": True,
            "reproducibility_checks_passed": True,
            "safe_for_downstream_use": True,
        },
    }


def validate_agent_output(
    agent_name: str,
    data: dict[str, Any] | None,
    *,
    substep: str | None = None,
    normalize: bool = True,
) -> list[str]:
    """Return human-readable schema violations (empty list == valid)."""
    if not data or not isinstance(data, dict):
        return ["output is not a JSON object"]

    if normalize:
        normalize_agent_output(agent_name, data, substep=substep)

    errors = _check_debug_wrapper(data, agent_name)
    errors.extend(_check_state_updates(data))

    if agent_name in _SPECIALIST_AGENTS:
        errors.extend(_check_array_fields(data, _ARRAY_FIELDS_SPECIALIST))
    if agent_name == "data_scientist":
        errors.extend(_check_array_fields(data, _ARRAY_FIELDS_DS_EXTRA))

    if substep and agent_name in _SPECIALIST_AGENTS:
        aid = str(data.get("assignment_id", ""))
        if aid and aid != substep:
            errors.append(f"assignment_id: expected '{substep}', got '{aid}'")

    if agent_name == "pm" and "action" not in data:
        return errors
    if agent_name == "domain" and "business_objectives" not in data:
        return errors

    model = _AGENT_MODELS.get(agent_name)
    if model is None:
        return errors

    try:
        model.model_validate(data)
    except ValidationError as exc:
        errors.extend(_format_validation_errors(exc))

    return errors


def schema_ok(
    agent_name: str,
    data: dict[str, Any] | None,
    *,
    substep: str | None = None,
) -> bool:
    return not validate_agent_output(agent_name, data, substep=substep)


def agent_json_schema(agent_name: str) -> dict[str, Any]:
    """JSON Schema for OpenAI structured outputs (strict mode)."""
    model = _AGENT_MODELS.get(agent_name)
    if model is None:
        return {"type": "object", "additionalProperties": True}
    schema = model.model_json_schema()
    return _prepare_strict_schema(schema)


def output_model_for_agent(agent_name: str) -> type[BaseModel] | None:
    """Pydantic model for CrewAI ``response_format`` when supported."""
    return _AGENT_MODELS.get(agent_name)


def _prepare_strict_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize Pydantic schema for OpenAI strict json_schema mode."""
    defs = schema.pop("$defs", None) or schema.pop("definitions", None)
    if defs:
        schema["$defs"] = defs

    def _walk(node: dict[str, Any]) -> None:
        if "additionalProperties" not in node and node.get("type") == "object":
            node["additionalProperties"] = False
        props = node.get("properties")
        if isinstance(props, dict):
            node["required"] = list(props.keys())
            for child in props.values():
                if isinstance(child, dict):
                    _walk(child)
        for key in ("items", "anyOf", "oneOf", "allOf"):
            child = node.get(key)
            if isinstance(child, dict):
                _walk(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, dict):
                        _walk(item)
        ref = node.get("$ref")
        if ref and defs:
            name = ref.rsplit("/", 1)[-1]
            target = defs.get(name)
            if isinstance(target, dict):
                _walk(target)

    _walk(schema)
    return schema


def minimal_data_engineer_output(
    substep: str,
    *,
    state_updates: dict[str, Any] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    payload = _minimal_de_response(substep)
    if state_updates:
        for section, values in state_updates.items():
            payload["state_updates"][section].update(values)
    if summary:
        payload["summary"] = summary
    return payload


def minimal_data_scientist_output(
    substep: str,
    *,
    state_updates: dict[str, Any] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "assignment_id": substep,
        "agent": "data_scientist",
        "status": "COMPLETED",
        "summary": summary or f"DS {substep}",
        "state_updates": {"du": {}, "md": {}, "ev": {}},
        "model_runs": [],
        "evidence": [],
        "decisions": [],
        "validations": [],
        "leakage_checks": [],
        "diagnostics": [],
        "artifacts": [],
        "assumptions": [],
        "risks": [],
        "blockers": [],
        "handoffs": [],
        "loop_signal": {
            "recommended": False,
            "contour": "NONE",
            "reason": None,
            "evidence_ids": [],
        },
        "completion_evidence": {
            "input_contract_valid": True,
            "required_outputs_present": True,
            "execution_succeeded": True,
            "baseline_established": True,
            "leakage_checks_passed": True,
            "uncertainty_reported": True,
            "evaluated_against_success_criterion": True,
            "safe_for_downstream_use": True,
        },
    }
    if state_updates:
        for section, values in state_updates.items():
            payload["state_updates"][section].update(values)
    return payload


def minimal_storyteller_output(
    substep: str,
    *,
    story_spec: dict[str, Any] | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    spec = story_spec or {
        "detected_problem_type": "binary_classification",
        "selected_model": "gradient_boosting",
        "selected_metrics": ["accuracy", "balanced_accuracy"],
        "interpretations": [
            {
                "metric": "accuracy",
                "value": 0.82,
                "interpretation": "The model is correct on most out-of-fold predictions.",
            },
        ],
        "methodological_warnings": [],
        "storytelling_summary": "Model performance is acceptable for a baseline.",
        "next_steps": ["Consider additional feature engineering."],
    }
    return {
        "assignment_id": substep,
        "agent": "storyteller",
        "status": "COMPLETED",
        "summary": summary or f"Storyteller {substep}",
        "story_spec": spec,
        "state_updates": {"dep": {}},
        "artifacts": [],
        "assumptions": [],
        "risks": [],
        "blockers": [],
        "handoffs": [],
        "loop_signal": {
            "recommended": False,
            "contour": "NONE",
            "reason": None,
            "evidence_ids": [],
        },
        "completion_evidence": {
            "input_contract_valid": True,
            "required_outputs_present": True,
            "execution_succeeded": True,
            "evaluation_bundle_present": True,
            "figures_generated": True,
            "report_rendered": substep == "6.3",
            "safe_for_downstream_use": True,
        },
    }


def minimal_agent_output(
    agent_name: str,
    substep: str,
    *,
    state_updates: dict[str, Any] | None = None,
    summary: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a schema-valid minimal payload for tests and fake LLM stubs."""
    if agent_name == "pm":
        payload: dict[str, Any] = {
            "action": "advance",
            "target_substep": None,
            "reason": summary or "ok",
        }
    elif agent_name == "data_engineer":
        payload = minimal_data_engineer_output(
            substep, state_updates=state_updates, summary=summary,
        )
    elif agent_name == "data_scientist":
        payload = minimal_data_scientist_output(
            substep, state_updates=state_updates, summary=summary,
        )
    elif agent_name == "storyteller":
        payload = minimal_storyteller_output(
            substep,
            story_spec=(extra or {}).get("story_spec") if extra else None,
            summary=summary,
        )
    else:
        payload = {}
    if extra:
        payload.update(extra)
    return payload


def _compact_model_shape(model: type[BaseModel]) -> str:
    """Required/optional top-level keys only — not full Pydantic JSON Schema."""
    required = set(model.model_json_schema().get("required") or [])
    lines = [f"Top-level keys for {model.__name__}:"]
    for name, field in model.model_fields.items():
        ann = getattr(field.annotation, "__name__", None) or str(field.annotation)
        flag = "required" if name in required else "optional"
        lines.append(f'  "{name}": {ann} ({flag})')
    return "\n".join(lines)


def schema_hint_for_agent(agent_name: str, *, substep: str | None = None) -> str:
    """Compact schema hint string for task prompts (not full model_json_schema)."""
    model = _AGENT_MODELS.get(agent_name)
    if model is None:
        return "{}"
    parts = [_compact_model_shape(model)]
    if agent_name in _SPECIALIST_AGENTS:
        parts.append(_SCHEMA_SHAPE_NOTES.strip())
        if substep:
            parts.append(
                f'assignment_id must be exactly "{substep}" '
                "(the current CRISP-DM substep id)."
            )
        example = _SUBSTEP_SCHEMA_EXAMPLES.get((agent_name, substep or ""))
        if example:
            parts.append(example)
    return "\n\n".join(parts)
