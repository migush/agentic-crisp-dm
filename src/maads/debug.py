"""Developer DEBUG mode — on-call repair for PythonExec and JSON parse failures."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from maads.codegen import Contract, _extract_code, _header, _last_json_line
from maads.crew import CrewKickoffError, _extract_json
from maads.prompts import JSON_EXPECTED_OUTPUT
from maads.prompts.identities.developer import format_developer_debug_task
from maads.state import CrispDMState
from maads.token_budget import TokenBudgetExceeded, debug_retries
from maads.tools import ExecResult, PythonExec

MAX_DEBUG_RETRIES = 3


@dataclass
class DebugOutcome:
    """Result of a Developer DEBUG intervention."""

    status: str  # FIXED | STUCK
    payload: dict[str, Any] | None = None
    code: str | None = None
    diagnosis: dict[str, Any] = field(default_factory=dict)
    fix_attempts: list[dict[str, Any]] = field(default_factory=list)
    repair_kind: str = "none"
    schema_ok: bool = False
    schema_errors: list[str] = field(default_factory=list)


def classify_exec_error(res: ExecResult | None, error_text: str = "") -> str:
    """Label a PythonExec failure using stderr/stdout evidence."""
    text = (error_text or (res.stderr if res else "") or "").lower()
    if res and res.timed_out:
        return "timeout"
    if "syntaxerror" in text or "indentationerror" in text:
        return "syntax_error"
    if "keyerror" in text or "not in index" in text or "no column" in text:
        return "schema_error"
    if "shape" in text or "n_features" in text or "dimension" in text:
        return "shape_mismatch"
    if "could not convert" in text or "dtype" in text or "typeerror" in text:
        return "type_error"
    if "memoryerror" in text:
        return "oom"
    if "has no attribute" in text or "importerror" in text or "modulenotfounderror" in text:
        return "lib_version"
    if "leakage" in text or "contamination" in text:
        return "leakage_signal"
    return "other"


def _schema_columns(state: CrispDMState) -> list[str]:
    merged = state.dp.merged_data or {}
    cols_train = merged.get("columns_train")
    if isinstance(cols_train, list) and cols_train:
        return [str(c) for c in cols_train]
    dataset_train = (state.dp.dataset or {}).get("train")
    if dataset_train:
        try:
            import pandas as pd

            return [str(c) for c in pd.read_parquet(dataset_train).columns]
        except Exception:
            pass
    report = state.du.data_description_report or {}
    cols = report.get("columns")
    if isinstance(cols, list):
        return [str(c) for c in cols]
    return []


def _excerpt(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[-limit:]


def debug_python_exec(
    *,
    pyexec: PythonExec,
    state: CrispDMState,
    artifact_dir: Path,
    requesting_agent: str,
    failing_code: str,
    header_vars: dict[str, Any],
    contract: Contract,
    contract_hint: str,
    last_error: str,
    last_exec: ExecResult | None = None,
    max_retries: int | None = None,
) -> DebugOutcome:
    """Developer diagnoses and re-executes failing agent-authored Python."""
    if max_retries is None:
        max_retries = debug_retries()
    if not failing_code:
        return DebugOutcome(
            status="STUCK",
            diagnosis={"error_class": "other", "root_cause": "no failing code to repair"},
        )

    error_class = classify_exec_error(last_exec, last_error)
    diagnosis = {
        "error_class": error_class,
        "root_cause": _excerpt(last_error, 500),
        "requesting_agent": requesting_agent,
    }
    fix_attempts: list[dict[str, Any]] = []
    header = _header(header_vars)
    prior_code = failing_code
    prior_error = last_error

    state.append_log(
        "developer",
        f"DEBUG python_exec for {requesting_agent} @ {state.substep} "
        f"({error_class}): {(last_error or '')[:200]}",
        level="warn",
    )

    for attempt in range(1, max_retries + 1):
        instruction = format_developer_debug_task(
            state,
            artifact_dir,
            failure_kind="python_exec",
            requesting_agent=requesting_agent,
            error_class=error_class,
            last_error=prior_error,
            stderr_excerpt=_excerpt(last_exec.stderr if last_exec else prior_error),
            failing_code=prior_code,
            header_var_names=list(header_vars.keys()),
            contract_hint=contract_hint,
            schema_columns=_schema_columns(state),
            attempt=attempt,
            max_retries=max_retries,
        )
        try:
            from maads.crew import run_text_task

            raw = run_text_task(
                "developer",
                instruction,
                state,
                expected_output="A Python code block.",
            )
        except TokenBudgetExceeded:
            raise
        except (CrewKickoffError, RuntimeError) as exc:
            fix_attempts.append({
                "attempt": attempt,
                "change": "developer LLM call",
                "exec_status": "FAILED",
                "stderr_excerpt": str(exc)[:500],
            })
            prior_error = f"developer LLM failed: {exc}"
            continue

        fixed = _extract_code(raw)
        if not fixed:
            fix_attempts.append({
                "attempt": attempt,
                "change": "no code block in developer response",
                "exec_status": "FAILED",
                "stderr_excerpt": _excerpt(raw, 300),
            })
            prior_error = "developer returned no Python code block"
            prior_code = failing_code
            continue

        full_code = header + fixed
        res = pyexec.run(
            full_code,
            label=f"developer_debug_{requesting_agent}_attempt{attempt}",
        )
        if not res.ok:
            fix_attempts.append({
                "attempt": attempt,
                "change": "re-execute fixed code",
                "exec_status": "TIMED_OUT" if res.timed_out else "FAILED",
                "stdout_excerpt": _excerpt(res.stdout, 300),
                "stderr_excerpt": _excerpt(res.stderr, 500),
            })
            prior_code = fixed
            prior_error = (res.stderr or "").strip()[-1500:] or "non-zero exit"
            last_exec = res
            error_class = classify_exec_error(res, prior_error)
            continue

        payload = _last_json_line(res.stdout)
        if payload is None:
            fix_attempts.append({
                "attempt": attempt,
                "change": "fixed code ran but no JSON on last stdout line",
                "exec_status": "FAILED",
                "stdout_excerpt": _excerpt(res.stdout, 300),
                "stderr_excerpt": "",
            })
            prior_code = fixed
            prior_error = "fixed code ran but printed no JSON object on its last line"
            continue

        errors = contract(payload)
        if errors:
            fix_attempts.append({
                "attempt": attempt,
                "change": "contract validation after fix",
                "exec_status": "FAILED",
                "stderr_excerpt": "; ".join(errors)[:500],
            })
            prior_code = fixed
            prior_error = "output failed validation: " + "; ".join(errors)
            continue

        fix_attempts.append({
            "attempt": attempt,
            "change": "developer fix accepted",
            "exec_status": "EXECUTED",
            "stdout_excerpt": _excerpt(res.stdout, 200),
            "stderr_excerpt": "",
        })
        state.append_log(
            "developer",
            f"DEBUG fixed {requesting_agent} python_exec on attempt {attempt}",
        )
        return DebugOutcome(
            status="FIXED",
            payload=payload,
            code=full_code,
            diagnosis=diagnosis,
            fix_attempts=fix_attempts,
        )

    state.append_log(
        "developer",
        f"DEBUG stuck on {requesting_agent} python_exec after {max_retries} attempts",
        level="warn",
    )
    return DebugOutcome(
        status="STUCK",
        diagnosis={**diagnosis, "smallest_fix": "exhausted debug retry budget"},
        fix_attempts=fix_attempts,
    )


def debug_json_parse(
    *,
    state: CrispDMState,
    artifact_dir: Path,
    requesting_agent: str,
    raw_text: str,
    schema_hint: str = "",
    instruction: str = "",
    failure_kind: str = "json_parse",
    invalid_payload: dict[str, Any] | None = None,
    max_retries: int | None = None,
) -> DebugOutcome:
    """Developer repairs malformed structured output from another agent."""
    from maads.output_contracts import normalize_agent_output, validate_agent_output

    if max_retries is None:
        max_retries = debug_retries()
    text = (raw_text or "").strip()
    if not text and invalid_payload is None:
        return DebugOutcome(
            status="STUCK",
            diagnosis={"error_class": "json_parse", "root_cause": "empty LLM output"},
        )

    def _validated(payload: dict[str, Any] | None, repair_kind: str) -> DebugOutcome:
        if payload is None:
            return DebugOutcome(
                status="STUCK",
                repair_kind=repair_kind,
                diagnosis={"error_class": failure_kind, "root_cause": "no parseable JSON"},
            )
        normalize_agent_output(
            requesting_agent, payload, substep=state.substep,
        )
        errors = validate_agent_output(
            requesting_agent, payload, substep=state.substep, normalize=False,
        )
        if errors:
            return DebugOutcome(
                status="STUCK",
                payload=payload,
                repair_kind=repair_kind,
                schema_ok=False,
                schema_errors=errors,
                diagnosis={
                    "error_class": "json_schema",
                    "root_cause": "; ".join(errors[:3]),
                },
            )
        return DebugOutcome(
            status="FIXED",
            payload=payload,
            repair_kind=repair_kind,
            schema_ok=True,
            diagnosis={"error_class": failure_kind, "root_cause": repair_kind},
        )

    # Deterministic repair pass (same helpers as crew.run_json_task).
    source_text = text
    if failure_kind == "json_schema" and invalid_payload is not None:
        source_text = json.dumps(invalid_payload)
    parsed = _extract_json(source_text) if source_text else invalid_payload
    if parsed is not None and failure_kind == "json_parse":
        outcome = _validated(parsed, "deterministic")
        if outcome.status == "FIXED":
            state.append_log(
                "developer",
                f"DEBUG repaired JSON for {requesting_agent} deterministically",
            )
            outcome.fix_attempts = [{
                "attempt": 0,
                "change": "deterministic json repair",
                "exec_status": "EXECUTED",
            }]
            return outcome

    if parsed is not None and failure_kind == "json_schema":
        coerced = dict(invalid_payload or parsed)
        outcome = _validated(coerced, "deterministic_schema")
        if outcome.status == "FIXED":
            state.append_log(
                "developer",
                f"DEBUG normalized schema for {requesting_agent} deterministically",
            )
            outcome.fix_attempts = [{
                "attempt": 0,
                "change": "deterministic schema normalization",
                "exec_status": "EXECUTED",
            }]
            return outcome
        prior_schema_errors = outcome.schema_errors
        malformed_source = coerced
    else:
        prior_schema_errors = []
        malformed_source = invalid_payload

    state.append_log(
        "developer",
        f"DEBUG {failure_kind} for {requesting_agent} @ {state.substep}",
        level="warn",
    )

    malformed = _excerpt(
        source_text or json.dumps(malformed_source or {}),
        4000,
    )
    prior_error = (
        "; ".join(prior_schema_errors[:5])
        if prior_schema_errors
        else (
            "upstream agent returned schema-invalid JSON"
            if failure_kind == "json_schema"
            else "upstream agent returned non-JSON or invalid JSON"
        )
    )
    fix_attempts: list[dict[str, Any]] = []
    last_outcome: DebugOutcome | None = None

    for attempt in range(1, max_retries + 1):
        debug_instruction = format_developer_debug_task(
            state,
            artifact_dir,
            failure_kind=failure_kind,
            requesting_agent=requesting_agent,
            error_class=failure_kind,
            last_error=prior_error,
            stderr_excerpt="",
            failing_code="",
            header_var_names=[],
            contract_hint=schema_hint,
            schema_columns=_schema_columns(state),
            malformed_json=malformed,
            task_instruction=instruction,
            attempt=attempt,
            max_retries=max_retries,
            schema_errors=prior_schema_errors or None,
        )
        try:
            from maads.crew import run_text_task

            raw = run_text_task(
                "developer",
                debug_instruction,
                state,
                expected_output=JSON_EXPECTED_OUTPUT,
            )
        except TokenBudgetExceeded:
            raise
        except (CrewKickoffError, RuntimeError) as exc:
            fix_attempts.append({
                "attempt": attempt,
                "change": "developer LLM call",
                "exec_status": "FAILED",
                "stderr_excerpt": str(exc)[:500],
            })
            prior_error = f"developer LLM failed: {exc}"
            last_outcome = DebugOutcome(
                status="STUCK",
                repair_kind="developer_llm",
                diagnosis={
                    "error_class": failure_kind,
                    "root_cause": prior_error,
                },
            )
            continue

        repaired = _extract_json(raw)
        if repaired is None:
            repaired = _extract_json(re.sub(r"^[^{]*", "", raw, count=1))

        outcome = _validated(repaired, "developer_llm")
        last_outcome = outcome
        if outcome.status == "FIXED":
            state.append_log(
                "developer",
                f"DEBUG repaired JSON for {requesting_agent} via developer LLM "
                f"(attempt {attempt})",
            )
            outcome.fix_attempts = fix_attempts + [{
                "attempt": attempt,
                "change": "developer json repair",
                "exec_status": "EXECUTED",
            }]
            return outcome

        fix_attempts.append({
            "attempt": attempt,
            "change": "developer json repair",
            "exec_status": "FAILED",
            "stderr_excerpt": (
                "; ".join(outcome.schema_errors[:5])
                if outcome.schema_errors
                else "could not repair malformed JSON"
            ),
        })
        prior_schema_errors = outcome.schema_errors
        prior_error = (
            "; ".join(outcome.schema_errors[:5])
            if outcome.schema_errors
            else "developer could not repair malformed JSON"
        )
        if repaired is not None:
            malformed = _excerpt(json.dumps(repaired), 4000)

    return DebugOutcome(
        status="STUCK",
        payload=last_outcome.payload if last_outcome else None,
        repair_kind="developer_llm",
        schema_ok=False,
        schema_errors=last_outcome.schema_errors if last_outcome else [],
        diagnosis={
            "error_class": failure_kind,
            "root_cause": (
                last_outcome.schema_errors[0]
                if last_outcome and last_outcome.schema_errors
                else "exhausted debug retry budget"
            ),
        },
        fix_attempts=fix_attempts,
    )
