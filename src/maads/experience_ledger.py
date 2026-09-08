"""Write-only ledger of LLM params + run outcomes (no auto-tuning yet)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from maads.artifact_paths import RunPaths
from maads.llm_params import AGENT_NAMES, resolve_all_agent_llm_params
from maads.outcome import ml_run_succeeded, workflow_complete
from maads.state import CrispDMState


_INFRA_HALT_MARKERS = (
    "preflight",
    "train file not found",
    "not found or not a file",
    "reasoning_effort",
    "unsupported value",
    "error code: 400",
    "invalid_request_error",
)


def _is_infra_halt(reason: str | None) -> bool:
    text = (reason or "").lower()
    return any(marker in text for marker in _INFRA_HALT_MARKERS)


def _sandbox_counts(paths: RunPaths) -> tuple[int, int]:
    manifest = paths.sandbox_manifest()
    if not manifest.is_file():
        return 0, 0
    attempts = failures = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        attempts += 1
        try:
            row = json.loads(line)
            if not row.get("ok"):
                failures += 1
        except json.JSONDecodeError:
            failures += 1
    return attempts, failures


def _parse_failure_count(artifact_dir: Path) -> int:
    postmortem = artifact_dir / "reports" / "postmortem.json"
    if not postmortem.is_file():
        return 0
    try:
        return int(json.loads(postmortem.read_text(encoding="utf-8")).get("parse_failures") or 0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0


def build_llm_param_experience_row(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Build one ledger row from the finished run (best-effort)."""
    paths = RunPaths(artifact_dir)
    rid = run_id or paths.run_dir.name
    try:
        params = resolve_all_agent_llm_params()
    except Exception:
        params = {}
    efforts = {
        name: (params[name].reasoning_effort if name in params else None)
        for name in AGENT_NAMES
    }
    models = {
        name: (params[name].model if name in params else None) for name in AGENT_NAMES
    }
    sandbox_attempts, sandbox_failures = _sandbox_counts(paths)
    halt = state.halt_reason
    return {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "run_id": rid,
        "case_id": state.case_id,
        "model": models.get("pm") or next((m for m in models.values() if m), None),
        "models_by_agent": models,
        "reasoning_effort_by_agent": efforts,
        "workflow_complete": workflow_complete(state),
        "ml_success": ml_run_succeeded(state),
        "parse_failures": _parse_failure_count(artifact_dir),
        "sandbox_attempts": sandbox_attempts,
        "sandbox_failures": sandbox_failures,
        "halt_reason": halt,
        "token_spend": dict(state.token_spend),
        "total_input_tokens": getattr(state, "total_input_tokens", 0),
        "total_output_tokens": getattr(state, "total_output_tokens", 0),
        "learnable": not _is_infra_halt(halt),
        "note": (
            "Write-only seed for future default tuning; infra halts "
            "(preflight, missing train, API 400) are not learnable."
        ),
    }


def append_llm_param_experience(
    state: CrispDMState,
    artifact_dir: Path,
    case_dir: Path,
    *,
    run_id: str | None = None,
) -> Path:
    """Write run-local JSON and append a case-level JSONL ledger row."""
    row = build_llm_param_experience_row(state, artifact_dir, run_id=run_id)
    reports = artifact_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    run_path = reports / "llm_param_experience.json"
    run_path.write_text(json.dumps(row, indent=2, default=str), encoding="utf-8")

    case_dir.mkdir(parents=True, exist_ok=True)
    ledger = case_dir / "llm_param_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, default=str) + "\n")
    return run_path
