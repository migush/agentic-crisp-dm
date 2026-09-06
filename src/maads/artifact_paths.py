"""Purpose-tagged artifact paths under ``runs/<run_id>/``."""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


class RunPaths:
    """Resolved paths for one run directory."""

    def __init__(self, run_dir: Path) -> None:
        self.run_dir = Path(run_dir).resolve()

    @property
    def manifest(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def collected(self) -> Path:
        return self.run_dir / "collected"

    @property
    def derived(self) -> Path:
        return self.run_dir / "derived"

    @property
    def deliverables(self) -> Path:
        return self.run_dir / "deliverables"

    @property
    def reports(self) -> Path:
        return self.run_dir / "reports"

    @property
    def trace_legacy(self) -> Path:
        return self.run_dir / "trace"

    def communications_jsonl(self) -> Path:
        primary = self.collected / "communications.jsonl"
        if primary.is_file():
            return primary
        legacy = self.trace_legacy / "communications.jsonl"
        return legacy if legacy.is_file() else primary

    def communications_summary(self) -> Path:
        derived = self.derived / "communications_summary.json"
        if derived.is_file():
            return derived
        legacy = self.trace_legacy / "communications_summary.json"
        return legacy if legacy.is_file() else derived

    def trace_json(self) -> Path:
        derived = self.derived / "trace.json"
        if derived.is_file():
            return derived
        legacy = self.trace_legacy / "trace.json"
        return legacy if legacy.is_file() else derived

    def live_summary(self) -> Path:
        return self.derived / "live_summary.json"

    def events_jsonl(self) -> Path:
        return self.collected / "events.jsonl"

    def sandbox_exec(self) -> Path:
        return self.run_dir / "sandbox" / "exec"

    def sandbox_manifest(self) -> Path:
        collected = self.collected / "sandbox" / "manifest.jsonl"
        if collected.is_file():
            return collected
        legacy = self.sandbox_exec() / "manifest.jsonl"
        return legacy if legacy.is_file() else collected


def ensure_run_layout(
    run_dir: Path, *, run_id: str, case_id: str, model: str | None = None,
) -> RunPaths:
    """Create bucket directories and an initial manifest stub."""
    paths = RunPaths(run_dir)
    paths.collected.mkdir(parents=True, exist_ok=True)
    (paths.collected / "sandbox").mkdir(parents=True, exist_ok=True)
    paths.derived.mkdir(parents=True, exist_ok=True)
    paths.deliverables.mkdir(parents=True, exist_ok=True)
    paths.reports.mkdir(parents=True, exist_ok=True)
    paths.trace_legacy.mkdir(parents=True, exist_ok=True)
    if not paths.manifest.is_file():
        write_manifest_stub(paths, run_id=run_id, case_id=case_id, model=model)
    return paths


def write_manifest_stub(
    paths: RunPaths, *, run_id: str, case_id: str, model: str | None = None,
) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "model": model,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "buckets": {
            "collected": [
                "collected/communications.jsonl",
                "collected/events.jsonl",
                "collected/sandbox/manifest.jsonl",
                "sandbox/exec/",
            ],
            "derived": [
                "derived/live_summary.json",
                "derived/communications_summary.json",
                "derived/trace.json",
                "status.json",
                "process.json",
                "state.json",
            ],
            "deliverables": [
                "prep/",
                "submission.csv",
                "final_report.md",
                "train.parquet",
                "test.parquet",
            ],
            "reports": [
                "reports/postmortem.json",
                "reports/case_report.json",
                "reports/case_report.md",
                "reports/execution_analysis.json",
                "reports/execution_analysis.md",
                "reports/case_workbook.ipynb",
                "reports/workbook_context.json",
                "reports/handoff_standard.zip",
                "reports/improvement_bundle.json",
            ],
        },
        "legacy_paths": {
            "trace/communications.jsonl": "collected/communications.jsonl",
            "trace/trace.json": "derived/trace.json",
        },
    }
    paths.manifest.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def seal_manifest(
    paths: RunPaths,
    *,
    ended_at: str,
    workflow_complete: bool,
    ml_success: bool,
    halt_reason: str | None,
    duration_ms: int | None = None,
) -> None:
    payload: dict[str, Any] = {}
    if paths.manifest.is_file():
        payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
    payload.update({
        "ended_at": ended_at,
        "workflow_complete": workflow_complete,
        "ml_success": ml_success,
        "halt_reason": halt_reason,
        "reports_generated_at": payload.get("reports_generated_at"),
    })
    if duration_ms is not None:
        payload["duration_ms"] = duration_ms
    paths.manifest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = RunPaths(run_dir).manifest
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def update_runs_index(case_root_path: Path, *, run_id: str, manifest: dict[str, Any]) -> None:
    """Maintain ``artifacts/<case>/runs_index.json`` for cross-run contrast."""
    index_path = case_root_path / "runs_index.json"
    entries: list[dict[str, Any]] = []
    if index_path.is_file():
        try:
            raw = json.loads(index_path.read_text(encoding="utf-8"))
            entries = list(raw.get("runs") or [])
        except (json.JSONDecodeError, OSError):
            entries = []
    entries = [e for e in entries if e.get("run_id") != run_id]
    entries.append({
        "run_id": run_id,
        "model": manifest.get("model"),
        "started_at": manifest.get("started_at"),
        "ended_at": manifest.get("ended_at"),
        "workflow_complete": manifest.get("workflow_complete"),
        "ml_success": manifest.get("ml_success"),
        "halt_reason": manifest.get("halt_reason"),
        "artifact_dir": f"runs/{run_id}",
    })
    entries.sort(key=lambda e: e.get("ended_at") or e.get("started_at") or "", reverse=True)
    # Atomic write: concurrent same-case completions must not corrupt the index.
    tmp_path = index_path.with_name(f"{index_path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(
        json.dumps({"case_id": manifest.get("case_id"), "runs": entries}, indent=2),
        encoding="utf-8",
    )
    os.replace(tmp_path, index_path)
