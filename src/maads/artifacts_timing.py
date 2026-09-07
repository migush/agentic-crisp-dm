"""Resolve and backfill run timing fields on artifact manifests and reports."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from maads.artifact_paths import RunPaths, load_manifest
from maads.observability.schema import TraceRun


def _read_trace_optional(artifact_dir: Path) -> TraceRun:
    path = RunPaths(artifact_dir).trace_json()
    if path.is_file():
        return TraceRun.model_validate_json(path.read_text(encoding="utf-8"))
    return TraceRun(run_id=artifact_dir.name, events=[])


def resolve_run_timing(artifact_dir: Path) -> dict[str, Any]:
    """Best-effort started_at, ended_at, duration_ms for a run directory."""
    paths = RunPaths(artifact_dir)
    manifest = load_manifest(artifact_dir)
    trace = _read_trace_optional(artifact_dir)

    started_at = (
        trace.started_at.isoformat()
        if trace.started_at
        else manifest.get("started_at")
    )
    ended_at = (
        trace.ended_at.isoformat()
        if trace.ended_at
        else manifest.get("ended_at")
    )
    duration_ms = manifest.get("duration_ms")
    if trace.events:
        duration_ms = int(trace.events[-1].ts_mono_ms)

    for name in ("execution_analysis.json", "postmortem.json"):
        report_path = paths.reports / name
        if duration_ms is not None and started_at and ended_at:
            break
        if not report_path.is_file():
            continue
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        started_at = started_at or report.get("started_at")
        ended_at = ended_at or report.get("ended_at")
        if duration_ms is None and report.get("duration_ms") is not None:
            duration_ms = int(report["duration_ms"])

    if duration_ms is None and started_at and ended_at:
        try:
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(ended_at)
            duration_ms = int((end - start).total_seconds() * 1000)
        except ValueError:
            pass

    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_ms": duration_ms,
    }


def _patch_json(path: Path, updates: dict[str, Any], *, dry_run: bool) -> bool:
    if not updates or not path.is_file():
        return False
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    changed = False
    for key, value in updates.items():
        if value is not None and payload.get(key) != value:
            payload[key] = value
            changed = True
    if changed and not dry_run:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return changed


def backfill_run_timing(artifact_dir: Path, *, dry_run: bool = False) -> dict[str, Any]:
    """Write resolved timing onto manifest and report JSON for one run."""
    paths = RunPaths(artifact_dir)
    timing = resolve_run_timing(artifact_dir)
    changed: list[str] = []

    manifest_updates = {
        k: timing[k]
        for k in ("started_at", "ended_at", "duration_ms")
        if timing.get(k) is not None
    }
    if manifest_updates and paths.manifest.is_file():
        if _patch_json(paths.manifest, manifest_updates, dry_run=dry_run):
            changed.append("manifest.json")

    for name in ("execution_analysis.json", "postmortem.json"):
        report_path = paths.reports / name
        report_updates = {
            k: timing[k]
            for k in ("started_at", "ended_at", "duration_ms")
            if timing.get(k) is not None
        }
        if _patch_json(report_path, report_updates, dry_run=dry_run):
            changed.append(f"reports/{name}")

    return {"run_id": artifact_dir.name, "timing": timing, "changed": changed}


def backfill_all_runs(
    artifact_root: Path,
    *,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Backfill timing for every run directory under ``artifact_root``."""
    results: list[dict[str, Any]] = []
    if not artifact_root.is_dir():
        return results
    for case_dir in sorted(artifact_root.iterdir()):
        if not case_dir.is_dir():
            continue
        runs_dir = case_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for run_dir in sorted(runs_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            results.append(backfill_run_timing(run_dir, dry_run=dry_run))
    return results
