"""Read trace artifacts from the filesystem."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from maads.artifact_paths import RunPaths, load_manifest
from maads.artifact_runs import resolve_active_run_dir
from maads.observability.llm_communications import LLMCommunicationRecord
from maads.observability.schema import TraceRun


def list_cases(artifact_root: Path) -> list[dict[str, Any]]:
    """Scan ``artifact_root/<case>/`` for the active run's ``status.json``."""
    if not artifact_root.is_dir():
        return []
    cases: list[dict[str, Any]] = []
    for child in sorted(artifact_root.iterdir()):
        if not child.is_dir():
            continue
        run_dir = resolve_active_run_dir(child)
        if run_dir is None:
            continue
        status_path = run_dir / "status.json"
        if not status_path.is_file():
            paths = RunPaths(run_dir)
            status_path = paths.derived / "status.json"
        if not status_path.is_file():
            continue
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        summary = _case_summary(child.name, run_dir, payload)
        runs_dir = child / "runs"
        summary["run_count"] = (
            sum(1 for p in runs_dir.iterdir() if p.is_dir())
            if runs_dir.is_dir()
            else 0
        )
        cases.append(summary)
    return cases


def _run_status(artifact_dir: Path, status: dict[str, Any]) -> str:
    """Derive run status (running/complete/halted) from status.json + trace."""
    if bool(status.get("halted")):
        return "halted"
    trace_path = RunPaths(artifact_dir).trace_json()
    if trace_path.is_file():
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            if trace.get("ended_at"):
                return "complete"
        except (json.JSONDecodeError, OSError):
            pass
    return "running"


def _case_summary(case_id: str, artifact_dir: Path, status: dict[str, Any]) -> dict[str, Any]:
    run_status = _run_status(artifact_dir, status)
    progress = status.get("progress") if isinstance(status.get("progress"), dict) else {}
    completed = progress.get("completed_substeps", status.get("completed_substeps"))
    model = load_manifest(artifact_dir).get("model")
    return {
        "case_id": case_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "status": run_status,
        "updated_at": status.get("updated_at"),
        "phase": status.get("phase"),
        "phase_name": status.get("phase_name"),
        "completed_substeps": completed,
        "total_substeps": status.get("total_substeps"),
        "model": model,
    }


def case_dir(artifact_root: Path, case_id: str, run_id: str | None = None) -> Path:
    case_path = artifact_root / case_id
    if run_id:
        run_dir = case_path / "runs" / run_id
        if run_dir.is_dir():
            return run_dir
        raise FileNotFoundError(f"Run not found: {case_id}/{run_id}")
    run_dir = resolve_active_run_dir(case_path)
    if run_dir is None:
        raise FileNotFoundError(f"Case not found: {case_id}")
    return run_dir


def read_live_summary(artifact_dir: Path) -> dict[str, Any]:
    paths = RunPaths(artifact_dir)
    path = paths.live_summary()
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return read_status(artifact_dir)


def read_status(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "status.json"
    if not path.is_file():
        path = RunPaths(artifact_dir).derived / "status.json"
    if not path.is_file():
        raise FileNotFoundError("status.json not found")
    return json.loads(path.read_text(encoding="utf-8"))


def read_trace(artifact_dir: Path) -> TraceRun:
    path = RunPaths(artifact_dir).trace_json()
    if not path.is_file():
        raise FileNotFoundError("trace.json not found")
    return TraceRun.model_validate_json(path.read_text(encoding="utf-8"))


def run_duration_ms(
    artifact_dir: Path,
    *,
    started_at: str | None = None,
    ended_at: str | None = None,
) -> int | None:
    """Wall-clock run duration from manifest, trace, reports, or timestamps."""
    from maads.artifacts_timing import resolve_run_timing

    manifest = load_manifest(artifact_dir)
    if manifest.get("duration_ms") is not None:
        return int(manifest["duration_ms"])

    timing = resolve_run_timing(artifact_dir)
    if timing.get("duration_ms") is not None:
        return int(timing["duration_ms"])

    if started_at and ended_at:
        try:
            start = datetime.fromisoformat(started_at)
            end = datetime.fromisoformat(ended_at)
            return int((end - start).total_seconds() * 1000)
        except ValueError:
            pass
    return None


def read_trace_optional(artifact_dir: Path, *, case_id: str = "") -> TraceRun:
    """Return trace data or an empty run shell when tracing has not flushed yet."""
    path = RunPaths(artifact_dir).trace_json()
    if path.is_file():
        return TraceRun.model_validate_json(path.read_text(encoding="utf-8"))
    status_path = artifact_dir / "status.json"
    case = case_id
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
            case = status.get("case_id") or case
        except (json.JSONDecodeError, OSError):
            pass
    return TraceRun(run_id=artifact_dir.name, case_id=case or None, events=[])


def read_communications(
    artifact_dir: Path,
    *,
    since_id: str | None = None,
    limit: int | None = None,
) -> list[LLMCommunicationRecord]:
    path = RunPaths(artifact_dir).communications_jsonl()
    if not path.is_file():
        return []
    records: list[LLMCommunicationRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(LLMCommunicationRecord.model_validate_json(line))
    if since_id:
        start = 0
        for i, rec in enumerate(records):
            if rec.id == since_id:
                start = i + 1
                break
        records = records[start:]
    if limit is not None and limit > 0:
        records = records[-limit:]
    return records


def read_communication(
    artifact_dir: Path,
    comm_id: str,
) -> LLMCommunicationRecord | None:
    for rec in read_communications(artifact_dir):
        if rec.id == comm_id:
            return rec
    return None


def read_communications_summary(artifact_dir: Path) -> dict[str, Any]:
    path = RunPaths(artifact_dir).communications_summary()
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_process_snapshot(artifact_dir: Path) -> dict[str, Any]:
    """Read live ``process.json`` or fall back to ``final_state.json``."""
    process_path = artifact_dir / "process.json"
    if not process_path.is_file():
        process_path = RunPaths(artifact_dir).derived / "process.json"
    if process_path.is_file():
        return json.loads(process_path.read_text(encoding="utf-8"))
    final_path = artifact_dir / "final_state.json"
    if final_path.is_file():
        return _process_snapshot_from_final_state(
            json.loads(final_path.read_text(encoding="utf-8"))
        )
    return {}


def _process_snapshot_from_final_state(state: dict[str, Any]) -> dict[str, Any]:
    """Map a ``final_state.json`` payload to the ``process.json`` shape."""
    from maads.state import CrispDMState

    parsed = CrispDMState.model_validate(state)
    from maads.run_status import _build_process_snapshot

    return _build_process_snapshot(parsed)


def read_state(artifact_dir: Path) -> dict[str, Any]:
    """Read live ``state.json`` or fall back to ``final_state.json``."""
    live_path = artifact_dir / "state.json"
    if live_path.is_file():
        raw = json.loads(live_path.read_text(encoding="utf-8"))
        if "state" in raw:
            return {
                "updated_at": raw.get("updated_at"),
                "source": "live",
                "state": raw["state"],
            }
        return {"updated_at": None, "source": "live", "state": raw}

    derived = RunPaths(artifact_dir).derived / "state_snapshot.json"
    if derived.is_file():
        raw = json.loads(derived.read_text(encoding="utf-8"))
        return {
            "updated_at": raw.get("updated_at"),
            "source": "live",
            "state": raw.get("state", raw),
        }

    final_path = artifact_dir / "final_state.json"
    if final_path.is_file():
        return {
            "updated_at": None,
            "source": "final",
            "state": json.loads(final_path.read_text(encoding="utf-8")),
        }

    raise FileNotFoundError("state.json not found")


def read_manifest(artifact_dir: Path) -> dict[str, Any]:
    return load_manifest(artifact_dir)


def read_report(artifact_dir: Path, name: str) -> dict[str, Any]:
    path = RunPaths(artifact_dir).reports / name
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {name}")
    return json.loads(path.read_text(encoding="utf-8"))


def read_report_text(artifact_dir: Path, name: str) -> str:
    path = RunPaths(artifact_dir).reports / name
    if not path.is_file():
        raise FileNotFoundError(f"report not found: {name}")
    return path.read_text(encoding="utf-8")


def list_runs(case_root_path: Path) -> list[dict[str, Any]]:
    """List every run dir under ``runs/`` with its model + derived status.

    Always scans the filesystem rather than trusting ``runs_index.json`` (which
    is written read-modify-write at run end and can lose an entry when two
    same-case runs finish concurrently).
    """
    runs: list[dict[str, Any]] = []
    runs_dir = case_root_path / "runs"
    if not runs_dir.is_dir():
        return runs
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        manifest = load_manifest(child)
        status: dict[str, Any] = {}
        status_path = child / "status.json"
        if not status_path.is_file():
            status_path = RunPaths(child).derived / "status.json"
        if status_path.is_file():
            try:
                status = json.loads(status_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                status = {}
        runs.append({
            "run_id": child.name,
            "artifact_dir": str(child),
            "model": manifest.get("model"),
            "status": _run_status(child, status),
            "started_at": manifest.get("started_at"),
            "ended_at": manifest.get("ended_at"),
        })
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return runs
