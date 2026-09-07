"""FastAPI routes for the trace monitoring dashboard."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel

from maads.dashboard import aggregators, store
from maads.dashboard.deps import CaseScope, case_scope_dep, launch_guard_dep
from maads.observability.llm_communications import LLMCommunicationRecord, record_for_export
from maads.reports.debug_index import build_substep_debug_index

router = APIRouter(prefix="/api")

# Which artifacts this request may read. Local runs get the single process-wide
# root; the hosted app overrides the dependency to pin each caller to their own.
Scope = Annotated[CaseScope, Depends(case_scope_dep)]


@router.get("/health")
def health(scope: Scope) -> dict[str, Any]:
    cases = scope.list_cases()
    return {
        "ok": True,
        "artifact_root": str(scope.write_root.resolve()),
        "case_count": len(cases),
        "cases": [c["case_id"] for c in cases],
    }


@router.get("/cases")
def list_cases(scope: Scope) -> list[dict[str, Any]]:
    return scope.list_cases()


@router.get("/cases/{case_id}/runs")
def get_case_runs(case_id: str, scope: Scope) -> list[dict[str, Any]]:
    return store.list_runs(scope.case_path(case_id))


@router.get("/cases/{case_id}/results")
def get_case_results(case_id: str, scope: Scope) -> list[dict[str, Any]]:
    """One comparison row per run of a case: which LLM, which ML model, the
    data-prep approach and the outcome — for comparing models side by side."""
    from maads.outcome import ml_run_succeeded, workflow_complete
    from maads.state import CrispDMState
    from maads.success_criterion import criterion_direction, score_meets_threshold

    results: list[dict[str, Any]] = []
    for run in store.list_runs(scope.case_path(case_id)):
        artifact_dir = Path(run["artifact_dir"])
        row: dict[str, Any] = {
            "run_id": run.get("run_id"),
            "llm_model": run.get("model") or "default (.env)",
            "status": run.get("status"),
            "started_at": run.get("started_at"),
            "ended_at": run.get("ended_at"),
            "duration_ms": store.run_duration_ms(
                artifact_dir,
                started_at=run.get("started_at"),
                ended_at=run.get("ended_at"),
            ),
            "chosen_model": None,
            "chosen_params": {},
            "modeling_technique": None,
            "data_prep": None,
            "derived_features": [],
            "derived_summary": None,
            "missing_summary": None,
            "problem_type": None,
            "score": None,
            "cv_score": None,
            "holdout_score": None,
            "score_metric": None,
            "success_threshold": None,
            "meets_threshold": None,
            "metrics": {},
            "confusion_matrix": None,
            "class_labels": {},
            "ml_success": None,
            "workflow_complete": None,
            "halt_reason": None,
            "total_tokens": None,
        }
        state_path = artifact_dir / "final_state.json"
        if state_path.is_file():
            try:
                state = CrispDMState.model_validate_json(
                    state_path.read_text(encoding="utf-8", errors="replace"),
                )
            except Exception:
                state = None
            if state is not None:
                cm = state.md.chosen_model
                sc = state.config.success_criterion
                bundle = cm.evaluation_bundle if cm else None
                score = None
                if bundle and bundle.metrics:
                    raw_metric = bundle.metrics.get(sc.metric)
                    if isinstance(raw_metric, (int, float)):
                        score = float(raw_metric)
                direction = criterion_direction(sc.metric, sc.direction)

                da = state.dp.derived_attributes or {}
                da_items = da.get("items") if isinstance(da, dict) else None
                derived = (
                    [
                        it.get("field")
                        for it in da_items
                        if isinstance(it, dict) and it.get("field")
                    ]
                    if isinstance(da_items, list)
                    else []
                )
                dc = state.dp.data_cleaning_report or {}
                mb = dc.get("missing_before") if isinstance(dc, dict) else {}
                dropped = dc.get("columns_dropped") or [] if isinstance(dc, dict) else []
                mb = mb if isinstance(mb, dict) else {}
                filled = [
                    c for c, n in mb.items()
                    if isinstance(n, (int, float)) and n > 0 and c not in dropped
                ]
                dropped_missing = [
                    c for c, n in mb.items()
                    if isinstance(n, (int, float)) and n > 0 and c in dropped
                ]
                parts = []
                if filled:
                    parts.append("filled " + ", ".join(filled))
                if dropped_missing:
                    parts.append("dropped " + ", ".join(dropped_missing))
                if parts:
                    _ms = "; ".join(parts) + "."
                    missing_summary = _ms[:1].upper() + _ms[1:]
                else:
                    missing_summary = "No missing values to handle."
                derived_summary = (
                    f"Engineered {len(derived)} feature(s): " + ", ".join(derived) + "."
                    if derived
                    else "No new features engineered."
                )

                row.update(
                    {
                        "chosen_model": cm.technique if cm else None,
                        "chosen_params": dict(cm.parameter_settings) if cm else {},
                        "modeling_technique": state.md.modeling_technique,
                        "data_prep": state.dp.dataset_description,
                        "derived_features": derived,
                        "derived_summary": derived_summary,
                        "missing_summary": missing_summary,
                        "problem_type": (
                            bundle.problem_type if bundle else state.config.problem_type
                        ),
                        "metrics": dict(bundle.metrics) if bundle else {},
                        "confusion_matrix": bundle.confusion_matrix if bundle else None,
                        "class_labels": dict(bundle.class_labels) if bundle else {},
                        "score": score,
                        "cv_score": cm.cv_score if cm else None,
                        "holdout_score": cm.holdout_score if cm else None,
                        "score_metric": sc.metric,
                        "success_threshold": sc.threshold,
                        "meets_threshold": (
                            score_meets_threshold(
                                score, sc.threshold, direction=direction
                            )
                            if score is not None and sc.threshold is not None
                            else None
                        ),
                        "ml_success": ml_run_succeeded(state),
                        "workflow_complete": workflow_complete(state),
                        "halt_reason": state.halt_reason,
                        "total_tokens": sum(state.token_spend.values()) or None,
                    },
                )
        results.append(row)

    from maads.dashboard.insights import attach_badges, build_run_insight

    for row in results:
        row["insight"] = build_run_insight(row)
    attach_badges(results)

    # newest first
    results.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    return results


@router.get("/cases/{case_id}/live_summary")
def get_live_summary(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_live_summary(scope.case_dir(case_id, run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/status")
def get_status(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_status(scope.case_dir(case_id, run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/manifest")
def get_manifest(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_manifest(scope.case_dir(case_id, run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/trace/summary")
def get_trace_summary(
    case_id: str,
    scope: Scope,
    limit: int = Query(50, ge=1, le=500),
    run_id: str | None = Query(None),
) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = store.read_trace_optional(artifact_dir, case_id=case_id)
    return aggregators.trace_summary(run, tail_limit=limit)


@router.get("/cases/{case_id}/trace/events")
def get_trace_events(
    case_id: str,
    scope: Scope,
    since_id: str | None = Query(None, alias="since_id"),
    run_id: str | None = Query(None),
) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = store.read_trace_optional(artifact_dir, case_id=case_id)
    return aggregators.trace_events_since(run, since_id)


@router.get("/cases/{case_id}/communications")
def get_communications(
    case_id: str,
    scope: Scope,
    since_id: str | None = Query(None, alias="since_id"),
    limit: int | None = Query(None, ge=1, le=500),
    run_id: str | None = Query(None),
) -> list[dict[str, Any]]:
    try:
        records = store.read_communications(
            scope.case_dir(case_id, run_id),
            since_id=since_id,
            limit=limit,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return [_comm_dict(r) for r in records]


# Registered before /{comm_id}: FastAPI matches in declaration order, so with
# the reverse order the literal "summary" path is swallowed by the id route and
# the frontend's fetchCommunicationsSummary always 404s.
@router.get("/cases/{case_id}/communications/summary")
def get_communications_summary(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_communications_summary(scope.case_dir(case_id, run_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/communications/{comm_id}")
def get_communication(case_id: str, comm_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        record = store.read_communication(
            scope.case_dir(case_id, run_id), comm_id,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    if record is None:
        raise HTTPException(status_code=404, detail=f"Communication not found: {comm_id}")
    return _comm_dict(record)


@router.get("/cases/{case_id}/debug/substep/{substep}")
def get_substep_debug(case_id: str, substep: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
        run = store.read_trace_optional(artifact_dir, case_id=case_id)
        comms = store.read_communications(artifact_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    from maads.artifact_paths import RunPaths

    return build_substep_debug_index(
        substep, RunPaths(artifact_dir), trace=run, communications=comms,
    )


@router.get("/cases/{case_id}/reports/postmortem")
def get_postmortem(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_report(scope.case_dir(case_id, run_id), "postmortem.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/reports/case_report")
def get_case_report(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_report(scope.case_dir(case_id, run_id), "case_report.json")
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/reports/case_report.md", response_class=PlainTextResponse)
def get_case_report_md(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> str:
    try:
        return store.read_report_text(
            scope.case_dir(case_id, run_id), "case_report.md",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/reports/execution_analysis.md", response_class=PlainTextResponse)
def get_execution_analysis_md(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> str:
    try:
        return store.read_report_text(
            scope.case_dir(case_id, run_id), "execution_analysis.md",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/final_report.md", response_class=PlainTextResponse)
def get_final_report_md(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> str:
    from maads.artifact_paths import RunPaths

    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = RunPaths(artifact_dir).run_dir / "final_report.md"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="final_report.md not found")
    return path.read_text(encoding="utf-8")


@router.get("/cases/{case_id}/artifacts/{artifact_path:path}")
def get_run_artifact(
    case_id: str,
    scope: Scope,
    artifact_path: str,
    run_id: str | None = Query(None),
) -> FileResponse:
    """Serve a file from a run directory (e.g. figures for report rendering)."""
    import mimetypes

    from maads.artifact_paths import RunPaths
    from maads.reports.md_paths import resolve_artifact_path

    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_dir = RunPaths(artifact_dir).run_dir.resolve()
    resolved = resolve_artifact_path(artifact_path, run_dir)
    if resolved is None:
        raise HTTPException(status_code=404, detail="artifact not found")
    try:
        resolved.relative_to(run_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="artifact not found")

    media_type, _ = mimetypes.guess_type(resolved.name)
    return FileResponse(resolved, media_type=media_type or "application/octet-stream")


@router.get("/cases/{case_id}/reports/improvement_bundle")
def get_improvement_bundle(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_report(
            scope.case_dir(case_id, run_id), "improvement_bundle.json",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/reports/case_workbook.ipynb")
def get_case_workbook(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> FileResponse:
    from maads.artifact_paths import RunPaths

    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    path = RunPaths(artifact_dir).reports / "case_workbook.ipynb"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="case_workbook.ipynb not found")
    return FileResponse(
        path,
        media_type="application/x-ipynb+json",
        filename=f"{case_id}_case_workbook.ipynb",
    )


@router.get("/cases/{case_id}/reports/workbook_context.json")
def get_workbook_context(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        return store.read_report(
            scope.case_dir(case_id, run_id), "workbook_context.json",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/reports/handoff_standard.zip")
def get_handoff_standard_zip(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> FileResponse:
    return _handoff_file_response(scope, case_id, run_id)


@router.get("/cases/{case_id}/handoff.zip")
def get_handoff_bundle(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> FileResponse:
    """Legacy alias for handoff_standard.zip."""
    return _handoff_file_response(scope, case_id, run_id)


def _handoff_file_response(
    scope: CaseScope, case_id: str, run_id: str | None = None,
) -> FileResponse:
    from maads.artifact_paths import RunPaths
    from maads.reports.handoff import HANDOFF_ZIP_NAME, build_handoff_zip, handoff_zip_path
    from maads.state import CrispDMState

    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    paths = RunPaths(artifact_dir)
    zip_path = handoff_zip_path(paths)
    if not zip_path.is_file():
        if scope.is_read_only(case_id):
            raise HTTPException(status_code=404, detail="handoff bundle not available")
        state_path = artifact_dir / "final_state.json"
        if not state_path.is_file():
            raise HTTPException(status_code=404, detail="handoff bundle not available")
        state = CrispDMState.model_validate_json(state_path.read_text(encoding="utf-8"))
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        zip_path.write_bytes(build_handoff_zip(state, paths))

    # Name the download by dataset + model so files from different runs/models
    # don't collide (e.g. house_prices__gpt-4o__handoff.zip).
    model = None
    try:
        model = json.loads(paths.manifest.read_text(encoding="utf-8")).get("model")
    except Exception:
        pass
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{case_id}__{model or 'default'}").strip("-")
    download_name = f"{slug}__{HANDOFF_ZIP_NAME}"

    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=download_name,
    )


@router.get("/cases/{case_id}/graph")
def get_graph(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    run = store.read_trace_optional(artifact_dir, case_id=case_id)
    return aggregators.build_graph(run)


@router.get("/cases/{case_id}/process")
def get_process(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
        status = store.read_status(artifact_dir)
        run = store.read_trace_optional(artifact_dir, case_id=case_id)
        snapshot = store.read_process_snapshot(artifact_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return aggregators.build_process_view(
        status, run, snapshot, artifact_dir=artifact_dir,
    )


@router.get("/cases/{case_id}/state")
def get_state(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
        return store.read_state(artifact_dir)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/cases/{case_id}/rag")
def get_rag(case_id: str, scope: Scope, run_id: str | None = Query(None)) -> dict[str, Any]:
    """RAG corpus, embedder, and passages for the Domain Expert (live from state)."""
    from maads.config import load_case_config
    from maads.dashboard.rag_view import build_rag_view
    from maads.paths import resolve_path
    from maads.state import CrispDMState

    state: CrispDMState | None = None
    try:
        artifact_dir = scope.case_dir(case_id, run_id)
        raw = store.read_state(artifact_dir)
        state = CrispDMState.model_validate(raw["state"])
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception:
        config_path = resolve_path("configs") / f"{case_id}.yaml"
        if config_path.is_file():
            state = CrispDMState.from_config(load_case_config(config_path))
    return build_rag_view(state, case_id=case_id)


@router.get("/configs")
def list_configs() -> list[dict[str, Any]]:
    """Return available case configs from the configs/ directory."""
    from maads.paths import repo_root
    import yaml as _yaml

    configs_dir = repo_root() / "configs"
    result = []
    # Several config files may declare the same case_id (e.g. titanic.yaml and a
    # titanic_loopdemo.yaml). Only the file whose stem matches case_id is
    # launchable (start_run resolves {case_id}.yaml), so dedupe to one entry per
    # case_id and keep the canonical file when present.
    seen: dict[str, bool] = {}
    for yaml_path in sorted(configs_dir.glob("*.yaml")):
        try:
            raw = _yaml.safe_load(yaml_path.read_text())
            case_id = raw.get("case_id", yaml_path.stem)
            entry = {
                "case_id": case_id,
                "problem_type": raw.get("problem_type"),
                "evaluation_metric": raw.get("evaluation_metric"),
                "problem_statement": (raw.get("problem_statement") or "").strip(),
                "success_threshold": (raw.get("success_criterion") or {}).get("threshold"),
            }
        except Exception:
            case_id = yaml_path.stem
            entry = {"case_id": case_id, "problem_type": None, "evaluation_metric": None, "problem_statement": None, "success_threshold": None}
        canonical = yaml_path.stem == case_id
        if case_id in seen:
            if canonical:
                # Replace the previously kept non-canonical entry.
                result = [e for e in result if e["case_id"] != case_id]
            else:
                continue
        seen[case_id] = True
        result.append(entry)
    return result


@router.get("/models")
def list_models() -> dict[str, list[dict[str, str]]]:
    """Return the curated, selectable models grouped by provider."""
    from maads.model_catalog import model_catalog

    return model_catalog()


@router.get("/models/info")
def model_info(
    model: str | None = Query(default=None, description="Model id; omit for MODEL from .env"),
    probe: bool = Query(default=False, description="Live-probe JSON capabilities (may cost tokens)"),
) -> dict[str, Any]:
    """Return provider metadata for a selectable model."""
    from maads.model_catalog import model_label
    from maads.model_info import fetch_model_info

    payload = fetch_model_info(model or "", probe=probe)
    mid = payload.get("model") or ""
    if mid:
        label = model_label(mid)
        if label:
            payload["label"] = label
    return payload


class RunRequest(BaseModel):
    case_id: str
    model: str | None = None


@router.post("/run", dependencies=[Depends(launch_guard_dep)])
def start_run(req: RunRequest, scope: Scope) -> dict[str, Any]:
    """Launch a pipeline run for the given case in the background."""
    from maads.paths import repo_root

    root = repo_root()
    configs_dir = root / "configs"
    config_path = configs_dir / f"{req.case_id}.yaml"
    if not config_path.is_file():
        raise HTTPException(status_code=404, detail=f"Config not found: {req.case_id}")

    cmd = [
        sys.executable, "-m", "maads", "run",
        "--case", req.case_id,
        # Write into the caller's own root rather than the process default, so a
        # launch can never land in someone else's (or a read-only demo's) tree.
        "--artifact-dir", str(scope.write_root),
    ]
    model = (req.model or "").strip()
    if model:
        cmd += ["--model", model]

    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return {"status": "started", "case_id": req.case_id, "model": model or None, "pid": proc.pid}


def _comm_dict(record: LLMCommunicationRecord) -> dict[str, Any]:
    return record_for_export(record)
