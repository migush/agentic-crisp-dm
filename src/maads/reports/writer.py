"""Write all post-run reports and seal the manifest."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from maads.artifact_config import reports_enabled
from maads.artifact_paths import RunPaths, seal_manifest, update_runs_index
from maads.observability.llm_communications import (
    LLMCommunicationRecord,
    build_communications_summary,
)
from maads.observability.schema import TraceRun
from maads.outcome import ml_run_succeeded, workflow_complete
from maads.reports.case_report import build_case_report, render_case_report_md
from maads.reports.execution_analysis import (
    build_execution_analysis,
    render_execution_analysis_md,
)
from maads.reports.final_report import build_story_spec_from_bundle, write_final_report
from maads.reports.improvement_bundle import build_improvement_bundle
from maads.reports.postmortem import build_postmortem
from maads.reports.handoff import write_handoff_bundle
from maads.reports.workbook import write_case_workbook
from maads.state import CrispDMState


def write_run_reports(
    state: CrispDMState,
    artifact_dir: Path,
    *,
    trace: TraceRun | None = None,
    communications: list[LLMCommunicationRecord] | None = None,
    case_root: Path | None = None,
    force: bool = False,
) -> None:
    if not reports_enabled():
        return
    paths = RunPaths(artifact_dir)
    paths.reports.mkdir(parents=True, exist_ok=True)
    stamp_path = paths.reports / ".generated"
    if stamp_path.is_file() and not force:
        return

    from maads.reports.md_paths import run_meta_md

    try:
        _model = json.loads(paths.manifest.read_text(encoding="utf-8")).get("model")
    except Exception:
        _model = None
    meta_md = run_meta_md(state.config.case_id, _model, paths.run_dir.name)

    comm_summary = build_communications_summary(communications or [])
    postmortem = build_postmortem(state, paths, trace=trace, comm_summary=comm_summary)
    (paths.reports / "postmortem.json").write_text(
        json.dumps(postmortem, indent=2, default=str), encoding="utf-8",
    )

    case_report = build_case_report(state)
    (paths.reports / "case_report.json").write_text(
        json.dumps(case_report, indent=2, default=str), encoding="utf-8",
    )
    (paths.reports / "case_report.md").write_text(
        meta_md
        + render_case_report_md(
            case_report,
            md_dir=paths.reports,
            run_dir=paths.run_dir,
        ),
        encoding="utf-8",
    )

    analysis = build_execution_analysis(
        state, paths, trace=trace, comm_summary=comm_summary,
    )
    (paths.reports / "execution_analysis.json").write_text(
        json.dumps(analysis, indent=2, default=str), encoding="utf-8",
    )
    (paths.reports / "execution_analysis.md").write_text(
        meta_md
        + render_execution_analysis_md(
            analysis,
            md_dir=paths.reports,
            run_dir=paths.run_dir,
        ),
        encoding="utf-8",
    )

    write_case_workbook(state, paths, analysis=analysis)

    if state.dep.story_spec_path and Path(state.dep.story_spec_path).is_file():
        story_spec = json.loads(Path(state.dep.story_spec_path).read_text(encoding="utf-8"))
    elif state.md.chosen_model:
        story_spec = build_story_spec_from_bundle(state)
    else:
        story_spec = None
    if story_spec is not None:
        report_path = write_final_report(state, story_spec, paths.run_dir)
        try:
            report_path.write_text(
                meta_md + report_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
        except Exception:
            pass
        state.dep.final_report_path = str(report_path)

    write_handoff_bundle(state, paths, analysis=analysis)

    bundle = build_improvement_bundle(
        state, paths, communications or [], case_root=case_root,
    )
    (paths.reports / "improvement_bundle.json").write_text(
        json.dumps(bundle, indent=2, default=str), encoding="utf-8",
    )

    ended_at = datetime.now(timezone.utc).isoformat()
    duration_ms = analysis.get("duration_ms")
    if duration_ms is not None:
        duration_ms = int(duration_ms)
    seal_manifest(
        paths,
        ended_at=ended_at,
        workflow_complete=workflow_complete(state),
        ml_success=ml_run_succeeded(state),
        halt_reason=state.halt_reason,
        duration_ms=duration_ms,
    )
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    manifest["reports_generated_at"] = ended_at
    paths.manifest.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    stamp_path.write_text(ended_at, encoding="utf-8")

    if case_root is not None:
        update_runs_index(case_root, run_id=paths.run_dir.name, manifest=manifest)
