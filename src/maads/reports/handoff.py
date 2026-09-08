"""Portable Standard handoff bundle (zip) for external data scientists."""
from __future__ import annotations

import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from maads.artifact_paths import RunPaths
from maads.state import SUBSTEP_NAMES, CrispDMState
from maads.success_criterion import criterion_direction

from maads.reports.case_report import build_case_report, render_case_report_md
from maads.reports.execution_analysis import render_execution_analysis_md
from maads.reports.md_paths import handoff_remap_factory, run_meta_md
from maads.reports.workbook import (
    _canonical_pipeline_code,
    _canonical_pipeline_markdown,
    _cell_code,
    _cell_md,
    _conclusions_by_substep,
    _findings_markdown,
    _title_markdown,
    build_workbook_context,
)

HANDOFF_ZIP_NAME = "handoff_standard.zip"
PROFILE_STANDARD = "standard"

_REQUIREMENTS = """pandas>=2.1
pyarrow>=15.0
scikit-learn>=1.4
matplotlib>=3.8
joblib>=1.3
jupyter>=1.0
"""


def _file_name(path: str | None) -> str:
    return Path(path).name if path else ""


def build_case_config_meta(state: CrispDMState) -> dict[str, Any]:
    cfg = state.config
    sc = cfg.success_criterion
    return {
        "case_id": cfg.case_id,
        "run_id": None,
        "problem_statement": cfg.problem_statement.strip(),
        "problem_type": cfg.problem_type,
        "target_column": cfg.target_column,
        "id_column": cfg.id_column,
        "evaluation_metric": cfg.evaluation_metric,
        "success_criterion": {
            "metric": sc.metric,
            "threshold": sc.threshold,
            "direction": criterion_direction(sc.metric, sc.direction),
        },
        "feature_hints": dict(cfg.feature_hints),
        "class_labels": dict(cfg.class_labels),
        "data_files": {
            "train_csv": _file_name(cfg.data.train_csv),
            "test_csv": _file_name(cfg.data.test_csv),
            "sample_submission_csv": _file_name(cfg.data.sample_submission_csv),
        },
        "data_mining_goals": state.bu.data_mining_goals,
    }


def _run_model(paths: RunPaths) -> str | None:
    """The LLM model this run was launched with (from the run manifest)."""
    try:
        return json.loads(paths.manifest.read_text(encoding="utf-8")).get("model")
    except Exception:
        return None


def build_run_summary(
    analysis: dict[str, Any], *, run_id: str, model: str | None = None,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "case_id": analysis.get("case_id"),
        "llm_model": model or "default (.env)",
        "workflow_complete": analysis.get("workflow_complete"),
        "ml_success": analysis.get("ml_success"),
        "halt_reason": analysis.get("halt_reason"),
        "success_criterion": analysis.get("success_criterion"),
        "chosen_model": analysis.get("chosen_model"),
        "decision": analysis.get("decision"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def render_handoff_readme(
    state: CrispDMState, *, run_id: str, model: str | None = None,
) -> str:
    cfg = state.config
    return f"""# {cfg.case_id} — MAADS Standard Handoff

| | |
|---|---|
| **Dataset** | `{cfg.case_id}` |
| **LLM model** | `{model or "default (.env)"}` |
| **Run ID** | `{run_id}` |

Portable bundle for continuing this case **without** the MAADS repository.

## You need

- Python 3.10+
- `pip install -r requirements.txt`
- Jupyter Lab or VS Code

## You get

- Raw CSVs and modeling-ready `train.parquet` / `test.parquet`
- Reference submission, figures, and human-readable reports
- A notebook with a **runnable baseline** plus agent code in the appendix

## Quick start

```bash
unzip {cfg.case_id}_handoff.zip
cd {cfg.case_id}_handoff
pip install -r requirements.txt
jupyter lab notebook/case_workbook.ipynb
```

Run the cells in **Part A — Start here** first.

## Layout

- `data/` — source CSV files
- `artifacts/` — prepared parquets, submission, figures
- `meta/` — case config and run summary JSON
- `reports/` — markdown reports from the automated run
- `notebook/case_workbook.ipynb` — continue from here

## You do not need

- The MAADS repo, `pyproject.toml`, or API keys
- Re-running the full agent pipeline (appendix is reference only)
"""


def _bundle_setup_cell_source() -> str:
    return '''"""Bundle paths — run from the handoff root (directory with README.md)."""
from pathlib import Path
import json

BUNDLE_DIR = Path.cwd()
if not (BUNDLE_DIR / "README.md").is_file() and (BUNDLE_DIR.parent / "README.md").is_file():
    BUNDLE_DIR = BUNDLE_DIR.parent

DATA_DIR = BUNDLE_DIR / "data"
ARTIFACT_DIR = BUNDLE_DIR / "artifacts"
META_DIR = BUNDLE_DIR / "meta"
FIGURES_DIR = ARTIFACT_DIR / "figures"
NOTEBOOK_OUT = ARTIFACT_DIR / "notebook_outputs"
NOTEBOOK_OUT.mkdir(parents=True, exist_ok=True)

_meta = json.loads((META_DIR / "case_config.json").read_text(encoding="utf-8"))
_files = _meta["data_files"]
CASE_ID = _meta["case_id"]
PROBLEM_TYPE = _meta["problem_type"]
TARGET = _meta["target_column"]
ID_COL = _meta["id_column"]
EVAL_METRIC = _meta["evaluation_metric"]
FEATURE_HINTS = _meta.get("feature_hints") or {}
CLASS_LABELS = _meta.get("class_labels") or {}
SUCCESS_METRIC = _meta["success_criterion"]["metric"]
SUCCESS_THRESHOLD = _meta["success_criterion"]["threshold"]
SUCCESS_DIRECTION = _meta["success_criterion"]["direction"]

DATA_TRAIN_CSV = DATA_DIR / _files["train_csv"]
DATA_TEST_CSV = DATA_DIR / _files["test_csv"]
SAMPLE_SUBMISSION = DATA_DIR / _files["sample_submission_csv"]
TRAIN_PARQUET = ARTIFACT_DIR / "train.parquet"
TEST_PARQUET = ARTIFACT_DIR / "test.parquet"

print(f"Case: {CASE_ID} ({PROBLEM_TYPE})")
print(f"Bundle: {BUNDLE_DIR}")
'''


def _bundle_eda_cell_source() -> str:
    return '''"""Optional: quick look at raw training data."""
import pandas as pd

df = pd.read_csv(DATA_TRAIN_CSV)
print(f"Shape: {df.shape}")
if TARGET in df.columns:
  print(df[TARGET].describe())
df.head()
'''


def _reference_appendix_md(
    context: dict[str, Any],
    conclusions: dict[str, Any],
) -> str:
    scripts = context.get("scripts") or {}
    by_substep = _conclusions_by_substep(conclusions)
    lines = [
        "## Appendix — agent-generated code (reference only)",
        "",
        "These scripts were executed during the automated MAADS run. "
        "They are **not** required to use the bundle. "
        "Re-running them needs the full `artifacts/prep/` tree from a Full handoff.",
        "",
    ]
    for sub in sorted(scripts, key=lambda s: tuple(int(p) for p in s.split("."))):
        item = by_substep.get(sub, {})
        title = item.get("name") or SUBSTEP_NAMES.get(sub, sub)
        summary = item.get("summary", "")
        lines.append(f"### {sub} — {title}")
        if summary:
            lines.append("")
            lines.append(summary)
        lines.append("")
        lines.append("```python")
        lines.append(scripts[sub].rstrip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def render_bundle_workbook_ipynb(
    context: dict[str, Any],
    state: CrispDMState,
    paths: RunPaths,
) -> dict[str, Any]:
    """Notebook optimized for the Standard handoff zip (no repo dependency)."""
    analysis = context["analysis"]
    conclusions = context["conclusions"]

    cells: list[dict[str, Any]] = [
        _cell_md(
            _title_markdown(state, paths, analysis)
            + "\n\n## Part A — Start here\n\n"
            "Runnable cells below work from this bundle alone.\n",
        ),
        _cell_code(_bundle_setup_cell_source()),
        _cell_md("### Explore raw training data (optional)\n"),
        _cell_code(_bundle_eda_cell_source()),
        _cell_md(_canonical_pipeline_markdown(state, from_agent_script="4.3" in (context.get("scripts") or {}))),
        _cell_code(_canonical_pipeline_code(state, context.get("scripts") or {})),
        _cell_md("## Part B — Findings\n\n" + _findings_markdown(state, conclusions)),
        _cell_md(_reference_appendix_md(context, conclusions)),
        _cell_md(
            "## Part D — Your experiments\n\n"
            "_Add notes, hypotheses, and next experiments below._\n",
        ),
    ]

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
            "maads": {
                "handoff_profile": PROFILE_STANDARD,
                "case_id": context.get("case_id"),
                "run_id": context.get("run_id"),
                "generated_at": context.get("generated_at"),
            },
        },
        "cells": cells,
    }


def _zip_add_file(zf: zipfile.ZipFile, source: Path, arcname: str) -> bool:
    if not source.is_file():
        return False
    zf.write(source, arcname)
    return True


def _zip_add_tree(zf: zipfile.ZipFile, source_dir: Path, arc_prefix: str) -> int:
    if not source_dir.is_dir():
        return 0
    count = 0
    for path in sorted(source_dir.rglob("*")):
        if path.is_file():
            rel = path.relative_to(source_dir).as_posix()
            zf.write(path, f"{arc_prefix}/{rel}")
            count += 1
    return count


def build_handoff_zip(
    state: CrispDMState,
    paths: RunPaths,
    *,
    analysis: dict[str, Any] | None = None,
    profile: str = PROFILE_STANDARD,
) -> bytes:
    """Build Standard handoff zip bytes."""
    if profile != PROFILE_STANDARD:
        raise ValueError(f"Unsupported handoff profile: {profile}")

    context = build_workbook_context(state, paths, analysis=analysis)
    analysis = context["analysis"]
    cfg = state.config
    run_id = paths.run_dir.name
    model = _run_model(paths)
    root = f"{cfg.case_id}_handoff"

    meta_md = run_meta_md(cfg.case_id, model, run_id)
    case_meta = build_case_config_meta(state)
    case_meta["run_id"] = run_id
    case_meta["llm_model"] = model or "default (.env)"
    run_summary = build_run_summary(analysis, run_id=run_id, model=model)
    readme = render_handoff_readme(state, run_id=run_id, model=model)
    notebook = render_bundle_workbook_ipynb(context, state, paths)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{root}/README.md", readme)
        zf.writestr(f"{root}/requirements.txt", _REQUIREMENTS.strip() + "\n")
        zf.writestr(
            f"{root}/meta/case_config.json",
            json.dumps(case_meta, indent=2, default=str),
        )
        zf.writestr(
            f"{root}/meta/run_summary.json",
            json.dumps(run_summary, indent=2, default=str),
        )
        zf.writestr(
            f"{root}/notebook/case_workbook.ipynb",
            json.dumps(notebook, indent=2),
        )

        data_pairs = []
        for path in (cfg.data.train_csv, cfg.data.test_csv, cfg.data.sample_submission_csv):
            if path:
                data_pairs.append((Path(path), f"{root}/data/{Path(path).name}"))
        for src in cfg.data.sources:
            if src.path and src.path not in {cfg.data.train_csv, cfg.data.test_csv, cfg.data.sample_submission_csv}:
                data_pairs.append((Path(src.path), f"{root}/data/{Path(src.path).name}"))
        for src, arc in data_pairs:
            _zip_add_file(zf, src, arc)

        artifact_sources = [
            paths.run_dir / "train.parquet",
            paths.run_dir / "test.parquet",
        ]
        if state.dep.submission_path:
            artifact_sources.append(Path(state.dep.submission_path))
        else:
            artifact_sources.append(paths.run_dir / "submission.csv")
        seen_arcs: set[str] = set()
        for src in artifact_sources:
            arc = f"{root}/artifacts/{src.name}"
            if arc not in seen_arcs:
                _zip_add_file(zf, src, arc)
                seen_arcs.add(arc)

        _zip_add_tree(zf, paths.run_dir / "figures", f"{root}/artifacts/figures")

        handoff_root = Path(root)
        handoff_reports = handoff_root / "reports"
        remap = handoff_remap_factory(paths.run_dir, handoff_root)

        case_report = build_case_report(state)
        zf.writestr(
            f"{root}/reports/case_report.md",
            meta_md
            + render_case_report_md(
                case_report,
                md_dir=handoff_reports,
                run_dir=paths.run_dir,
                remap=remap,
            ),
        )
        zf.writestr(
            f"{root}/reports/execution_analysis.md",
            meta_md
            + render_execution_analysis_md(
                analysis,
                md_dir=handoff_reports,
                run_dir=paths.run_dir,
                remap=remap,
            ),
        )
        from maads.reports.final_report import build_story_spec_from_bundle, render_final_report_md

        story_spec: dict[str, Any]
        if state.dep.story_spec_path and Path(state.dep.story_spec_path).is_file():
            story_spec = json.loads(
                Path(state.dep.story_spec_path).read_text(encoding="utf-8"),
            )
        else:
            story_spec = build_story_spec_from_bundle(state)
        zf.writestr(
            f"{root}/reports/final_report.md",
            meta_md
            + render_final_report_md(
                state,
                story_spec,
                md_dir=handoff_reports,
                run_dir=paths.run_dir,
                remap=remap,
            ),
        )

    return buf.getvalue()


def write_handoff_bundle(
    state: CrispDMState,
    paths: RunPaths,
    *,
    analysis: dict[str, Any] | None = None,
    profile: str = PROFILE_STANDARD,
) -> Path:
    """Write ``reports/handoff_standard.zip`` for the run."""
    paths.reports.mkdir(parents=True, exist_ok=True)
    out_path = paths.reports / HANDOFF_ZIP_NAME
    payload = build_handoff_zip(state, paths, analysis=analysis, profile=profile)
    out_path.write_bytes(payload)
    return out_path


def handoff_zip_path(paths: RunPaths) -> Path:
    return paths.reports / HANDOFF_ZIP_NAME
