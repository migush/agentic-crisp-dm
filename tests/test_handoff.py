"""Tests for Standard handoff zip bundle."""
from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from maads.artifact_paths import RunPaths, ensure_run_layout
from maads.config import load_case_config
from maads.paths import resolve_path
from maads.reports.handoff import (
    HANDOFF_ZIP_NAME,
    build_handoff_zip,
    render_bundle_workbook_ipynb,
    write_handoff_bundle,
)
from maads.reports.workbook import build_workbook_context
from maads.reports.writer import write_run_reports
from maads.state import CrispDMState, ModelRun, Phase


def _minimal_state(cfg_path: str, run_dir: Path) -> CrispDMState:
    cfg = load_case_config(resolve_path(cfg_path))
    state = CrispDMState.from_config(cfg)
    state.halted = True
    state.phase = Phase.DEPLOYMENT
    state.substep = "6.4"
    state.halt_reason = "completed phase 6"
    state.md.chosen_model = ModelRun(
        technique="gradient_boosting",
        cv_score=0.9,
        cv_std=0.01,
        assessment="selected",
    )
    metric = cfg.success_criterion.metric
    state.ev.assessment_of_dm_results = {
        "metric": metric,
        "achieved_score": 0.9,
        "threshold": cfg.success_criterion.threshold,
        "success_criterion_met": True,
    }
    state.ev.decision = "deploy"
    sub = run_dir / "submission.csv"
    sub.write_text("id,pred\n1,0\n", encoding="utf-8")
    state.dep.submission_path = str(sub)
    state.dep.final_report_path = str(run_dir / "final_report.md")
    (run_dir / "final_report.md").write_text("# report\n", encoding="utf-8")
    (run_dir / "train.parquet").write_bytes(b"")
    (run_dir / "test.parquet").write_bytes(b"")
    (run_dir / "reports" / "case_report.md").parent.mkdir(parents=True, exist_ok=True)
    (run_dir / "reports" / "case_report.md").write_text("# case\n", encoding="utf-8")
    return state


def _zip_names(payload: bytes) -> set[str]:
    with zipfile.ZipFile(BytesIO(payload)) as zf:
        return set(zf.namelist())


@pytest.mark.parametrize("config_path", ["configs/house_prices.yaml", "configs/titanic.yaml"])
def test_handoff_zip_structure(tmp_path: Path, config_path: str):
    cfg = load_case_config(resolve_path(config_path))
    run_dir = tmp_path / "runs" / f"handoff-{cfg.case_id}"
    ensure_run_layout(run_dir, run_id=run_dir.name, case_id=cfg.case_id)
    state = _minimal_state(config_path, run_dir)
    paths = RunPaths(run_dir)

    payload = build_handoff_zip(state, paths)
    root = f"{cfg.case_id}_handoff"
    names = _zip_names(payload)

    assert f"{root}/README.md" in names
    assert f"{root}/requirements.txt" in names
    assert f"{root}/meta/case_config.json" in names
    assert f"{root}/meta/run_summary.json" in names
    assert f"{root}/notebook/case_workbook.ipynb" in names
    assert f"{root}/data/{Path(cfg.data.train_csv).name}" in names
    assert f"{root}/data/{Path(cfg.data.test_csv).name}" in names
    assert f"{root}/artifacts/train.parquet" in names
    assert f"{root}/artifacts/submission.csv" in names
    assert f"{root}/reports/case_report.md" in names


def test_bundle_notebook_uses_winning_43_script(tmp_path: Path):
    cfg = load_case_config(resolve_path("configs/house_prices.yaml"))
    run_dir = tmp_path / "runs" / "bundle-43"
    ensure_run_layout(run_dir, run_id="bundle-43", case_id=cfg.case_id)
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    paths = RunPaths(run_dir)
    marker = "# MAADS_HANDOFF_43_MARKER"
    sandbox_dir = run_dir / "sandbox" / "exec"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    script_name = "00001_data_scientist_attempt1.py"
    (sandbox_dir / script_name).write_text(
        "# --- injected by maads; do not redefine ---\n"
        "import json\nfrom pathlib import Path\nimport pandas as pd\n"
        f"{marker}\nprint(json.dumps({{'technique': 'xgboost'}}))\n",
        encoding="utf-8",
    )
    manifest = sandbox_dir / "manifest.jsonl"
    manifest.write_text(
        json.dumps({
            "seq": 1,
            "label": "data_scientist_attempt1",
            "substep": "4.3",
            "script": script_name,
            "ok": True,
            "return_code": 0,
        }) + "\n",
        encoding="utf-8",
    )
    context = build_workbook_context(state, paths)
    notebook = render_bundle_workbook_ipynb(context, state, paths)
    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert marker in sources
    assert "successful 4.3 Build Model" in sources


def test_bundle_notebook_has_no_repo_dependency(tmp_path: Path):
    cfg = load_case_config(resolve_path("configs/house_prices.yaml"))
    run_dir = tmp_path / "runs" / "bundle-nb"
    ensure_run_layout(run_dir, run_id="bundle-nb", case_id=cfg.case_id)
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    paths = RunPaths(run_dir)
    context = build_workbook_context(state, paths)
    notebook = render_bundle_workbook_ipynb(context, state, paths)

    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert "BUNDLE_DIR" in sources
    assert "REPO_ROOT" not in sources
    assert "pyproject.toml" not in sources
    assert notebook["metadata"]["maads"]["handoff_profile"] == "standard"


def test_bundle_notebook_tfidf_for_nlp_classification(tmp_path: Path):
    run_dir = tmp_path / "runs" / "bundle-nlp"
    ensure_run_layout(run_dir, run_id="bundle-nlp", case_id="titanic")
    state = _minimal_state("configs/titanic.yaml", run_dir)
    state.config.problem_type = "classification"
    state.config.feature_hints = {
        "text_free": ["text"],
        "representation_options": ["tfidf_logreg"],
    }
    state.md.chosen_model.technique = "tfidf_logreg"
    paths = RunPaths(run_dir)
    context = build_workbook_context(state, paths)
    notebook = render_bundle_workbook_ipynb(context, state, paths)
    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert "TfidfVectorizer" in sources
    assert "TECHNIQUE = 'tfidf_logreg'" in sources
    assert "SAMPLE_SUBMISSION.is_file()" in sources
    assert "_read_csv_any" in sources


def test_write_handoff_bundle_writes_zip(tmp_path: Path):
    run_dir = tmp_path / "runs" / "write-handoff"
    ensure_run_layout(run_dir, run_id="write-handoff", case_id="house_prices")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    out = write_handoff_bundle(state, RunPaths(run_dir))
    assert out.name == HANDOFF_ZIP_NAME
    assert out.is_file()
    assert zipfile.is_zipfile(out)


def test_write_run_reports_emits_handoff(tmp_path: Path):
    run_dir = tmp_path / "runs" / "handoff-integration"
    ensure_run_layout(run_dir, run_id="handoff-integration", case_id="house_prices")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    write_run_reports(state, run_dir, force=True)
    assert (run_dir / "reports" / HANDOFF_ZIP_NAME).is_file()


def test_api_serves_handoff_zip(tmp_path: Path):
    from fastapi.testclient import TestClient

    from maads.dashboard.server import create_app

    case = tmp_path / "house_prices"
    run_dir = case / "runs" / "handoff-run"
    ensure_run_layout(run_dir, run_id="handoff-run", case_id="house_prices")
    (run_dir / "status.json").write_text(
        json.dumps({"case_id": "house_prices", "phase": 6, "halted": True}),
        encoding="utf-8",
    )
    (case / "current").write_text("handoff-run", encoding="utf-8")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    write_handoff_bundle(state, RunPaths(run_dir))

    import maads.dashboard.server as server_mod

    server_mod._artifact_root = tmp_path
    client = TestClient(create_app())
    resp = client.get("/api/cases/house_prices/reports/handoff_standard.zip")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/zip"
    assert zipfile.is_zipfile(BytesIO(resp.content))

    legacy = client.get("/api/cases/house_prices/handoff.zip")
    assert legacy.status_code == 200
