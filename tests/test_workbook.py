"""Tests for case-agnostic Jupyter workbook generation."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from maads.artifact_paths import RunPaths, ensure_run_layout
from maads.config import load_case_config
from maads.paths import resolve_path, repo_root
from maads.reports.workbook import (
    build_workbook_context,
    render_workbook_ipynb,
    write_case_workbook,
)
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
        technique="test_model",
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
    return state


def _write_sandbox_script(
    run_dir: Path,
    *,
    substep: str,
    script_name: str,
    code: str,
    ok: bool = True,
    seq: int = 1,
) -> None:
    sandbox_dir = run_dir / "sandbox" / "exec"
    sandbox_dir.mkdir(parents=True, exist_ok=True)
    script_path = sandbox_dir / script_name
    header = (
        "# --- injected by maads; do not redefine ---\n"
        "import json\nfrom pathlib import Path\nimport pandas as pd\n"
        f"OUTDIR = {str(run_dir)!r}\n"
    )
    script_path.write_text(header + code, encoding="utf-8")
    manifest = sandbox_dir / "manifest.jsonl"
    row = {
        "seq": seq,
        "label": f"data_engineer_attempt{seq}",
        "substep": substep,
        "script": script_name,
        "ok": ok,
        "return_code": 0 if ok else 1,
    }
    with manifest.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    "config_path",
    [
        "configs/house_prices.yaml",
        "configs/titanic.yaml",
        "configs/disaster_tweets.yaml",
    ],
)
def test_workbook_is_case_agnostic(tmp_path: Path, config_path: str):
    cfg = load_case_config(resolve_path(config_path))
    run_dir = tmp_path / "runs" / f"run-{cfg.case_id}"
    ensure_run_layout(run_dir, run_id=run_dir.name, case_id=cfg.case_id)
    state = _minimal_state(config_path, run_dir)

    _write_sandbox_script(
        run_dir,
        substep="3.3",
        script_name="00001_data_engineer_attempt1.py",
        code='print(json.dumps({"derived": ["feat_a"]}))\n',
    )
    _write_sandbox_script(
        run_dir,
        substep="4.3",
        script_name="00002_data_scientist_attempt1.py",
        code='print(json.dumps({"technique": "test_model", "cv_score": 0.9}))\n',
        seq=2,
    )

    nb_path = write_case_workbook(state, RunPaths(run_dir))
    assert nb_path.is_file()
    ctx_path = run_dir / "reports" / "workbook_context.json"
    assert ctx_path.is_file()

    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    meta = notebook["metadata"]["maads"]
    assert meta["case_id"] == cfg.case_id
    assert meta["problem_type"] == cfg.problem_type

    sources = "".join(
        "".join(c.get("source") or [])
        for c in notebook["cells"]
    )
    assert cfg.case_id in sources
    assert cfg.target_column in sources
    assert cfg.problem_type in sources
    assert "3.3" in sources
    assert "4.3" in sources
    assert "print(json.dumps" in sources

    context = json.loads(ctx_path.read_text(encoding="utf-8"))
    assert context["scripts"]["3.3"]
    assert context["scripts"]["4.3"]


def test_write_run_reports_emits_workbook(tmp_path: Path):
    run_dir = tmp_path / "runs" / "wb-integration"
    ensure_run_layout(run_dir, run_id="wb-integration", case_id="house_prices")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    write_run_reports(state, run_dir, force=True)
    assert (run_dir / "reports" / "case_workbook.ipynb").is_file()


_AGENT_43_MARKER = "# MAADS_AGENT_43_CANONICAL_MARKER"


def test_canonical_pipeline_uses_winning_43_script(tmp_path: Path):
    run_dir = tmp_path / "runs" / "canonical-43"
    ensure_run_layout(run_dir, run_id="canonical-43", case_id="titanic")
    state = _minimal_state("configs/titanic.yaml", run_dir)
    (run_dir / "train.parquet").write_bytes(b"")
    _write_sandbox_script(
        run_dir,
        substep="4.3",
        script_name="00001_data_scientist_attempt1.py",
        code=f"{_AGENT_43_MARKER}\nprint(json.dumps({{'technique': 'lightgbm', 'cv_score': 0.9}}))\n",
    )
    nb_path = write_case_workbook(state, RunPaths(run_dir))
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert _AGENT_43_MARKER in sources
    assert "successful 4.3 Build Model" in sources
    assert "No 4.3 sandbox script was captured" not in sources


def test_canonical_pipeline_fallback_without_43(tmp_path: Path):
    run_dir = tmp_path / "runs" / "canonical-fallback"
    ensure_run_layout(run_dir, run_id="canonical-fallback", case_id="titanic")
    state = _minimal_state("configs/titanic.yaml", run_dir)
    (run_dir / "train.parquet").write_bytes(b"")
    nb_path = write_case_workbook(state, RunPaths(run_dir))
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert "model.joblib" in sources
    assert "joblib.dump" in sources
    assert "build_pipeline" in sources
    assert "No 4.3 sandbox script was captured" in sources


def test_workbook_includes_canonical_pipeline(tmp_path: Path):
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    run_dir = tmp_path / "runs" / "canonical"
    ensure_run_layout(run_dir, run_id="canonical", case_id=cfg.case_id)
    state = _minimal_state("configs/titanic.yaml", run_dir)
    (run_dir / "train.parquet").write_bytes(b"")
    nb_path = write_case_workbook(state, RunPaths(run_dir))
    notebook = json.loads(nb_path.read_text(encoding="utf-8"))
    sources = "".join("".join(c.get("source") or []) for c in notebook["cells"])
    assert "model.joblib" in sources
    assert "Canonical pipeline" in sources
    assert "joblib.dump" in sources
    assert "logistic_regression" in sources or "_ESTIMATORS" in sources


def test_api_serves_case_workbook(tmp_path: Path):
    from fastapi.testclient import TestClient
    from maads.dashboard.server import create_app

    case = tmp_path / "house_prices"
    run_dir = case / "runs" / "wb-run"
    ensure_run_layout(run_dir, run_id="wb-run", case_id="house_prices")
    (run_dir / "status.json").write_text(
        json.dumps({"case_id": "house_prices", "phase": 6, "halted": True}),
        encoding="utf-8",
    )
    (case / "current").write_text("wb-run", encoding="utf-8")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    write_case_workbook(state, RunPaths(run_dir))

    import maads.dashboard.server as server_mod

    server_mod._artifact_root = tmp_path
    client = TestClient(create_app())
    resp = client.get("/api/cases/house_prices/reports/case_workbook.ipynb")
    assert resp.status_code == 200
    assert "nbformat" in resp.text


def test_api_generates_case_workbook_from_final_state(tmp_path: Path):
    from fastapi.testclient import TestClient
    from maads.dashboard.server import create_app

    case = tmp_path / "house_prices"
    run_dir = case / "runs" / "wb-lazy"
    ensure_run_layout(run_dir, run_id="wb-lazy", case_id="house_prices")
    (run_dir / "status.json").write_text(
        json.dumps({"case_id": "house_prices", "phase": 6, "halted": True}),
        encoding="utf-8",
    )
    (case / "current").write_text("wb-lazy", encoding="utf-8")
    state = _minimal_state("configs/house_prices.yaml", run_dir)
    (run_dir / "final_state.json").write_text(state.model_dump_json(indent=2), encoding="utf-8")
    assert not (run_dir / "reports" / "case_workbook.ipynb").is_file()

    import maads.dashboard.server as server_mod

    server_mod._artifact_root = tmp_path
    client = TestClient(create_app())
    resp = client.get("/api/cases/house_prices/reports/case_workbook.ipynb")
    assert resp.status_code == 200
    assert "nbformat" in resp.text
    assert (run_dir / "reports" / "case_workbook.ipynb").is_file()


def test_path_rewrite_uses_run_dir(tmp_path: Path):
    run_dir = tmp_path / "runs" / "path-test"
    repo = repo_root()
    from maads.reports.workbook import _rewrite_paths

    raw = f"TRAIN_CSV = '{repo / 'data/titanic/train.csv'}'\n"
    raw += f"TRAIN_IN = '{run_dir / 'prep/train_clean.parquet'}'\n"
    raw += f"x = '{run_dir}'\n"
    code = _rewrite_paths(raw, run_dir, repo)
    assert "str(REPO_ROOT / 'data/titanic/train.csv')" in code
    assert "str(RUN_DIR / 'prep/train_clean.parquet')" in code
    assert "str(RUN_DIR)" in code
    assert str(run_dir) not in code
    assert str(repo) not in code


def test_strip_injection_header_removes_injected_vars(tmp_path: Path):
    from maads.reports.workbook import _strip_injection_header

    raw = (
        "# --- injected by maads; do not redefine ---\n"
        "import json\nfrom pathlib import Path\nimport pandas as pd\n"
        f"TRAIN_CSV = '{repo_root() / 'data/x.csv'}'\n"
        "TARGET = 'y'\n"
        "import pandas as pd\n"
        "df = pd.read_csv(TRAIN_CSV)\n"
    )
    cleaned = _strip_injection_header(raw)
    assert "injected by maads" not in cleaned
    assert "TRAIN_CSV =" not in cleaned
    assert "df = pd.read_csv" in cleaned


def test_path_rewrite_integration(tmp_path: Path):
    run_dir = tmp_path / "runs" / "path-test"
    ensure_run_layout(run_dir, run_id="path-test", case_id="titanic")
    state = _minimal_state("configs/titanic.yaml", run_dir)
    _write_sandbox_script(
        run_dir,
        substep="6.1",
        script_name="00001_developer_attempt1.py",
        code=f'x = {str(run_dir)!r}\nprint(json.dumps({{"ok": True}}))\n',
    )
    context = build_workbook_context(state, RunPaths(run_dir))
    code = context["scripts"]["6.1"]
    assert "str(RUN_DIR)" in code
    assert str(run_dir) not in code

    notebook = render_workbook_ipynb(context, state, RunPaths(run_dir))
    setup = "".join(notebook["cells"][1]["source"])
    assert "REPO_ROOT" in setup
    assert "PosixPath" not in setup
    assert "DATA_TRAIN_CSV = REPO_ROOT / " in setup
