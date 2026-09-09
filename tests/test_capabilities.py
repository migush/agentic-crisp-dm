"""Tests for capability modules (CRISP-DM-independent APIs)."""
from __future__ import annotations

from pathlib import Path

import json

import pandas as pd
import pytest

import maads.crew as crew
from maads.capabilities.data_engineer import execution_evidence
from maads.capabilities.data_scientist import (
    _train_schema_context,
    _text_modeling_hint,
    apply_response as ds_apply_response,
    execution_evidence as ds_execution_evidence,
)
from maads.state import CrispDMState
from maads.capabilities.shared import (
    abspath,
    de_dataset_context,
    has_keys,
    prep_inputs,
    prep_workdir,
    target_preserved,
)
from maads.config import load_case_config
from maads.paths import resolve_path
from maads.state import CrispDMState
from maads.tools import PythonExec, inspect_dataset
from tests.fixtures.titanic_exec import fake_run_text_task


@pytest.fixture
def state() -> CrispDMState:
    return CrispDMState.from_config(load_case_config(resolve_path("configs/titanic.yaml")))


def test_has_keys_contract():
    assert has_keys({"a": 1}, "a") == []
    assert has_keys({}, "a") == ["missing key 'a'"]


def test_inspect_dataset_train_only_with_sources(tmp_path):
    from maads.config import CaseConfig, DataPaths, DataSource, SuccessCriterion

    train = tmp_path / "obs.csv"
    train.write_text("id,note,label\n1,hello,a\n2,world,b\n", encoding="utf-8")
    extra = tmp_path / "notes.md"
    extra.write_text("problem notes", encoding="utf-8")
    out = inspect_dataset(train, None)
    assert out["train_rows"] == 2
    assert out["train_columns"] == ["id", "note", "label"]

    cfg = CaseConfig(
        case_id="obs_case",
        problem_statement="Predict label from a short note.",
        problem_type="classification",
        data=DataPaths(
            train_csv=str(train),
            sources=[
                DataSource(path=str(train), original_filename="obs.csv"),
                DataSource(path=str(extra), original_filename="notes.md"),
            ],
        ),
        success_criterion=SuccessCriterion(metric="accuracy", threshold=0.0, direction="maximize"),
    )
    ctx = de_dataset_context(CrispDMState.from_config(cfg), str(train), "")
    summary = json.loads(ctx["DATASET_INSPECT_JSON"])
    assert summary["train_rows"] == 2
    assert len(summary["source_paths"]) == 2


def test_execution_evidence_collect_is_deterministic(monkeypatch, state, tmp_path):
    """2.1 uses profile/collect tools — no authored-code LLM call."""
    calls: list[str] = []

    def track(*a, **k):
        calls.append("run_text_task")
        return ""

    monkeypatch.setattr(crew, "run_text_task", track)
    state.substep = "2.1"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = execution_evidence(pyexec, state, "2.1", tmp_path)
    assert "initial_data_collection_report" in out
    assert out["initial_data_collection_report"]["train_rows"] > 0
    assert calls == []


def test_execution_evidence_collect_with_stub_code(monkeypatch, state, tmp_path):
    monkeypatch.setattr(crew, "run_text_task", fake_run_text_task)
    state.substep = "2.1"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = execution_evidence(pyexec, state, "2.1", tmp_path)
    assert "initial_data_collection_report" in out


def test_ds_23_adopts_blank_target_from_exploration(state):
    state.config = state.config.model_copy(update={"target_column": ""})
    delta = ds_apply_response(
        {},
        state,
        "2.3",
        {"data_exploration_report": {"n_rows": 10, "target": "Sentiment"}},
    )
    assert not delta.failed
    assert "config.target_column" in delta.fields_written
    assert state.config.target_column == "Sentiment"
    assert state.resolved_target() == "Sentiment"


def test_ds_execution_evidence_explore_with_stub_code(monkeypatch, state, tmp_path):
    monkeypatch.setattr(crew, "run_text_task", fake_run_text_task)
    state.substep = "2.3"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = ds_execution_evidence(pyexec, state, "2.3", tmp_path)
    assert "data_exploration_report" in out


def test_train_schema_context_notes_absent_id(tmp_path):
    train_path = tmp_path / "train.parquet"
    pd.DataFrame({"target": [0, 1], "text": ["a", "b"]}).to_parquet(train_path)
    cols, note = _train_schema_context(str(train_path), "id")
    assert cols == ["target", "text"]
    assert "absent" in note
    assert "if c in train.columns" in note


def test_text_modeling_hint_for_text_cases(state):
    state.config.feature_hints = {"text_free": ["text"]}
    hint = _text_modeling_hint(state)
    assert "tfidf_logreg" in hint
    assert "ml_tools" in hint


def test_text_modeling_hint_skipped_for_titanic_mixed_tabular(state):
    assert _text_modeling_hint(state) == ""


def test_baseline_techniques_titanic_is_tabular():
    from maads.capabilities.ml_tools import baseline_techniques_for

    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    techniques = baseline_techniques_for(
        problem_type=cfg.problem_type,
        feature_hints=cfg.feature_hints,
    )
    assert techniques[0] == "logistic_regression"
    assert "tfidf_logreg" not in techniques


def test_baseline_techniques_disaster_tweets_is_tfidf():
    from maads.capabilities.ml_tools import baseline_techniques_for

    cfg = load_case_config(resolve_path("configs/disaster_tweets.yaml"))
    techniques = baseline_techniques_for(
        problem_type=cfg.problem_type,
        feature_hints=cfg.feature_hints,
    )
    assert techniques[0] == "tfidf_logreg"


def test_quality_report_documents_high_missing():
    from maads.capabilities.ml_tools import quality_report_from_profile

    profile = {
        "n_rows": 891,
        "columns": ["Survived", "Cabin", "Age"],
        "missing": {"Cabin": 687, "Age": 177},
        "constant_columns": [],
        "duplicate_id_count": 0,
        "target": {"name": "Survived", "missing": 0},
    }
    report = quality_report_from_profile(
        profile,
        target="Survived",
        na_means_absent=["Cabin"],
        high_missing=["Cabin"],
    )
    assert not any("Cabin" in b for b in report["blockers"])
    assert any("Cabin" in t for t in report["tolerable"])
    assert any("Age" in t for t in report["tolerable"])


def test_select_best_model_minimizes_rmse():
    from maads.capabilities.ml_tools import select_best_model
    from maads.state import ModelRun

    models = [
        ModelRun(technique="a", cv_score=0.40, description="worse"),
        ModelRun(technique="b", cv_score=0.12, description="better"),
        ModelRun(technique="c", cv_score=0.25, description="mid"),
    ]
    best = select_best_model(models, metric="rmse", direction="minimize")
    assert best.technique == "b"


def test_select_best_model_maximizes_auc():
    from maads.capabilities.ml_tools import select_best_model
    from maads.state import ModelRun

    models = [
        ModelRun(technique="a", cv_score=0.70, description=""),
        ModelRun(technique="b", cv_score=0.85, description=""),
    ]
    best = select_best_model(models, metric="roc_auc")
    assert best.technique == "b"


def test_schema_columns_prefers_prepared_train(state):
    from maads.debug import _schema_columns

    state.dp.merged_data = {"columns_train": ["target", "text", "keyword"]}
    state.du.data_description_report = {"columns": ["id", "target", "text", "keyword", "location"]}
    assert _schema_columns(state) == ["target", "text", "keyword"]


def _tiny_text_train(path: Path) -> None:
    texts = [f"event {i} flood fire" if i % 2 else f"ok {i} sunny day" for i in range(40)]
    pd.DataFrame({"target": [i % 2 for i in range(40)], "text": texts}).to_parquet(path)


def test_text_model_baseline_on_prepared_train_without_id(tmp_path):
    from maads.capabilities.data_scientist import _run_text_model_baseline
    from maads.tools import PythonExec

    train_path = tmp_path / "train.parquet"
    _tiny_text_train(train_path)
    cols = list(pd.read_parquet(train_path).columns)
    header_vars = {
        "TRAIN_PARQUET": str(train_path),
        "TRAIN_COLUMNS": json.dumps(cols),
        "TARGET": "target",
        "ID_COL": "id",
        "METRIC": "f1",
        "PROBLEM_TYPE": "binary_classification",
        "PRIMARY_TEXT_COL": "text",
    }
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    payload = _run_text_model_baseline(pyexec, header_vars)
    assert payload["technique"] == "tfidf_logreg"
    assert isinstance(payload["cv_score"], float)
    assert isinstance(payload["cv_std"], float)


def test_ds_43_deterministic_text_baseline(monkeypatch, tmp_path):
    import maads.crew as crew
    from maads.config import load_case_config
    from maads.paths import resolve_path
    from maads.tools import PythonExec

    cfg = load_case_config(resolve_path("configs/disaster_tweets.yaml"))
    state = CrispDMState.from_config(cfg)
    state.substep = "4.3"
    train_path = tmp_path / "train.parquet"
    _tiny_text_train(train_path)
    state.dp.dataset = {"train": str(train_path)}
    state.md.modeling_technique = "tfidf_logreg"

    calls: list[str] = []
    monkeypatch.setattr(crew, "run_text_task", lambda *a, **k: calls.append("llm") or "")
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = ds_execution_evidence(pyexec, state, "4.3", tmp_path)
    assert "model_run" in out
    assert out["model_run"]["cv_score"] is not None
    assert out["model_run"]["technique"] == "tfidf_logreg"
    assert out["model_run"].get("artifact_path")
    assert calls == []


def test_ds_44_deterministic_text_assess(monkeypatch, tmp_path):
    import maads.crew as crew
    from maads.config import load_case_config
    from maads.paths import resolve_path
    from maads.state import ModelRun
    from maads.tools import PythonExec

    cfg = load_case_config(resolve_path("configs/disaster_tweets.yaml"))
    state = CrispDMState.from_config(cfg)
    state.substep = "4.4"
    train_path = tmp_path / "train.parquet"
    _tiny_text_train(train_path)
    state.dp.dataset = {"train": str(train_path)}
    state.md.models.append(
        ModelRun(technique="tfidf_logreg", cv_score=0.74, cv_std=0.01, description="test"),
    )

    monkeypatch.setattr(crew, "run_text_task", lambda *a, **k: "")
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = ds_execution_evidence(pyexec, state, "4.4", tmp_path)
    assert "evaluation_bundle" in out
    bundle = out["evaluation_bundle"]
    assert bundle["problem_type"] == "binary_classification"
    assert "f1" in bundle["metrics"]
    assert bundle["confusion_matrix"]


# --- prep_inputs stage routing ------------------------------------------------

def _write_stage(wd: Path, stage: str) -> None:
    frame = pd.DataFrame({"x": [1, 2]})
    frame.to_parquet(wd / f"train_{stage}.parquet")
    frame.to_parquet(wd / f"test_{stage}.parquet")


def test_prep_inputs_routes_to_correct_upstream(state, tmp_path):
    wd = prep_workdir(tmp_path)
    for stage in ("clean", "constructed", "integrated"):
        _write_stage(wd, stage)
    raw = (abspath(state.config.data.train_csv), abspath(state.config.data.test_csv))

    assert prep_inputs(tmp_path, state, "3.2") == raw  # Clean reads raw
    assert prep_inputs(tmp_path, state, "3.3") == (
        str((wd / "train_clean.parquet").resolve()),
        str((wd / "test_clean.parquet").resolve()),
    )
    assert prep_inputs(tmp_path, state, "3.4") == (
        str((wd / "train_constructed.parquet").resolve()),
        str((wd / "test_constructed.parquet").resolve()),
    )
    assert prep_inputs(tmp_path, state, "3.5") == (
        str((wd / "train_integrated.parquet").resolve()),
        str((wd / "test_integrated.parquet").resolve()),
    )


def test_prep_inputs_falls_back_to_earlier_stage(state, tmp_path):
    wd = prep_workdir(tmp_path)
    _write_stage(wd, "clean")  # only the clean stage exists
    clean = (
        str((wd / "train_clean.parquet").resolve()),
        str((wd / "test_clean.parquet").resolve()),
    )

    # 3.4 wants 'constructed', 3.5 wants 'integrated' — both walk back to 'clean'.
    assert prep_inputs(tmp_path, state, "3.4") == clean
    assert prep_inputs(tmp_path, state, "3.5") == clean


def test_prep_inputs_empty_workdir_reads_raw(state, tmp_path):
    # Locks the assumption test_de_prep_reports_measured_from_parquet_not_llm relies
    # on: with no prep artifacts, even 3.5 falls all the way back to the raw CSVs.
    prep_workdir(tmp_path)
    raw = (abspath(state.config.data.train_csv), abspath(state.config.data.test_csv))
    assert prep_inputs(tmp_path, state, "3.5") == raw
    assert prep_inputs(tmp_path, state, "3.2") == raw


# --- target_preserved contract helper ----------------------------------------

def test_target_preserved_detects_drop(tmp_path):
    kept = tmp_path / "kept.parquet"
    dropped = tmp_path / "dropped.parquet"
    pd.DataFrame({"Survived": [0, 1], "Age": [22, 38]}).to_parquet(kept)
    pd.DataFrame({"Age": [22, 38]}).to_parquet(dropped)

    assert target_preserved({"train_out": str(kept)}, "Survived") == []
    assert target_preserved({"train_out": str(dropped)}, "Survived")  # non-empty
    assert target_preserved({}, "Survived") == ["missing key 'train_out'"]


# --- end-to-end: 3.4 Integrate must preserve the target -----------------------

def test_integrate_preserves_target(monkeypatch, state, tmp_path):
    monkeypatch.setattr(crew, "run_text_task", fake_run_text_task)
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    for sub in ("3.2", "3.3", "3.4"):
        state.substep = sub
        execution_evidence(pyexec, state, sub, tmp_path)
    integrated = prep_workdir(tmp_path) / "train_integrated.parquet"
    cols = pd.read_parquet(integrated).columns
    assert state.config.target_column in cols


def test_deterministic_integrate_preserves_target_column(state, tmp_path):
    """Deterministic 3.4 must keep TARGET even when aligning to test schema."""
    from maads.capabilities.ml_tools import integrate_tables

    wd = prep_workdir(tmp_path)
    train = wd / "train_constructed.parquet"
    test = wd / "test_constructed.parquet"
    target = state.config.target_column
    pd.DataFrame({target: [0, 1], "Age": [22, 38], "Sex": ["m", "f"]}).to_parquet(train)
    pd.DataFrame({"Age": [30], "Sex": ["f"]}).to_parquet(test)
    out = integrate_tables(str(train), str(test), wd, target=target)
    cols = pd.read_parquet(out["train_out"]).columns
    assert target in cols


def test_execution_evidence_inspect_error_skips_authored_code(monkeypatch, state, tmp_path):
    """Missing train must fail closed before any deterministic profiling."""
    calls: list[str] = []

    def boom(*a, **k):
        calls.append("profile_dataset")
        raise AssertionError("profile_dataset must not be called")

    monkeypatch.setattr(
        "maads.capabilities.ml_tools.profile_dataset",
        boom,
    )
    monkeypatch.setattr(
        "maads.capabilities.ml_tools.collect_report",
        boom,
    )
    state.config = state.config.model_copy(
        update={
            "data": state.config.data.model_copy(
                update={"train_csv": str(tmp_path / "missing_train.csv"), "test_csv": None}
            )
        }
    )
    state.substep = "2.1"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    with pytest.raises(RuntimeError, match="dataset inspect failed"):
        execution_evidence(pyexec, state, "2.1", tmp_path)
    assert calls == []


def test_measure_prep_artifacts_missing_source_train(tmp_path):
    from maads.capabilities.shared import measure_prep_artifacts

    train_pq = tmp_path / "train.parquet"
    test_pq = tmp_path / "test.parquet"
    pd.DataFrame({"a": [1], "y": [0]}).to_parquet(train_pq)
    pd.DataFrame({"a": [2]}).to_parquet(test_pq)
    out = measure_prep_artifacts(
        source_train=str(tmp_path / "gone.csv"),
        source_test="",
        train_parquet=str(train_pq),
        test_parquet=str(test_pq),
        target="y",
        payload_derived=["feat"],
        payload_dropped=["id"],
    )
    assert out["merged_data"]["train_rows"] == 1
    assert "source train missing" in out["data_cleaning_report"]["source"]


def test_codegen_instruction_forbids_filesystem_search():
    from maads.codegen import _build_instruction

    text = _build_instruction(
        "do work",
        {"TRAIN_CSV": "/x", "SOURCE_PATHS": "[]"},
        "Required keys: ok",
        None,
        None,
    )
    assert "os.walk" in text
    assert "SOURCE_PATHS" in text
