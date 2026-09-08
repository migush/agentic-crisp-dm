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
from maads.tools import PythonExec
from tests.fixtures.titanic_exec import fake_run_text_task


@pytest.fixture
def state() -> CrispDMState:
    return CrispDMState.from_config(load_case_config(resolve_path("configs/titanic.yaml")))


def test_has_keys_contract():
    assert has_keys({"a": 1}, "a") == []
    assert has_keys({}, "a") == ["missing key 'a'"]


def test_execution_evidence_collect_requires_authored_code(monkeypatch, state, tmp_path):
    monkeypatch.setattr(crew, "run_text_task", lambda *a, **k: "")
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    with pytest.raises(crew.CrewKickoffError):
        execution_evidence(pyexec, state, "2.1", tmp_path)


def test_execution_evidence_collect_with_stub_code(monkeypatch, state, tmp_path):
    monkeypatch.setattr(crew, "run_text_task", fake_run_text_task)
    state.substep = "2.1"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = execution_evidence(pyexec, state, "2.1", tmp_path)
    assert "initial_data_collection_report" in out


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
    assert "TfidfVectorizer" in hint
    assert "FunctionTransformer" in hint


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


def test_ds_43_text_fallback_when_authored_code_fails(monkeypatch, tmp_path):
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

    broken = '''```python
train = pd.read_parquet(TRAIN_PARQUET)
if ID_COL not in train.columns:
    raise ValueError(f"missing {ID_COL}")
```'''

    monkeypatch.setattr(crew, "run_text_task", lambda *a, **k: broken)
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    out = ds_execution_evidence(pyexec, state, "4.3", tmp_path)
    assert "model_run" in out
    assert out["model_run"]["cv_score"] is not None
    assert out["model_run"]["technique"] == "tfidf_logreg"


def test_ds_44_text_fallback_when_authored_code_fails(monkeypatch, tmp_path):
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

    broken = '''```python
raise RuntimeError("no pipeline in state")
```'''

    monkeypatch.setattr(crew, "run_text_task", lambda *a, **k: broken)
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


def test_contract_fails_when_target_dropped(monkeypatch, state, tmp_path):
    """A 3.4 author that drops the target must never satisfy the contract.

    Locks the B1 wiring: the target check is unconditional, so an otherwise
    well-formed integrate (all keys present, valid JSON) still fails when the
    written parquet lost the target, exhausts retries + DEBUG, and halts.
    """
    drop_target = '''```python
import pandas as pd, json, os
def read_table(path):
    return pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path)
os.makedirs(OUTDIR, exist_ok=True)
tr = read_table(TRAIN_IN); te = read_table(TEST_IN)
shared = [c for c in tr.columns if c in te.columns]  # drops train-only TARGET
tr = tr[shared]
tp = os.path.join(OUTDIR, "train_integrated.parquet")
sp = os.path.join(OUTDIR, "test_integrated.parquet")
tr.to_parquet(tp); te.to_parquet(sp)
print(json.dumps({"train_out": tp, "test_out": sp,
                  "train_rows": int(len(tr)), "test_rows": int(len(te)),
                  "columns_train": list(tr.columns), "columns_test": list(te.columns)}))
```'''

    def fake(agent_name, instruction, st, **kwargs):
        if agent_name == "data_engineer" and st.substep == "3.4":
            return drop_target
        return ""  # developer DEBUG gets no usable fix -> stays STUCK

    monkeypatch.setattr(crew, "run_text_task", fake)
    state.substep = "3.4"
    pyexec = PythonExec(workdir=tmp_path / "sandbox")
    with pytest.raises(crew.CrewKickoffError):
        execution_evidence(pyexec, state, "3.4", tmp_path)
