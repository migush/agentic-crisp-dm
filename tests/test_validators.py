"""Tests for the state-artifact validators."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from maads.config import load_case_config
from maads.paths import resolve_path
from maads.state import CrispDMState, ModelRun
from maads.validators import validate_phase_3_artifacts, validate_phase_4_models

@pytest.fixture
def state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def test_phase3_clean_passes(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"Survived": [0, 1], "Age": [22, 38], "FamilySize": [1, 2]}))
    state.dp.dataset = {"train": str(train), "test": str(train)}
    state.dp.derived_attributes = {"items": ["FamilySize"]}
    assert validate_phase_3_artifacts(state) == []


def test_phase3_no_feature_columns(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"PassengerId": [1, 2], "Survived": [0, 1]}))
    state.dp.dataset = {"train": str(train), "test": str(train)}
    errors = validate_phase_3_artifacts(state)
    assert any("no feature columns" in e for e in errors)


def test_phase3_merged_data_no_predictors(state: CrispDMState):
    state.dp.merged_data = {"columns_train": ["PassengerId", "Survived"]}
    state.dp.dataset = {"train": "/missing/train.parquet", "test": "/missing/test.parquet"}
    errors = validate_phase_3_artifacts(state)
    assert any("merged_data.columns_train has no predictor" in e for e in errors)


def test_phase3_rationale_columns_missing(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"Survived": [0, 1], "Age": [22, 38]}))
    state.dp.dataset = {"train": str(train), "test": str(train)}
    state.dp.rationale_for_inclusion_exclusion = {
        "included_columns": ["Survived", "Age", "Sex", "Pclass"],
    }
    errors = validate_phase_3_artifacts(state)
    assert any("rationale included columns missing" in e for e in errors)


def test_phase3_missing_parquet(state: CrispDMState):
    state.dp.dataset = {"train": "/no/such/train.parquet", "test": "/no/such/test.parquet"}
    errors = validate_phase_3_artifacts(state)
    assert any("does not exist" in e for e in errors)


def test_phase3_missing_derived_feature(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"Survived": [0, 1], "Age": [22, 38]}))
    state.dp.dataset = {"train": str(train), "test": str(train)}
    state.dp.derived_attributes = {"items": ["FamilySize"]}  # claimed but absent
    errors = validate_phase_3_artifacts(state)
    assert any("FamilySize" in e for e in errors)


def test_phase3_blank_config_target_uses_exploration_report(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(
        train,
        pd.DataFrame({"Sentiment": ["pos", "neg"], "OriginalTweet": ["a", "b"]}),
    )
    state.config = state.config.model_copy(update={"target_column": ""})
    state.du.data_exploration_report = {"n_rows": 2, "target": "Sentiment"}
    state.dp.dataset = {"train": str(train), "test": str(train)}
    assert validate_phase_3_artifacts(state) == []
    assert state.config.target_column == "Sentiment"


def test_phase3_blank_target_without_inference_is_explicit(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"Sentiment": ["pos", "neg"], "note": ["a", "b"]}))
    state.config = state.config.model_copy(update={"target_column": ""})
    state.dp.dataset = {"train": str(train), "test": str(train)}
    errors = validate_phase_3_artifacts(state)
    assert any("target_column is unset" in e for e in errors)
    assert not any("target '' not in" in e for e in errors)


def test_phase3_target_nan(tmp_path: Path, state: CrispDMState):
    train = tmp_path / "train.parquet"
    _write_parquet(train, pd.DataFrame({"Survived": [0, None], "Age": [22, 38]}))
    state.dp.dataset = {"train": str(train), "test": str(train)}
    errors = validate_phase_3_artifacts(state)
    assert any("missing values" in e for e in errors)


def test_phase4_no_models(state: CrispDMState):
    assert validate_phase_4_models(state) == ["no models were produced in Phase 4"]


def test_phase4_assessment_without_score(state: CrispDMState):
    state.md.models = [ModelRun(technique="rf", cv_score=None, assessment="great")]
    errors = validate_phase_4_models(state)
    assert any("no cv_score" in e for e in errors)


def test_phase4_clean(state: CrispDMState):
    run = ModelRun(technique="rf", cv_score=0.81, assessment="ok")
    state.md.models = [run]
    state.md.chosen_model = run
    assert validate_phase_4_models(state) == []


def test_phase4_missing_chosen_model(state: CrispDMState):
    state.md.models = [ModelRun(technique="rf", cv_score=0.81, assessment="ok")]
    errors = validate_phase_4_models(state)
    assert any("chosen_model is not set" in e for e in errors)


def test_phase4_suspicious_cv_score(state: CrispDMState):
    run = ModelRun(technique="rf", cv_score=1.0, assessment="perfect")
    state.md.models = [run]
    state.md.chosen_model = run
    errors = validate_phase_4_models(state)
    assert any("exceeds sanity ceiling" in e for e in errors)
