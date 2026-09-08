"""Tests for deterministic case preflight (zero LLM tokens)."""
from __future__ import annotations

from pathlib import Path

import pytest

from maads.config import CaseConfig, DataPaths, SuccessCriterion
from maads.preflight import preflight_case


def _cfg(
    tmp_path: Path,
    *,
    case_id: str = "demo_case",
    train_rel: str | None = None,
    test_rel: str | None = None,
    target: str = "label",
    write_train: bool = True,
    write_test: bool = True,
    train_header: str = "id,label\n1,a\n",
) -> CaseConfig:
    data_dir = tmp_path / "data" / case_id
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if write_train:
        train_path.write_text(train_header, encoding="utf-8")
    if write_test and test_rel is not False:
        test_path.write_text("id\n1\n", encoding="utf-8")

    train_csv = train_rel or str(train_path)
    test_csv = None if test_rel is False else (test_rel or str(test_path))
    return CaseConfig(
        case_id=case_id,
        problem_statement="unit test",
        problem_type="classification",
        target_column=target,
        id_column="id",
        evaluation_metric="accuracy",
        data=DataPaths(train_csv=train_csv, test_csv=test_csv),
        success_criterion=SuccessCriterion(
            metric="accuracy", threshold=0.0, direction="maximize"
        ),
    )


def test_preflight_missing_train(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, write_train=False, write_test=False, test_rel=False)
    # Point at a missing file under the aligned data dir.
    missing = tmp_path / "data" / "demo_case" / "missing.csv"
    cfg = cfg.model_copy(
        update={"data": cfg.data.model_copy(update={"train_csv": str(missing), "test_csv": None})}
    )
    errors = preflight_case(cfg)
    assert errors
    assert any("train" in e.lower() for e in errors)


def test_preflight_hyphen_dir_vs_underscore_case_id(tmp_path: Path) -> None:
    bad_dir = tmp_path / "data" / "disaster-tweets"
    bad_dir.mkdir(parents=True)
    train = bad_dir / "train.csv"
    train.write_text("id,target\n1,0\n", encoding="utf-8")
    test = bad_dir / "test.csv"
    test.write_text("id\n1\n", encoding="utf-8")
    cfg = CaseConfig(
        case_id="disaster_tweets",
        problem_statement="unit test",
        problem_type="binary_classification",
        target_column="target",
        id_column="id",
        evaluation_metric="f1",
        data=DataPaths(train_csv=str(train), test_csv=str(test)),
        success_criterion=SuccessCriterion(
            metric="f1", threshold=0.0, direction="maximize"
        ),
    )
    errors = preflight_case(cfg)
    assert errors
    assert any("disaster-tweets" in e and "disaster_tweets" in e for e in errors)


def test_preflight_happy_path(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert preflight_case(cfg) == []


def test_preflight_blank_target_skips_column_check(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, target="")
    assert preflight_case(cfg) == []


def test_preflight_missing_target_column(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path, target="Sentiment", train_header="id,label\n1,a\n")
    errors = preflight_case(cfg)
    assert any("Sentiment" in e for e in errors)
