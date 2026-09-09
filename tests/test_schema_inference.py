"""Deterministic hosted-case schema inference (no case_id branches)."""
from __future__ import annotations

from pathlib import Path

import yaml

from maads.config import CaseConfig, DataPaths, DataSource, SuccessCriterion, load_case_config
from maads.schema_inference import (
    apply_inferred_schema,
    filename_role,
    infer_column_roles_from_profile,
    infer_feature_hints_from_profile,
    infer_source_paths,
    read_csv,
    reconcile_data_paths,
)
from maads.state import CrispDMState


def test_filename_role_token_boundaries():
    assert filename_role("Corona_NLP_train.csv") == "train"
    assert filename_role("Corona_NLP_test.csv") == "test"
    assert filename_role("holdout.csv") == "test"
    assert filename_role("sample_submission.csv") == "sample_submission"
    assert filename_role("contest.csv") is None
    assert filename_role("widgets.csv") is None


def test_infer_source_paths_prefers_train_named_over_alphabetical_test():
    sources = [
        DataSource(path="/data/Corona_NLP_test.csv", original_filename="Corona_NLP_test.csv"),
        DataSource(path="/data/Corona_NLP_train.csv", original_filename="Corona_NLP_train.csv"),
    ]
    inferred = infer_source_paths(sources)
    assert Path(inferred["train"]).name == "Corona_NLP_train.csv"
    assert Path(inferred["test"]).name == "Corona_NLP_test.csv"


def test_reconcile_overrides_test_named_train_csv():
    data = DataPaths(
        train_csv="/data/Corona_NLP_test.csv",
        sources=[
            DataSource(path="/data/Corona_NLP_test.csv", original_filename="Corona_NLP_test.csv"),
            DataSource(path="/data/Corona_NLP_train.csv", original_filename="Corona_NLP_train.csv"),
        ],
    )
    out = reconcile_data_paths(data)
    assert Path(out.train_csv).name == "Corona_NLP_train.csv"
    assert Path(out.test_csv).name == "Corona_NLP_test.csv"


def test_load_case_config_rebinds_swapped_train_test(tmp_path: Path):
    train = tmp_path / "Corona_NLP_train.csv"
    test = tmp_path / "Corona_NLP_test.csv"
    train.write_text("UserName,Sentiment\n1,pos\n", encoding="utf-8")
    test.write_text("UserName,Sentiment\n9,neg\n", encoding="utf-8")
    yaml_path = tmp_path / "case.yaml"
    payload = {
        "case_id": "covid_like",
        "problem_statement": "Classify sentiment in a status.",
        "problem_type": "classification",
        "target_column": "",
        "id_column": "",
        "evaluation_metric": "accuracy",
        "data": {
            "sources": [
                {"path": str(test), "original_filename": "Corona_NLP_test.csv"},
                {"path": str(train), "original_filename": "Corona_NLP_train.csv"},
            ],
            "train_csv": str(test),
        },
        "success_criterion": {"metric": "accuracy", "threshold": 0.0, "direction": "maximize"},
    }
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    cfg = load_case_config(yaml_path)
    assert Path(cfg.data.train_csv).name == "Corona_NLP_train.csv"
    assert Path(cfg.data.test_csv).name == "Corona_NLP_test.csv"


def test_infer_sentiment_and_username_from_profile():
    n = 40
    sentiments = ["Positive", "Negative", "Neutral", "Extremely Positive", "Extremely Negative"]
    profile = {
        "n_rows": n,
        "columns": ["UserName", "ScreenName", "Location", "TweetAt", "OriginalTweet", "Sentiment"],
        "dtypes": {
            "UserName": "int64",
            "ScreenName": "int64",
            "Location": "object",
            "TweetAt": "object",
            "OriginalTweet": "object",
            "Sentiment": "object",
        },
        "missing": {"UserName": 0, "ScreenName": 0, "Location": 8, "TweetAt": 0, "OriginalTweet": 0, "Sentiment": 0},
        "cardinality": {
            "UserName": n,
            "ScreenName": n,
            "Location": 22,
            "TweetAt": 12,
            "OriginalTweet": n,
            "Sentiment": 5,
        },
    }
    roles = infer_column_roles_from_profile(profile, "classification")
    assert roles["target"] == "Sentiment"
    assert roles["id"] == "UserName"
    hints = infer_feature_hints_from_profile(profile, target="Sentiment", id_column="UserName")
    assert hints["text_free"][0] == "OriginalTweet"
    assert "tfidf_logreg" in hints["representation_options"]


def test_apply_inferred_schema_on_blank_covid_like_state(tmp_path: Path):
    n = 30
    sentiments = ["pos", "neg", "neu"]
    train = tmp_path / "Corona_NLP_train.csv"
    rows = ["UserName,OriginalTweet,Sentiment"]
    for i in range(n):
        rows.append(f"{i},tweet text {i},{sentiments[i % 3]}")
    train.write_text("\n".join(rows) + "\n", encoding="utf-8")
    cfg = CaseConfig(
        case_id="covid_like",
        problem_statement="Classify sentiment.",
        problem_type="classification",
        target_column="",
        id_column="",
        evaluation_metric="accuracy",
        data=DataPaths(train_csv=str(train)),
        success_criterion=SuccessCriterion(metric="accuracy", threshold=0.0, direction="maximize"),
    )
    state = CrispDMState.from_config(cfg)
    from maads.capabilities import ml_tools

    profile = ml_tools.profile_dataset(train, target=None)
    fields = apply_inferred_schema(state, profile=profile)
    assert "config.target_column" in fields
    assert state.config.target_column == "Sentiment"
    assert state.config.id_column == "UserName"
    assert state.config.feature_hints.get("text_free") == ["OriginalTweet"]


def test_read_csv_latin1(tmp_path: Path):
    path = tmp_path / "Corona_NLP_train.csv"
    path.write_bytes("UserName,OriginalTweet\n1,café status\n".encode("latin-1"))
    df = read_csv(path)
    assert list(df.columns) == ["UserName", "OriginalTweet"]
    assert "café" in str(df.iloc[0]["OriginalTweet"])


def test_does_not_override_configured_target():
    profile = {
        "n_rows": 10,
        "columns": ["id", "text", "target"],
        "dtypes": {"id": "int64", "text": "object", "target": "int64"},
        "missing": {"id": 0, "text": 0, "target": 0},
        "cardinality": {"id": 10, "text": 10, "target": 2},
    }
    cfg = CaseConfig(
        case_id="demo",
        problem_statement="x",
        problem_type="classification",
        target_column="target",
        id_column="id",
        evaluation_metric="accuracy",
        data=DataPaths(train_csv="missing.csv"),
        feature_hints={"text_free": ["text"]},
        success_criterion=SuccessCriterion(metric="accuracy", threshold=0.0, direction="maximize"),
    )
    state = CrispDMState.from_config(cfg)
    apply_inferred_schema(state, profile=profile)
    assert state.config.target_column == "target"
    assert state.config.feature_hints == {"text_free": ["text"]}
