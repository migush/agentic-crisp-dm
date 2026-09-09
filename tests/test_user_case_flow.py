"""User-case YAML: raw labelled sources, no sample submission, mocked agents complete."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from maads.config import load_case_config
from maads.agents import Plan
from maads.flow import phase_runner as pr
from maads.state import CrispDMState
from maads.testing.fake_llm import fake_llm_response
from maads.testing.flow_harness import make_flow


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    path.write_text(
        "rec_id,note,label\n" + "\n".join(f"{a},{b},{c}" for a, b, c in rows) + "\n",
        encoding="utf-8",
    )


@pytest.fixture
def user_case_config(tmp_path: Path) -> Path:
    labelled = tmp_path / "observations.csv"
    holdout = tmp_path / "holdout.csv"
    labels = ["alpha", "beta", "gamma"]
    train_rows = [(str(i), f"note {i}", labels[i % 3]) for i in range(1, 31)]
    holdout_rows = [(str(100 + i), f"hold {i}", labels[i % 3]) for i in range(6)]
    _write_csv(labelled, train_rows)
    _write_csv(holdout, holdout_rows)
    yaml_path = tmp_path / "case.yaml"
    payload = {
        "case_id": "widget_labels",
        "problem_statement": "Predict the class of each widget from a short note.",
        "problem_type": "classification",
        "target_column": "label",
        "id_column": "rec_id",
        "evaluation_metric": "accuracy",
        "data": {
            "sources": [
                {"path": str(labelled), "role": "labelled"},
                {"path": str(holdout), "role": "holdout"},
            ],
            "train_csv": str(labelled),
            "test_csv": str(holdout),
        },
        "success_criterion": {"metric": "accuracy", "threshold": 0.0, "direction": "maximize"},
    }
    yaml_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return yaml_path


def test_load_user_case_optional_sample_and_sources(user_case_config: Path):
    cfg = load_case_config(user_case_config)
    assert cfg.data.sample_submission_csv is None
    assert cfg.data.test_csv
    assert len(cfg.data.sources) == 2
    assert Path(cfg.data.train_csv).name == "observations.csv"
    de_view_paths = cfg.data.model_dump()
    assert de_view_paths["sample_submission_csv"] is None


@patch("maads.agents.run_json_task")
def test_mocked_flow_completes_raw_labelled_bundle(
    mock_llm, user_case_config: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")
    mock_llm.side_effect = fake_llm_response
    cfg = load_case_config(user_case_config)
    state = CrispDMState.from_config(cfg)
    artifact_dir = tmp_path / "artifacts" / "widget_labels"
    artifact_dir.mkdir(parents=True)

    orig = pr.run_substep
    seen_sources: list[str] = []

    def track(ctx, substep: str) -> bool:
        if substep == "2.1":
            view = ctx.state.view_for("data_engineer")
            raw = view.get("raw_data_paths") or {}
            for src in raw.get("sources") or []:
                if isinstance(src, dict):
                    seen_sources.append(str(src.get("path") or ""))
                else:
                    seen_sources.append(str(getattr(src, "path", src)))
            seen_sources.append(str(raw.get("train_csv") or ""))
        return orig(ctx, substep)

    flow = make_flow(state, artifact_dir)
    plans = [Plan(action="advance", reason="ok") for _ in range(60)]
    plan_iter = iter(plans)
    flow._pm.plan = lambda _s: next(plan_iter, Plan(action="halt", reason="done"))  # type: ignore[method-assign]
    with patch.object(pr, "run_substep", side_effect=track):
        flow.run()

    joined = " ".join(seen_sources)
    assert "observations.csv" in joined
    assert state.config.data.sample_submission_csv is None
    assert state.dep.submission_path
    assert Path(state.dep.submission_path).is_file()


@patch("maads.agents.run_json_task")
def test_mocked_flow_infers_blank_target_and_train_file(
    mock_llm, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")
    mock_llm.side_effect = fake_llm_response
    labels = ["Positive", "Negative", "Neutral"]
    train = tmp_path / "Corona_NLP_train.csv"
    test = tmp_path / "Corona_NLP_test.csv"
    train_rows = ["UserName,OriginalTweet,Sentiment"]
    test_rows = ["UserName,OriginalTweet,Sentiment"]
    for i in range(1, 37):
        train_rows.append(f"{i},status update number {i} about supplies,{labels[i % 3]}")
    for i in range(100, 112):
        test_rows.append(f"{i},holdout status {i},{labels[i % 3]}")
    train.write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    test.write_text("\n".join(test_rows) + "\n", encoding="utf-8")
    yaml_path = tmp_path / "case.yaml"
    payload = {
        "case_id": "covid_like",
        "problem_statement": "Classify the sentiment expressed in a status.",
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
    state = CrispDMState.from_config(cfg)
    artifact_dir = tmp_path / "artifacts" / "covid_like"
    artifact_dir.mkdir(parents=True)

    flow = make_flow(state, artifact_dir)
    plans = [Plan(action="advance", reason="ok") for _ in range(60)]
    plan_iter = iter(plans)
    flow._pm.plan = lambda _s: next(plan_iter, Plan(action="halt", reason="done"))  # type: ignore[method-assign]
    flow.run()

    assert state.config.target_column == "Sentiment"
    assert not state.halted or "execution failed at 4.3" not in (state.halt_reason or "")
    assert state.md.models
    assert state.dep.submission_path
    assert Path(state.dep.submission_path).is_file()
