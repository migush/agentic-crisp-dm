"""Tests for semantic Loop A, quality_gate view, and domain 1.3 persistence."""
from __future__ import annotations

import json

import pytest

from maads.capabilities.domain import apply_refine_goals
from maads.config import load_case_config
from maads.paths import resolve_path
from maads.state import CrispDMState


@pytest.fixture
def house_state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/house_prices.yaml"))
    return CrispDMState.from_config(cfg)


def test_pm_view_quality_gate_includes_na_means_absent(house_state: CrispDMState):
    house_state.substep = "3.1"
    house_state.du.data_quality_report = {
        "blockers": ["Column 'Alley' has 93.77% missing values (> 40%)."],
        "tolerable": ["Column 'LotFrontage' has 17.74% missing values."],
    }
    house_state.bu.inventory_of_resources = {
        "domain_artifacts": {
            "loop_a_recommendation": {"should_trigger": False, "reason": "structural absence"},
            "domain_data_quality_flags": [{"feature": "Alley", "risk": "NA means no alley"}],
        },
    }
    pm_view = house_state.view_for("pm")
    gate = pm_view["quality_gate"]
    assert "Alley" in gate["na_means_absent"]
    assert gate["data_quality_report"]["blockers"]
    assert gate["loop_a_recommendation"]["should_trigger"] is False
    assert pm_view["latest_quality_blockers"]


def test_apply_refine_goals_persists_loop_a_recommendation(house_state: CrispDMState):
    apply_refine_goals(
        {
            "data_mining_goal": "Predict SalePrice with absence encoding.",
            "success_criterion": {"metric": "rmse_log", "target_value": "0.15", "direction": "minimize"},
            "loop_a_recommendation": {
                "should_trigger": False,
                "reason": "High missingness is structural absence per na_means_absent.",
            },
        },
        house_state,
    )
    artifacts = house_state.bu.inventory_of_resources["domain_artifacts"]
    assert artifacts["loop_a_recommendation"]["should_trigger"] is False


def test_de_dataset_context_includes_na_means_absent(house_state: CrispDMState):
    from maads.capabilities.shared import de_dataset_context

    train = resolve_path("data/house_prices/train.csv")
    test = resolve_path("data/house_prices/test.csv")
    if not train.exists():
        pytest.skip("house_prices data not present")
    ctx = de_dataset_context(house_state, str(train), str(test))
    inspect = json.loads(ctx["DATASET_INSPECT_JSON"])
    assert "PoolQC" in inspect["na_means_absent"]
    assert "Alley" in inspect["na_means_absent"]


def test_disaster_tweets_marks_keyword_location_as_structural_absence() -> None:
    cfg = load_case_config(resolve_path("configs/disaster_tweets.yaml"))
    assert cfg.feature_hints["na_means_absent"] == ["keyword", "location"]
    state = CrispDMState.from_config(cfg)
    assert state.view_for("pm")["quality_gate"]["na_means_absent"] == [
        "keyword",
        "location",
    ]


def test_titanic_documents_cabin_as_structural_absence() -> None:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    assert "Cabin" in cfg.feature_hints["na_means_absent"]
    assert "Cabin" in cfg.feature_hints["high_missing"]
    state = CrispDMState.from_config(cfg)
    gate = state.view_for("pm")["quality_gate"]
    assert "Cabin" in gate["na_means_absent"]
    assert "Cabin" in gate["high_missing"]
