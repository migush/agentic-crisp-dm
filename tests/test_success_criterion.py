"""Tests for canonical success-criterion evaluation."""
from __future__ import annotations

from maads.success_criterion import (
    assessment_meets,
    assessment_summary_phrase,
    criterion_direction,
    normalize_assessment,
    score_meets_threshold,
)


def test_rmse_log_minimize_meets_when_below_threshold():
    assert criterion_direction("rmse_log") == "minimize"
    assert score_meets_threshold(0.1355, 0.15, direction="minimize") is True
    assert score_meets_threshold(0.16, 0.15, direction="minimize") is False


def test_accuracy_maximize_meets_when_above_threshold():
    assert criterion_direction("accuracy") == "maximize"
    assert score_meets_threshold(0.82, 0.77, direction="maximize") is True
    assert score_meets_threshold(0.70, 0.77, direction="maximize") is False


def test_normalize_assessment_maps_success_criterion_met_alias():
    out = normalize_assessment(
        {"success_criterion_met": True, "achieved_score": 0.1355},
        metric="rmse_log",
        threshold=0.15,
        direction="minimize",
    )
    assert out["meets"] is True
    assert out["success_criterion_met"] is True


def test_normalize_assessment_recomputes_meets_from_score():
    out = normalize_assessment(
        {"meets": False, "success_criterion_met": False},
        metric="rmse_log",
        threshold=0.15,
        direction="minimize",
        cv_score=0.1355,
    )
    assert out["meets"] is True
    assert out["success_criterion_met"] is True
    assert out["cv_score"] == 0.1355


def test_normalize_assessment_ignores_non_dict_llm_response():
    out = normalize_assessment(
        "Model achieved 0.819 accuracy, exceeding the 0.77 threshold.",
        metric="accuracy",
        threshold=0.77,
        direction="maximize",
        cv_score=0.819,
    )
    assert out["meets"] is True
    assert out["cv_score"] == 0.819

    out_list = normalize_assessment(
        ["meets threshold"],
        metric="accuracy",
        threshold=0.77,
        direction="maximize",
        cv_score=0.5,
    )
    assert out_list["meets"] is False
    assert out_list["cv_score"] == 0.5


def test_assessment_meets_reads_either_field():
    assert assessment_meets({"meets": True}) is True
    assert assessment_meets({"success_criterion_met": True}) is True
    assert assessment_meets({"meets_success_criterion": True}) is True
    assert assessment_meets({"meets": False, "success_criterion_met": True}) is False
    assert assessment_meets(None) is False


def test_normalize_assessment_coerces_nested_achieved_score():
    out = normalize_assessment(
        {
            "success_threshold": 0.78,
            "achieved_score": {
                "cv_mean": 0.7519275973248275,
                "cv_std": 0.01620300942747244,
            },
            "meets_success_criterion": False,
        },
        metric="f1",
        threshold=0.78,
    )
    assert out["achieved_score"] == 0.7519275973248275
    assert out["cv_score"] == 0.7519275973248275
    assert out["threshold"] == 0.78
    assert out["meets"] is False
    assert out["success_criterion_met"] is False


def test_assessment_summary_phrase_handles_nested_achieved_score():
    assert assessment_summary_phrase(
        {"metric": "rmse_log", "direction": "minimize", "meets": True},
        cv_score=0.1355,
    ) == "CV 0.1355 meets threshold."
    assert "above threshold" in assessment_summary_phrase(
        {"metric": "rmse_log", "direction": "minimize", "meets": False},
        cv_score=0.16,
    )
    assert "below threshold" in assessment_summary_phrase(
        {"metric": "accuracy", "direction": "maximize", "meets": False},
        cv_score=0.70,
    )
    assert "below threshold" in assessment_summary_phrase(
        {
            "metric": "f1",
            "direction": "maximize",
            "meets": False,
            "achieved_score": {"cv_mean": 0.7519},
        },
    )
