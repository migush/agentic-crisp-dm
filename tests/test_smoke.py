"""Smoke tests for the drop-in utilities.

These confirm the scaffold's helper modules import and behave correctly.
The multi-agent orchestration itself is provided by whichever framework
you pick, so it isn't tested here.
"""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from maads.paths import repo_root
from maads.config import load_case_config
from maads.data_utils import CASE_SHORTHANDS, download_kaggle_competition
from maads.prompts import AGENT_PROMPTS
from maads.state import (
    CrispDMState,
    ModelRun,
    Phase,
    SUBSTEPS,
    SUBSTEP_NAMES,
    SUBSTEP_OWNER,
    next_substep,
)


REPO_ROOT = repo_root()


def test_all_three_configs_load():
    for case in ("titanic", "house_prices", "disaster_tweets"):
        cfg = load_case_config(REPO_ROOT / "configs" / f"{case}.yaml")
        assert cfg.case_id == case
        assert cfg.problem_statement
        assert cfg.target_column


def test_state_initialises_from_config():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    assert state.case_id == "titanic"
    assert state.phase == Phase.BUSINESS_UNDERSTANDING
    assert state.substep == "1.1"
    assert state.log == []
    assert state.loop_history == []
    # All six phase sub-objects exist and are empty.
    assert state.bu.business_objectives is None
    assert state.du.data_description_report is None
    assert state.dp.dataset == {}
    assert state.md.models == []
    assert state.ev.decision is None
    assert state.dep.submission_path is None


def test_24_substeps_with_names_and_owners():
    """The CRISP-DM 1.0 Reference Model has exactly 24 generic tasks."""
    assert sum(len(v) for v in SUBSTEPS.values()) == 24
    flat_ids = [s for v in SUBSTEPS.values() for s in v]
    assert set(flat_ids) == set(SUBSTEP_NAMES.keys()) == set(SUBSTEP_OWNER.keys())
    # Spot-check a few canonical names.
    assert SUBSTEP_NAMES["1.1"] == "Determine Business Objectives"
    assert SUBSTEP_NAMES["3.3"] == "Construct Data"
    assert SUBSTEP_NAMES["4.4"] == "Assess Model"
    assert SUBSTEP_NAMES["6.4"] == "Review Project"


def test_prereqs_block_phase_jumping():
    """A consumer of state can ask whether a substep's prereqs are met."""
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)

    assert state.substep_prereqs_satisfied("4.1") is False
    state.dp.dataset = {"train": "x.parquet", "test": "y.parquet"}
    assert state.substep_prereqs_satisfied("4.1") is True

    assert state.substep_prereqs_satisfied("5.1") is False
    state.md.models.append(ModelRun(technique="logistic_regression", cv_score=0.78))
    assert state.substep_prereqs_satisfied("5.1") is True

    assert state.substep_prereqs_satisfied("6.1") is False
    state.md.chosen_model = state.md.models[0]
    assert state.substep_prereqs_satisfied("6.1") is True


def test_view_for_returns_only_agent_slice():
    """Token-economy check: view_for must NOT return the full state."""
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)

    view = state.view_for("data_scientist")
    assert "case_id" in view
    assert "data_mining_goals" in view
    # The Data Scientist should not get the BU sub-object — that's for the
    # Domain Expert. The view should be a flat dict, not the state itself.
    assert "bu" not in view


def test_case_shorthands_are_just_a_convenience():
    """The data-download path must accept any Kaggle slug.

    The CASE_SHORTHANDS map is a convenience; the system is not limited
    to the keys in it.
    """
    assert set(CASE_SHORTHANDS) == {"titanic", "house_prices", "disaster_tweets"}
    # The download function used directly must take an arbitrary slug.
    sig = inspect.signature(download_kaggle_competition)
    assert "slug" in sig.parameters


@pytest.mark.parametrize("agent_id", list(AGENT_PROMPTS))
def test_agent_identity_embedded(agent_id):
    """Each registered agent has stable embedded role/goal/backstory."""
    p = AGENT_PROMPTS[agent_id]
    assert p["role"] and p["goal"] and p["backstory"]
    for token in ("{case_id}", "{state_view}", "{phase}", "{substep}"):
        assert token not in p["backstory"]
    if agent_id == "domain":
        assert "{dataset_name}" in p["role"]
    if agent_id == "data_engineer":
        assert p["role"] == "Senior Data Engineer"
        assert "CRISP-DM" in p["backstory"]
        assert "MODELING BOUNDARY" not in p["backstory"]
    if agent_id == "data_scientist":
        assert p["role"] == "Senior Data Scientist (Modeling & Evaluation)"
        assert "MODEL FAMILY SELECTION" in p["backstory"]
        assert "Leakage prevention" in p["backstory"]
        assert "MODELING BOUNDARY" not in p["backstory"]


def test_pm_backstory_contains_directive_schema():
    backstory = AGENT_PROMPTS["pm"]["backstory"]
    assert "four loop contours" in backstory
    assert '"action": "advance | loop_back | halt"' in backstory


def test_view_for_pm_outputs_status():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    view = state.view_for("pm")
    assert "outputs_status" in view
    assert view["outputs_status"]["phase_1_ready"] is False
    state.bu.business_objectives = "obj"
    state.bu.business_success_criteria = "crit"
    state.bu.data_mining_goals = "goals"
    state.bu.project_plan = ["step"]
    view = state.view_for("pm")
    assert view["outputs_status"]["phase_1_ready"] is True


def test_phase_6_ready_requires_experience_documentation():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    state.dep.submission_path = "/artifacts/titanic/submission.csv"
    state.dep.final_report_path = "/artifacts/titanic/final_report.md"
    view = state.view_for("pm")
    assert view["outputs_status"]["phase_6_ready"] is False
    state.dep.experience_documentation = "Lessons from the run."
    view = state.view_for("pm")
    assert view["outputs_status"]["phase_6_ready"] is True


def test_next_substep_walks_phases():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    assert next_substep(state) == "1.2"
    state.substep = "1.4"
    assert next_substep(state) == "2.1"
    state.phase = Phase.DEPLOYMENT
    state.substep = "6.4"
    assert next_substep(state) is None


def test_de_view_for_is_substep_scoped():
    cfg = load_case_config(REPO_ROOT / "configs" / "titanic.yaml")
    state = CrispDMState.from_config(cfg)
    state.du.data_description_report = {"n_rows": 10}
    state.du.data_quality_report = {"blockers": []}
    state.du.data_exploration_report = {"notes": "x"}
    state.dp.rationale_for_inclusion_exclusion = {"keep": ["Age"]}
    state.dp.data_cleaning_report = {"ops": []}
    state.dp.derived_attributes = {"items": [{"field": "FamilySize"}]}
    state.dp.generated_records = {"count": 0}
    state.dp.merged_data = {"path": "merged.parquet"}
    state.dp.reformatted_data = {"huge": True}

    state.substep = "2.1"
    v21 = state.view_for("data_engineer")
    assert v21["du_so_far"] == {}
    assert v21["dp_so_far"] == {}
    assert "feature_hints" in v21

    state.substep = "2.4"
    v24 = state.view_for("data_engineer")
    assert set(v24["du_so_far"]) == {"data_description_report"}
    assert v24["dp_so_far"] == {}

    state.substep = "3.1"
    v31 = state.view_for("data_engineer")
    assert set(v31["du_so_far"]) == {
        "data_quality_report",
        "data_description_report",
        "data_exploration_report",
    }
    assert v31["dp_so_far"] == {}

    state.substep = "3.5"
    v35 = state.view_for("data_engineer")
    assert "reformatted_data" not in v35["dp_so_far"]
    assert set(v35["dp_so_far"]) == {
        "rationale_for_inclusion_exclusion",
        "data_cleaning_report",
        "derived_attributes",
        "generated_records",
        "merged_data",
    }


def test_domain_agent_role_formats_dataset_name():
    from maads.crew_base import agent_for, reset_llm_caches

    reset_llm_caches()
    agent = agent_for("domain", "titanic")
    assert "{dataset_name}" not in agent.role
    assert "titanic" in agent.role.lower() or "titanic" in agent.role

