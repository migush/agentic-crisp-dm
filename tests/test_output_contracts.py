"""Tests for agent output contract validation."""
from __future__ import annotations

from maads.output_contracts import (
    normalize_agent_output,
    schema_hint_for_agent,
    validate_agent_output,
    _minimal_de_response,
    minimal_data_scientist_output,
    minimal_storyteller_output,
)


def test_validate_data_engineer_accepts_minimal_2_4():
    payload = _minimal_de_response()
    assert validate_agent_output("data_engineer", payload, substep="2.4") == []


def test_normalize_coerces_string_assumptions_for_data_scientist():
    payload = minimal_data_scientist_output(
        "4.1",
        state_updates={
            "md": {
                "modeling_technique": "logistic_regression",
                "modeling_assumptions": ["text is primary signal"],
            },
        },
        summary="Selected baseline technique.",
    )
    payload["assumptions"] = [
        "Text is the primary predictive signal.",
        "TF-IDF with logistic regression is the baseline.",
    ]
    payload["risks"] = ["Keyword missing not at random."]
    normalize_agent_output("data_scientist", payload)
    assert validate_agent_output(
        "data_scientist", payload, substep="4.1", normalize=False,
    ) == []
    assert payload["assumptions"][0] == {
        "statement": "Text is the primary predictive signal.",
    }
    assert payload["state_updates"]["md"]["modeling_assumptions"] == [
        "text is primary signal",
    ]


def test_normalize_coerces_empty_string_list_fields():
    payload = _minimal_de_response()
    payload["assumptions"] = ""
    payload["risks"] = ""
    normalize_agent_output("data_engineer", payload)
    assert payload["assumptions"] == []
    assert payload["risks"] == []
    assert validate_agent_output(
        "data_engineer", payload, substep="2.4", normalize=False,
    ) == []


def test_validate_rejects_debug_wrapper_comm_0010():
    wrapper = {
        "assignment_id": "debug-2.4",
        "agent": "data_engineer",
        "status": "FIXED",
        "summary": "The malformed JSON was corrected.",
        "state_updates": "",
        "evidence": [{"evidence_id": "x", "claim": "y", "source": "z", "method": "m"}],
        "decisions": [],
        "operations": [],
        "quality_findings": "",
        "validations": [],
        "artifacts": [],
        "assumptions": "",
        "risks": "",
        "blockers": "",
        "handoffs": [],
        "loop_signal": {
            "recommended": False,
            "contour": "NONE",
            "reason": None,
            "evidence_ids": [],
        },
        "completion_evidence": {
            "input_contract_valid": True,
            "required_outputs_present": True,
            "execution_succeeded": True,
            "artifacts_verified": True,
            "leakage_checks_passed": True,
            "reproducibility_checks_passed": True,
            "safe_for_downstream_use": True,
        },
    }
    errors = validate_agent_output("data_engineer", wrapper, substep="2.4")
    assert any("debug wrapper" in e for e in errors)
    assert any("FIXED" in e for e in errors)
    assert wrapper["state_updates"] == {}
    assert wrapper["assumptions"] == []


def test_validate_coerces_wrong_assignment_id_to_substep():
    payload = _minimal_de_response(substep="2.4")
    payload["assignment_id"] = "run-xyz:2.4:data_engineer"
    errors = validate_agent_output("data_engineer", payload, substep="2.4")
    assert errors == []
    assert payload["assignment_id"] == "2.4"


def test_validate_reports_wrong_assignment_id_when_normalize_disabled():
    payload = _minimal_de_response(substep="2.4")
    payload["assignment_id"] = "2.2"
    errors = validate_agent_output(
        "data_engineer", payload, substep="2.4", normalize=False,
    )
    assert any("assignment_id" in e for e in errors)
    assert payload["assignment_id"] == "2.2"


def test_normalize_leaves_debug_assignment_id_for_wrapper_check():
    payload = _minimal_de_response(substep="2.4")
    payload["assignment_id"] = "debug-2.4"
    normalize_agent_output("data_engineer", payload, substep="2.4")
    assert payload["assignment_id"] == "debug-2.4"


def test_validate_coerces_storyteller_assignment_id():
    payload = minimal_storyteller_output("6.2")
    payload["assignment_id"] = "abc:6.2:storyteller"
    errors = validate_agent_output("storyteller", payload, substep="6.2")
    assert errors == []
    assert payload["assignment_id"] == "6.2"


def test_schema_hint_states_exact_assignment_id():
    hint = schema_hint_for_agent("storyteller", substep="6.2")
    assert 'assignment_id must be exactly "6.2"' in hint
    hint_de = schema_hint_for_agent("data_engineer", substep="3.1")
    assert 'assignment_id must be exactly "3.1"' in hint_de
