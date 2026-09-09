"""Tests for Developer DEBUG mode (PythonExec + JSON repair)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from maads.codegen import run_authored_code
from maads.config import load_case_config
from maads.crew import CrewKickoffError, run_json_task
from maads.debug import (
    MAX_DEBUG_RETRIES,
    classify_exec_error,
    debug_json_parse,
    debug_python_exec,
)
from maads.paths import resolve_path
from maads.state import CrispDMState
from maads.tools import ExecResult, PythonExec
from maads.output_contracts import _minimal_de_response, minimal_data_scientist_output


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")


@pytest.fixture
def state() -> CrispDMState:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    return CrispDMState.from_config(cfg)


@pytest.fixture
def artifact_dir(tmp_path: Path) -> Path:
    d = tmp_path / "artifacts" / "titanic"
    d.mkdir(parents=True)
    return d


@pytest.fixture
def pyexec(artifact_dir: Path) -> PythonExec:
    return PythonExec(workdir=artifact_dir / "sandbox")


def test_classify_exec_error_labels_schema_and_timeout():
    assert classify_exec_error(
        ExecResult(ok=False, stdout="", stderr="KeyError: 'Age'", return_code=1),
    ) == "schema_error"
    assert classify_exec_error(
        ExecResult(ok=False, stdout="", stderr="timed out", return_code=-1, timed_out=True),
    ) == "timeout"


def test_debug_json_parse_normalizes_schema_without_developer_llm(
    state: CrispDMState,
    artifact_dir: Path,
):
    state.substep = "4.1"
    invalid = minimal_data_scientist_output(
        "4.1",
        state_updates={
            "md": {
                "modeling_technique": "logistic_regression",
                "modeling_assumptions": ["text baseline"],
            },
        },
    )
    invalid["assumptions"] = ["Plain string assumption"]
    invalid["risks"] = ["Plain string risk"]
    outcome = debug_json_parse(
        state=state,
        artifact_dir=artifact_dir,
        requesting_agent="data_scientist",
        raw_text=json.dumps(invalid),
        failure_kind="json_schema",
        invalid_payload=invalid,
    )
    assert outcome.status == "FIXED"
    assert outcome.repair_kind == "deterministic_schema"
    assert outcome.payload is not None
    assert outcome.payload["assumptions"][0]["statement"] == "Plain string assumption"


def test_debug_json_parse_repairs_trailing_comma(state: CrispDMState, artifact_dir: Path):
    raw = '{"action": "advance", "reason": "ok",}'
    outcome = debug_json_parse(
        state=state,
        artifact_dir=artifact_dir,
        requesting_agent="pm",
        raw_text=raw,
        schema_hint='{"action": "advance"}',
    )
    assert outcome.status == "FIXED"
    assert outcome.payload == {"action": "advance", "reason": "ok"}


def test_debug_json_parse_uses_developer_llm_when_needed(
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: '{"action": "advance", "reason": "repaired"}',
    )
    outcome = debug_json_parse(
        state=state,
        artifact_dir=artifact_dir,
        requesting_agent="pm",
        raw_text="not json at all",
        schema_hint="",
    )
    assert outcome.status == "FIXED"
    assert outcome.payload["action"] == "advance"


def test_debug_json_parse_rejects_developer_wrapper_for_data_engineer(
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    state.substep = "2.4"
    wrapper = {
        "assignment_id": "debug-2.4",
        "agent": "data_engineer",
        "status": "FIXED",
        "summary": "repaired",
        "state_updates": "",
        "evidence": [],
        "decisions": [],
        "operations": [],
        "quality_findings": "",
        "validations": [],
        "artifacts": [],
        "assumptions": "",
        "risks": "",
        "blockers": [],
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
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: json.dumps(wrapper),
    )
    outcome = debug_json_parse(
        state=state,
        artifact_dir=artifact_dir,
        requesting_agent="data_engineer",
        raw_text="not json",
        schema_hint="",
    )
    assert outcome.status == "STUCK"
    assert outcome.schema_ok is False
    assert outcome.schema_errors


@patch("maads.crew._kickoff")
def test_run_json_task_rejects_schema_invalid_developer_repair(
    mock_kickoff,
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    state.substep = "2.4"
    mock_kickoff.return_value = "broken output"
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: json.dumps({
            "assignment_id": "debug-2.4",
            "agent": "data_engineer",
            "status": "FIXED",
            "summary": "repaired",
            "state_updates": "",
            "evidence": [],
            "decisions": [],
            "operations": [],
            "quality_findings": "",
            "validations": [],
            "artifacts": [],
            "assumptions": "",
            "risks": "",
            "blockers": [],
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
        }),
    )
    with pytest.raises(CrewKickoffError, match="schema-invalid|non-JSON"):
        run_json_task(
            "data_engineer",
            "verify quality",
            state,
            artifact_dir=artifact_dir,
        )


@patch("maads.crew._kickoff")
def test_run_json_task_accepts_schema_valid_developer_repair(
    mock_kickoff,
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    state.substep = "2.4"
    mock_kickoff.return_value = "broken output"
    repaired = _minimal_de_response()
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: json.dumps(repaired),
    )
    parsed = run_json_task(
        "data_engineer",
        "verify quality",
        state,
        artifact_dir=artifact_dir,
    )
    assert parsed["assignment_id"] == "2.4"
    assert parsed["status"] == "COMPLETED"


def test_debug_python_exec_fixes_failing_code(
    monkeypatch: pytest.MonkeyPatch,
    pyexec: PythonExec,
    state: CrispDMState,
    artifact_dir: Path,
):
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: '```python\nimport json\nprint(json.dumps({"n": 2}))\n```',
    )
    outcome = debug_python_exec(
        pyexec=pyexec,
        state=state,
        artifact_dir=artifact_dir,
        requesting_agent="data_engineer",
        failing_code="raise RuntimeError('boom')",
        header_vars={"N": 1},
        contract=lambda p: [] if p.get("n") == 2 else ["wrong"],
        contract_hint='{"n": int}',
        last_error="RuntimeError: boom",
        last_exec=ExecResult(ok=False, stdout="", stderr="RuntimeError: boom", return_code=1),
        max_retries=MAX_DEBUG_RETRIES,
    )
    assert outcome.status == "FIXED"
    assert outcome.payload == {"n": 2}


def test_run_authored_code_uses_developer_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
    pyexec: PythonExec,
    state: CrispDMState,
    artifact_dir: Path,
):
    calls: list[str] = []

    def fake_text(agent_name, *_a, **_k):
        calls.append(agent_name)
        if agent_name == "developer":
            return '```python\nimport json\nprint(json.dumps({"ok": True}))\n```'
        return '```python\nraise ValueError("fail")\n```'

    monkeypatch.setattr("maads.crew.run_text_task", fake_text)
    monkeypatch.setattr("maads.crew.run_text_task", fake_text)

    res = run_authored_code(
        pyexec=pyexec,
        agent_name="data_engineer",
        state=state,
        instruction="produce ok",
        header_vars={},
        contract=lambda p: [] if p.get("ok") else ["missing ok"],
        fallback=lambda: {"ok": False},
        max_retries=1,
        artifact_dir=artifact_dir,
    )
    assert res.ok and not res.degraded
    assert res.payload == {"ok": True}
    assert "developer" in calls


@patch("maads.crew._kickoff")
def test_run_json_task_routes_malformed_json_to_developer(
    mock_kickoff,
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    mock_kickoff.return_value = "Here is JSON: {bad"
    monkeypatch.setattr(
        "maads.crew.run_text_task",
        lambda *_a, **_k: '{"action": "advance", "reason": "fixed"}',
    )
    parsed = run_json_task(
        "pm", "decide", state, artifact_dir=artifact_dir,
    )
    assert parsed == {"action": "advance", "reason": "fixed"}


@patch("maads.crew._kickoff")
def test_run_json_task_still_raises_when_debug_stuck(
    mock_kickoff,
    monkeypatch: pytest.MonkeyPatch,
    state: CrispDMState,
    artifact_dir: Path,
):
    mock_kickoff.return_value = "totally broken"
    monkeypatch.setattr("maads.crew.run_text_task", lambda *_a, **_k: "still not json")
    with pytest.raises(CrewKickoffError, match="non-JSON"):
        run_json_task("pm", "decide", state, artifact_dir=artifact_dir)


def test_de_32_uses_deterministic_clean_without_authored_debug(
    monkeypatch: pytest.MonkeyPatch,
    pyexec: PythonExec,
    state: CrispDMState,
    artifact_dir: Path,
):
    """DE 3.2 runs deterministic clean_tables — no authored-code / DEBUG path."""
    from maads.agents import DataEngineerAgent

    calls: list[str] = []

    def fake_text(agent_name, *_a, **_k):
        calls.append(agent_name)
        return "```python\nraise RuntimeError('should not author code')\n```"

    monkeypatch.setattr("maads.crew.run_text_task", fake_text)

    def fake_json_task(_agent, _instruction, st, *_a, **_k):
        payload = _minimal_de_response(st.substep)
        payload["state_updates"]["dp"]["data_cleaning_report"] = {"status": "ok"}
        return payload

    monkeypatch.setattr("maads.agents.run_json_task", fake_json_task)

    agent = DataEngineerAgent(artifact_dir=artifact_dir)
    from maads.state import Phase
    state.phase = Phase.DATA_PREPARATION
    state.substep = "3.2"
    delta = agent.act(state)
    assert not delta.failed
    assert calls == []  # execution_authoritative skips LLM
    assert state.dp.data_cleaning_report
    assert state.dp.data_cleaning_report.get("source", "").startswith("deterministic")
    assert not state.degraded_flags
