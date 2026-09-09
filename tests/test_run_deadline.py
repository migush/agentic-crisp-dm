"""Wall-clock deadline: halt cleanly before a parent process kill."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from maads.artifact_paths import ensure_run_layout
from maads.config import load_case_config
from maads.flow.crisp_dm_flow import CrispDMFlow
from maads.flow.phase_runner import check_global_halt
from maads.paths import resolve_path
from maads.reports.writer import write_run_reports
from maads.run_deadline import (
    HALT_REASON,
    deadline_exceeded,
    reset_deadline_clock,
    start_deadline_clock,
)
from maads.state import CrispDMState
from maads.testing.flow_harness import make_run_context_stub
from maads.testing.fake_llm import fake_llm_response


@pytest.fixture(autouse=True)
def _reset_deadline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MAADS_RUN_DEADLINE_SEC", raising=False)
    reset_deadline_clock()
    yield
    reset_deadline_clock()


def test_deadline_disabled_by_default() -> None:
    start_deadline_clock()
    assert deadline_exceeded() is False


def test_deadline_exceeded_after_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAADS_RUN_DEADLINE_SEC", "10")
    times = iter([100.0, 111.0])
    monkeypatch.setattr("maads.run_deadline.time.monotonic", lambda: next(times))
    start_deadline_clock()
    assert deadline_exceeded() is True


def test_check_global_halt_returns_deadline_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    state = CrispDMState.from_config(cfg)
    ctx = make_run_context_stub(state, tmp_path)
    monkeypatch.setenv("MAADS_RUN_DEADLINE_SEC", "1")
    times = iter([0.0, 5.0])
    monkeypatch.setattr("maads.run_deadline.time.monotonic", lambda: next(times))
    start_deadline_clock()
    assert check_global_halt(ctx) == HALT_REASON


@patch("maads.agents.run_json_task")
def test_flow_halts_on_deadline_and_writes_reports(
    mock_llm, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    mock_llm.side_effect = fake_llm_response
    monkeypatch.setenv("MAADS_RUN_DEADLINE_SEC", "1")
    monkeypatch.setenv("MAADS_TRACE", "0")
    monkeypatch.setenv("MAADS_PROGRESS", "0")
    monkeypatch.setenv("MAADS_REPORTS", "1")
    monkeypatch.setattr("maads.flow.phase_runner.deadline_exceeded", lambda: True)

    cfg = load_case_config(resolve_path("configs/titanic.yaml"))
    state = CrispDMState.from_config(cfg)
    artifact_dir = tmp_path / "artifacts" / "titanic"
    ensure_run_layout(artifact_dir, run_id="deadline-test", case_id="titanic")
    state = CrispDMFlow(state, artifact_dir).run()

    assert state.halted
    assert state.halt_reason == HALT_REASON
    write_run_reports(state, artifact_dir, force=True)
    reports = artifact_dir / "reports"
    assert (reports / "postmortem.json").is_file()
    assert (reports / "improvement_bundle.json").is_file()
