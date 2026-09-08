"""Smoke test for write-only LLM param experience ledger."""
from __future__ import annotations

import json
from pathlib import Path

from maads.config import load_case_config
from maads.experience_ledger import append_llm_param_experience
from maads.paths import resolve_path
from maads.state import CrispDMState


def test_append_llm_param_experience(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MAADS_SKIP_MODEL_PROBE", "1")
    monkeypatch.setenv("MODEL", "gpt-6-astra")
    monkeypatch.delenv("REASONING_EFFORT", raising=False)
    state = CrispDMState.from_config(load_case_config(resolve_path("configs/titanic.yaml")))
    run_dir = tmp_path / "runs" / "abc"
    run_dir.mkdir(parents=True)
    case_dir = tmp_path / "case"
    path = append_llm_param_experience(state, run_dir, case_dir, run_id="abc")
    assert path.is_file()
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["run_id"] == "abc"
    assert row["case_id"] == "titanic"
    assert row["reasoning_effort_by_agent"]["pm"] == "medium"
    assert row["reasoning_effort_by_agent"]["developer"] == "high"
    ledger = case_dir / "llm_param_ledger.jsonl"
    assert ledger.is_file()
    assert "abc" in ledger.read_text(encoding="utf-8")
