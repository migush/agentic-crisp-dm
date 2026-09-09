"""Per-kickoff token accounting uses prompt+completion, not cumulative totals."""
from __future__ import annotations

from types import SimpleNamespace

from maads.crew import extract_kickoff_token_usage


def test_prefers_prompt_plus_completion_over_cumulative_total() -> None:
    usage = SimpleNamespace(prompt_tokens=80, completion_tokens=20, total_tokens=500)
    n_tokens, n_in, n_out = extract_kickoff_token_usage(usage)
    assert n_tokens == 100
    assert n_in == 80
    assert n_out == 20


def test_falls_back_to_total_when_prompt_completion_missing() -> None:
    usage = SimpleNamespace(total_tokens=42)
    n_tokens, n_in, n_out = extract_kickoff_token_usage(usage)
    assert n_tokens == 42
    assert n_in is None
    assert n_out is None


def test_none_usage_returns_nones() -> None:
    assert extract_kickoff_token_usage(None) == (None, None, None)
