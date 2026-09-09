"""Tests for maads.prompt_context consolidation utilities."""
from __future__ import annotations

from maads.text_normalize import (
    dedupe_nested_dict,
    dedupe_passages,
    normalize_passage_text,
    strip_markdown_headers,
)
from maads.prompt_context import compile_task_payload


def test_strip_markdown_headers_keep_first():
    text = "# Titanic — domain notes\n\nPassenger survival context."
    assert strip_markdown_headers(text) == "Passenger survival context."


def test_strip_markdown_headers_remove_all():
    text = "## Section\nBody text.\n### Sub\nMore."
    result = strip_markdown_headers(text, keep_first=False)
    assert "Body text." in result
    assert "More." in result
    assert "##" not in result


def test_normalize_passage_text_collapses_whitespace():
    assert normalize_passage_text("  Foo   bar  ") == "foo bar"


def test_dedupe_passages_ignores_source_prefix():
    passages = [
        "[a.md] Titanic survival uses passenger features.",
        "[b.md] Titanic survival uses passenger features.",
        "[a.md] Unique content here.",
    ]
    deduped = dedupe_passages(passages)
    assert len(deduped) == 2
    assert deduped[0] == passages[0]
    assert deduped[1] == passages[2]


def test_dedupe_nested_dict_drops_equal_overlay_keys():
    base = {"du_so_far": {"report": 1}, "case_id": "titanic"}
    overlay = {"du_so_far": {"report": 1}, "extra": "value"}
    assert dedupe_nested_dict(base, overlay) == {"extra": "value"}


def test_dedupe_nested_dict_keeps_different_values():
    base = {"du_so_far": {"report": 1}}
    overlay = {"du_so_far": {"report": 2}}
    assert dedupe_nested_dict(base, overlay) == {"du_so_far": {"report": 2}}


def test_compile_task_payload_includes_state_view_once():
    payload = compile_task_payload(
        agent_name="data_engineer",
        instruction="Do the work.",
        state_view={"case_id": "titanic", "substep": "2.2"},
        schema_hint='{"ok": true}',
        template_kind="substep_json",
        substep="2.2",
    )
    assert payload.count("Relevant state (JSON):") == 1
    assert '"case_id": "titanic"' in payload
    assert "Do the work." in payload


def test_compile_task_payload_state_only():
    payload = compile_task_payload(
        agent_name="pm",
        instruction="Decide next.",
        state_view={"phase": 1},
        template_kind="state_only",
    )
    assert "Current CRISP-DM state (JSON):" in payload
    assert "Decide next." in payload


def test_compile_task_payload_authored_code_skips_json_scaffold():
    payload = compile_task_payload(
        agent_name="data_engineer",
        instruction="Write cleaning code.",
        state_view={"case_id": "titanic", "substep": "3.2"},
        template_kind="authored_code",
        substep="3.2",
    )
    assert "Write cleaning code." in payload
    assert "Relevant state (JSON):" in payload
    assert "Respond ONLY with JSON" not in payload
    assert "schema_hint" not in payload
    assert "Python code block" in payload or "Do not wrap the code in JSON" in payload
