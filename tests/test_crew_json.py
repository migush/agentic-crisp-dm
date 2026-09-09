"""Tests for LLM JSON response cleanup and parsing."""

from maads.crew import _extract_json

_PM_SAMPLE = {
    "action": "advance",
    "target_substep": None,
    "loop_label": None,
    "loop_to_phase": None,
    "reason": (
        "The project is in the initial phase; substep 1.1 must be executed "
        "to establish business objectives."
    ),
}


def test_extract_json_plain_object() -> None:
    raw = (
        '{"action": "advance", "target_substep": null, '
        '"loop_label": null, "loop_to_phase": null, "reason": "ok"}'
    )
    assert _extract_json(raw) == {
        "action": "advance",
        "target_substep": None,
        "loop_label": None,
        "loop_to_phase": None,
        "reason": "ok",
    }


def test_extract_json_markdown_fences_and_horizontal_rules() -> None:
    raw = """---
```json
{
  "action": "advance",
  "target_substep": null,
  "loop_label": null,
  "loop_to_phase": null,
  "reason": "The project is in the initial phase; substep 1.1 must be executed to establish business objectives."
}
```
---"""
    assert _extract_json(raw) == _PM_SAMPLE


def test_extract_json_trailing_commas() -> None:
    raw = """```json
{
  "action": "advance",
  "target_substep": null,
  "reason": "ok",
}
```"""
    assert _extract_json(raw) == {
        "action": "advance",
        "target_substep": None,
        "reason": "ok",
    }


def test_extract_json_prose_before_object() -> None:
    raw = (
        "Here is the directive:\n"
        '{"action": "halt", "target_substep": null, "reason": "done"}'
    )
    assert _extract_json(raw) == {
        "action": "halt",
        "target_substep": None,
        "reason": "done",
    }


def test_extract_json_invalid_returns_none() -> None:
    assert _extract_json("not json at all") is None
    assert _extract_json("") is None


def test_extract_json_extra_wrapping_braces() -> None:
    """LLMs sometimes wrap a valid object in an extra brace pair."""
    raw = '{{"action": "advance", "reason": "ok"}}'
    assert _extract_json(raw) == {"action": "advance", "reason": "ok"}
