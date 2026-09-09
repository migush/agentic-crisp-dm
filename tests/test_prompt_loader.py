"""Tests for YAML/md prompt loader."""
from __future__ import annotations

import yaml

from maads.prompts import JSON_EXPECTED_OUTPUT, JSON_TERMINATION_INSTRUCTION
from maads.prompts.loader import load_agent_prompts, load_task_scaffolds, task_scaffold


def test_load_agent_prompts_all_agents():
    prompts = load_agent_prompts()
    assert set(prompts) == {
        "pm",
        "domain",
        "data_engineer",
        "data_scientist",
        "developer",
        "storyteller",
    }
    for name, persona in prompts.items():
        assert persona["role"]
        assert persona["goal"]
        assert len(persona["backstory"]) > 100, f"{name} backstory too short"


def test_domain_role_has_dataset_placeholder():
    domain = load_agent_prompts()["domain"]
    assert "{dataset_name}" in domain["role"]
    assert "{dataset_name}" in domain["goal"]


def test_task_scaffolds_have_placeholders():
    state_only = task_scaffold("state_only")
    assert "{state_view}" in state_only["description"]
    assert "{instruction}" in state_only["description"]

    substep = task_scaffold("substep_json")
    assert "{substep}" in substep["description"]
    assert "{schema_hint}" in substep["description"]

    all_kinds = load_task_scaffolds()
    assert set(all_kinds) >= {"state_only", "substep_json", "authored_code"}


def test_agents_yaml_tier_field_present():
    """tier is documented in agents.yaml for CrewAI config readability."""
    from importlib.resources import files

    spec = yaml.safe_load(
        files("maads").joinpath("config/agents.yaml").read_text(encoding="utf-8")
    )
    for name, fields in spec.items():
        assert fields.get("tier") in ("top", "mid"), f"{name} missing tier"


def test_json_termination_constants_exported():
    assert JSON_TERMINATION_INSTRUCTION
    assert JSON_EXPECTED_OUTPUT
    assert "begin with '{'" in JSON_TERMINATION_INSTRUCTION
    assert "end with '}'" in JSON_TERMINATION_INSTRUCTION


def test_json_task_scaffolds_put_termination_in_expected_output():
    """JSON termination lives once in expected_output, not duplicated in description."""
    for kind in ("state_only", "substep_json"):
        scaffold = task_scaffold(kind)
        if kind == "state_only":
            rendered = scaffold["description"].format(
                state_view="{}",
                instruction="test",
            )
        else:
            rendered = scaffold["description"].format(
                substep="1.1",
                substep_name="Test",
                instruction="test",
                state_view="{}",
                schema_hint="{}",
            )
        assert JSON_TERMINATION_INSTRUCTION not in rendered
        assert "Respond ONLY with JSON" in rendered
        assert "begin with '{'" in scaffold["expected_output"]
        assert "end with '}'" in scaffold["expected_output"]


def test_authored_code_scaffold_has_no_json_termination():
    scaffold = task_scaffold("authored_code")
    rendered = scaffold["description"].format(
        instruction="Write prepare.py",
        state_view='{"case_id": "titanic"}',
    )
    assert "Respond ONLY with JSON" not in rendered
    assert JSON_TERMINATION_INSTRUCTION not in rendered
    assert "Python code block" in scaffold["expected_output"]
