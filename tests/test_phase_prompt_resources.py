"""Tests for packaged JSON phase-prompt resources and compatibility exports."""

from __future__ import annotations

from importlib.resources import files
import json

import pytest

from infinidev.prompts import _phase_resources as phase_resources
from infinidev.prompts.phases import execute, investigate, plan, questions
from infinidev.prompts.phases import STRATEGIES


_RESOURCE_NAMES = {"shared.json", "bug.json", "feature.json", "refactor.json", "other.json"}


def _document(name: str) -> dict[str, object]:
    resource = files(phase_resources.PHASE_RESOURCE_PACKAGE).joinpath(name)
    document = json.loads(resource.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_phase_prompt_resources_are_complete_json_documents() -> None:
    resources = {
        resource.name
        for resource in files(phase_resources.PHASE_RESOURCE_PACKAGE).iterdir()
        if resource.name.endswith(".json")
    }

    assert resources == _RESOURCE_NAMES
    shared = _document("shared.json")
    assert set(shared) == {
        "schema_version",
        "investigate_rules",
        "execute_edit_contract",
        "planner_identity",
        "followup_prompt",
    }
    assert shared["schema_version"] == 1

    for task_type in phase_resources.PHASE_TASK_TYPES:
        document = _document(f"{task_type}.json")
        assert set(document) == {
            "schema_version",
            "task_type",
            "questions",
            "investigate",
            "plan",
            "execute",
        }
        assert document["schema_version"] == 1
        assert document["task_type"] == task_type
        assert set(document["questions"]) == {"prompt", "fallback"}
        for phase in ("investigate", "plan", "execute"):
            assert set(document[phase]) == {"prompt", "identity"}


@pytest.mark.parametrize(
    ("task_type", "prefix"),
    [
        ("bug", "BUG"),
        ("feature", "FEATURE"),
        ("refactor", "REFACTOR"),
        ("other", "OTHER"),
    ],
)
def test_compatibility_modules_export_loaded_phase_prompts(
    task_type: str,
    prefix: str,
) -> None:
    bundle = phase_resources.load_phase_prompt_bundle(task_type)

    assert getattr(questions, f"{prefix}_QUESTIONS") == bundle.questions_prompt
    assert getattr(questions, f"{prefix}_FALLBACK") == list(bundle.fallback_questions)
    assert getattr(investigate, f"{prefix}_INVESTIGATE") == bundle.investigate_prompt
    assert (
        getattr(investigate, f"{prefix}_INVESTIGATE_IDENTITY")
        == bundle.investigate_identity
    )
    assert getattr(plan, f"{prefix}_PLAN") == bundle.plan_prompt
    assert getattr(plan, f"{prefix}_PLAN_IDENTITY") == bundle.plan_identity
    assert getattr(execute, f"{prefix}_EXECUTE") == bundle.execute_prompt
    assert getattr(execute, f"{prefix}_EXECUTE_IDENTITY") == bundle.execute_identity


def test_shared_markers_expand_without_consuming_runtime_placeholders() -> None:
    shared = phase_resources.load_shared_phase_prompts()

    assert investigate._INVESTIGATE_RULES == shared.investigate_rules
    assert investigate.FOLLOWUP_PROMPT == shared.followup_prompt
    assert plan.PLANNER_IDENTITY == shared.planner_identity
    assert execute._EDIT_CONTRACT == shared.execute_edit_contract

    for task_type in phase_resources.PHASE_TASK_TYPES:
        bundle = phase_resources.load_phase_prompt_bundle(task_type)
        assert "<<INVESTIGATE_RULES>>" not in bundle.investigate_prompt
        assert "<<EXECUTE_EDIT_CONTRACT>>" not in bundle.execute_prompt
        assert bundle.investigate_prompt.count(shared.investigate_rules) == 1
        assert bundle.execute_prompt.count(shared.execute_edit_contract) == 1
        assert "{{q_num}}" in bundle.investigate_prompt
        assert "{{question}}" in bundle.investigate_prompt
        assert "{{step_num}}" in bundle.execute_prompt
        assert "{{step_files}}" in bundle.execute_prompt


def test_sysadmin_strategy_preserves_other_prompt_aliases() -> None:
    other = STRATEGIES["other"]
    sysadmin = STRATEGIES["sysadmin"]

    assert sysadmin.questions_prompt == other.questions_prompt
    assert sysadmin.investigate_prompt == other.investigate_prompt
    assert sysadmin.plan_prompt == other.plan_prompt
    assert sysadmin.execute_prompt == other.execute_prompt


def test_unknown_phase_prompt_task_type_fails_closed() -> None:
    with pytest.raises(ValueError, match="Unknown phase prompt task type"):
        phase_resources.load_phase_prompt_bundle("deployment")
