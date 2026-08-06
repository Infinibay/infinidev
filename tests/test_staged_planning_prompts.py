"""Contracts for the Stage and Task Planner prompt surfaces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from infinidev.prompts.analyst.planner_prompt import ANALYST_PLANNER_SYSTEM_PROMPT
from infinidev.prompts.analyst.stage_planner_prompt import STAGE_PLANNER_SYSTEM_PROMPT
from infinidev.prompts.analyst.task_planner_prompt import TASK_PLANNER_SYSTEM_PROMPT
from infinidev.tools import get_tools_for_role
from infinidev.tools.planner.emit_plan_tool import EmitTaskPlanInput
from infinidev.tools.planner.stage_decision_tools import EmitStageInput


def _tool_names(role: str) -> set[str]:
    return {
        tool.name
        for tool in get_tools_for_role(role, supports_vision=False)
    }


def test_planner_roles_expose_only_their_terminal_decisions() -> None:
    task_tools = _tool_names("task_planner")
    stage_tools = _tool_names("stage_planner")

    assert "emit_task_plan" in task_tools
    assert not ({"emit_stage", "complete_goal", "block_goal"} & task_tools)

    assert {"emit_stage", "complete_goal", "block_goal"} <= stage_tools
    assert "emit_task_plan" not in stage_tools


def test_legacy_prompt_import_now_names_the_task_planner_contract() -> None:
    assert ANALYST_PLANNER_SYSTEM_PROMPT == TASK_PLANNER_SYSTEM_PROMPT
    assert "emit_task_plan" in ANALYST_PLANNER_SYSTEM_PROMPT
    assert "emit_plan``" not in ANALYST_PLANNER_SYSTEM_PROMPT


def test_prompts_define_methods_without_arbitrary_size_thresholds() -> None:
    combined = STAGE_PLANNER_SYSTEM_PROMPT + TASK_PLANNER_SYSTEM_PROMPT

    assert "1-5" not in combined
    assert "seven steps" not in combined.lower()
    assert "two stages" not in combined.lower()
    assert "Step count follows those evidence boundaries" in combined
    assert "Do not create ceremonial Tasks to fill a count" in combined


def test_examples_cannot_be_read_as_task_evidence() -> None:
    assert "not evidence for another Goal" in STAGE_PLANNER_SYSTEM_PROMPT
    assert "not evidence for another Task" in TASK_PLANNER_SYSTEM_PROMPT


def test_task_plan_schema_labels_planner_checks_as_derived() -> None:
    plan = EmitTaskPlanInput(
        overview="Measure and repair the observed behavior.",
        derived_verification_criteria=["The observed failing behavior no longer occurs"],
        steps=[{"title": "Repair the observed behavior"}],
    )

    assert plan.derived_verification_criteria == [
        "The observed failing behavior no longer occurs"
    ]
    assert "acceptance_criteria" not in EmitTaskPlanInput.model_fields


def test_stage_schema_accepts_dependency_flow() -> None:
    stage = EmitStageInput(
        title="Measure before choosing the change",
        outcome="The next change can be selected from measured evidence",
        exit_criteria=["A measured cause is named"],
        tasks=[
            {
                "id": "baseline",
                "title": "Record the baseline",
                "outcome": "The current behavior is measured",
                "acceptance_criteria": ["The measurement is recorded"],
            },
            {
                "id": "diagnose",
                "title": "Locate the measured cause",
                "outcome": "A cause is supported by the baseline",
                "acceptance_criteria": ["The cause is tied to an observation"],
                "depends_on": ["baseline"],
            },
        ],
    )

    assert stage.tasks[1].depends_on == ["baseline"]


def test_stage_schema_rejects_unknown_dependency() -> None:
    with pytest.raises(ValidationError, match="unknown ids"):
        EmitStageInput(
            title="Use invalid dependency",
            outcome="This artifact should fail validation",
            exit_criteria=["The invalid edge is rejected"],
            tasks=[
                {
                    "id": "diagnose",
                    "title": "Locate the cause",
                    "outcome": "A cause is named",
                    "acceptance_criteria": ["The cause has evidence"],
                    "depends_on": ["missing"],
                }
            ],
        )


def test_stage_schema_rejects_dependency_cycle() -> None:
    with pytest.raises(ValidationError, match="must not contain a cycle"):
        EmitStageInput(
            title="Reject a cyclic task graph",
            outcome="Only dependency-ready Tasks can enter execution",
            exit_criteria=["The cycle is rejected"],
            tasks=[
                {
                    "id": "first",
                    "title": "Produce the first result",
                    "outcome": "The first result exists",
                    "acceptance_criteria": ["The first result is observed"],
                    "depends_on": ["second"],
                },
                {
                    "id": "second",
                    "title": "Produce the second result",
                    "outcome": "The second result exists",
                    "acceptance_criteria": ["The second result is observed"],
                    "depends_on": ["first"],
                },
            ],
        )
