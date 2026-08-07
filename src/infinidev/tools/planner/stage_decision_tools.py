"""Schema-level terminal tools for the Stage Planner."""

from __future__ import annotations

import json
from typing import Literal, Type

from pydantic import BaseModel, Field, model_validator

from infinidev.tools.base.base_tool import InfinibayBaseTool


class StageTaskArg(BaseModel):
    """One independently checkable contribution to a Stage."""

    id: str = Field(
        ...,
        min_length=1,
        description="Stable identifier used by dependency edges inside this Stage.",
    )
    title: str = Field(..., description="One-line name displayed for the Task.")
    outcome: str = Field(
        ...,
        description="State this Task produces without prescribing its execution Steps.",
    )
    acceptance_criteria: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Planner-derived observations that decide whether the Task outcome "
            "was reached. They cannot expand the Goal or Stage."
        ),
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of Tasks whose outputs this Task consumes. Leave empty when "
            "the relationship is ordering without data or evidence flow."
        ),
    )


class EmitStageInput(BaseModel):
    title: str = Field(..., description="One-line name displayed for the Stage.")
    outcome: str = Field(
        ...,
        description="Result this Stage produces before the next Stage decision.",
    )
    exit_criteria: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "Observed conditions that decide whether the Stage outcome was "
            "reached. These are derived checks, not user requirements."
        ),
    )
    purpose: Literal["discovery", "delivery"] = Field(
        "delivery",
        description=(
            "discovery resolves one named uncertainty before delivery; delivery "
            "advances the requested result. A discovery Stage has exactly one Task."
        ),
    )
    tasks: list[StageTaskArg] = Field(
        ...,
        min_length=1,
        description="Tasks whose combined outcomes establish the Stage exit criteria.",
    )

    @model_validator(mode="after")
    def _validate_task_graph(self) -> "EmitStageInput":
        if self.purpose == "discovery" and len(self.tasks) != 1:
            raise ValueError(
                "A discovery Stage must contain exactly one focused Task."
            )
        ids = [task.id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("Stage Task ids must be unique")
        known = set(ids)
        for task in self.tasks:
            missing = set(task.depends_on) - known
            if missing:
                raise ValueError(
                    f"Task {task.id!r} depends on unknown ids: {sorted(missing)}"
                )
            if task.id in task.depends_on:
                raise ValueError(f"Task {task.id!r} cannot depend on itself")

        remaining = {task.id: set(task.depends_on) for task in self.tasks}
        while remaining:
            ready = {task_id for task_id, deps in remaining.items() if not deps}
            if not ready:
                raise ValueError("Stage Task dependencies must not contain a cycle")
            remaining = {
                task_id: deps - ready
                for task_id, deps in remaining.items()
                if task_id not in ready
            }
        return self


class CompleteGoalInput(BaseModel):
    evidence: list[str] = Field(
        ...,
        min_length=1,
        description=(
            "One entry per Goal acceptance condition naming the observation "
            "that establishes it and citing that observation's evidence-ledger ID."
        ),
    )


class BlockGoalInput(BaseModel):
    reason: str = Field(
        ...,
        description=(
            "Why no in-scope Stage can produce new evidence from the current state."
        ),
    )
    missing: str = Field(
        ...,
        description="User decision, authority or external state that would unblock the Goal.",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Observed facts that establish the obstacle.",
    )


class EmitStageTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "emit_stage"
    description: str = (
        "Emit the next Stage and end the Stage Planner turn. Use this when the "
        "Goal is not complete and an in-scope Stage can produce new evidence "
        "or advance a Goal acceptance condition."
    )
    args_schema: Type[BaseModel] = EmitStageInput

    def _run(
        self,
        title: str,
        outcome: str,
        exit_criteria: list,
        tasks: list,
        purpose: str = "delivery",
    ) -> str:
        return _json_result(
            "stage",
            title=title,
            outcome=outcome,
            exit_criteria=exit_criteria,
            tasks=tasks,
            purpose=purpose,
        )


class CompleteGoalTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "complete_goal"
    description: str = (
        "Declare the Goal complete and end the Stage Planner turn. Use this "
        "only when each Goal acceptance condition has named evidence and no "
        "observed contradiction remains unresolved. Cite evidence-ledger IDs."
    )
    args_schema: Type[BaseModel] = CompleteGoalInput

    def _run(self, evidence: list) -> str:
        return _json_result("goal_complete", evidence=evidence)


class BlockGoalTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "block_goal"
    description: str = (
        "Report that no in-scope Stage can produce new evidence and end the "
        "Stage Planner turn. Name the user decision, missing authority or "
        "external state that would allow planning to continue."
    )
    args_schema: Type[BaseModel] = BlockGoalInput

    def _run(self, reason: str, missing: str, evidence: list | None = None) -> str:
        return _json_result(
            "goal_blocked",
            reason=reason,
            missing=missing,
            evidence=evidence or [],
        )


def _json_result(kind: str, **payload: object) -> str:
    return json.dumps(
        {"kind": kind, **payload},
        default=lambda value: (
            value.model_dump() if hasattr(value, "model_dump") else str(value)
        ),
    )
