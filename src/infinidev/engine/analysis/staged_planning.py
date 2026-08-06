"""Durable domain artefacts for adaptive Goal/Stage/Task planning."""

from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class GoalSpec(BaseModel):
    """Immutable user-owned objective followed across one or more Stages."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: _new_id("goal"))
    title: str
    user_request: str
    understanding: str = ""
    acceptance_criteria: list[str] = Field(default_factory=list)
    derived_verification_criteria: list[str] = Field(default_factory=list)
    out_of_scope: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    planning_context: str = ""


class StageTaskSpec(BaseModel):
    """One independently checkable deliverable inside a Stage."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1)
    acceptance_criteria: list[str] = Field(..., min_length=1)
    depends_on: list[str] = Field(default_factory=list)

    @property
    def derived_verification_criteria(self) -> list[str]:
        """Expose the provenance-explicit name used by execution and review."""
        return list(self.acceptance_criteria)


class StageSpec(BaseModel):
    """The next evidence-grounded planning horizon."""

    model_config = ConfigDict(frozen=True)

    title: str = Field(..., min_length=1)
    outcome: str = Field(..., min_length=1)
    exit_criteria: list[str] = Field(..., min_length=1)
    tasks: list[StageTaskSpec] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _validate_task_graph(self) -> "StageSpec":
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

    def fingerprint(self) -> str:
        """Stable identity used to notice an unchanged proposed route."""
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        return hashlib.sha256(encoded).hexdigest()


class EmitStageDecision(BaseModel):
    kind: Literal["stage"] = "stage"
    stage: StageSpec


class CompleteGoalDecision(BaseModel):
    kind: Literal["goal_complete"] = "goal_complete"
    evidence: list[str] = Field(..., min_length=1)


class BlockGoalDecision(BaseModel):
    kind: Literal["goal_blocked"] = "goal_blocked"
    reason: str = Field(..., min_length=1)
    missing: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)


StageDecision = EmitStageDecision | CompleteGoalDecision | BlockGoalDecision


TaskExecutionStatus = Literal[
    "pending", "active", "completed", "blocked", "failed", "cancelled"
]
StageExecutionStatus = Literal[
    "planned", "active", "evaluating", "completed", "blocked", "failed", "cancelled"
]
GoalExecutionStatus = Literal["active", "complete", "blocked", "cancelled", "failed"]


class EvidenceEntry(BaseModel):
    """Observed output retained for later Stage and Goal decisions."""

    id: str = Field(default_factory=lambda: _new_id("evidence"))
    kind: str
    summary: str
    stage_id: str | None = None
    task_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        normalized = " ".join(self.summary.lower().split())
        return hashlib.sha256(f"{self.kind}\0{normalized}".encode()).hexdigest()


class TaskExecutionRecord(BaseModel):
    spec: StageTaskSpec
    status: TaskExecutionStatus = "pending"
    plan: dict[str, Any] | None = None
    result: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    error: str = ""
    attempts: int = 0


class StageExecutionRecord(BaseModel):
    id: str = Field(default_factory=lambda: _new_id("stage"))
    number: int
    spec: StageSpec
    status: StageExecutionStatus = "planned"
    tasks: list[TaskExecutionRecord] = Field(default_factory=list)
    evidence_before: list[str] = Field(default_factory=list)
    evidence_after: list[str] = Field(default_factory=list)
    outcome_summary: str = ""

    @model_validator(mode="after")
    def _seed_tasks(self) -> "StageExecutionRecord":
        if not self.tasks:
            self.tasks = [TaskExecutionRecord(spec=task) for task in self.spec.tasks]
        return self


class GoalTerminalState(BaseModel):
    kind: Literal["goal_complete", "goal_blocked", "cancelled", "failed"]
    summary: str
    evidence: list[str] = Field(default_factory=list)
    missing: str = ""


class StagedPlanningState(BaseModel):
    """Serializable state needed to resume between any two planning actions."""

    version: int = 1
    goal: GoalSpec
    status: GoalExecutionStatus = "active"
    stages: list[StageExecutionRecord] = Field(default_factory=list)
    evidence: list[EvidenceEntry] = Field(default_factory=list)
    guidance: list[str] = Field(default_factory=list)
    terminal: GoalTerminalState | None = None
    revision: int = 0

    @property
    def active_stage(self) -> StageExecutionRecord | None:
        for stage in reversed(self.stages):
            if stage.status in {"planned", "active", "evaluating"}:
                return stage
        return None

    def add_stage(self, spec: StageSpec) -> StageExecutionRecord:
        record = StageExecutionRecord(
            number=len(self.stages) + 1,
            spec=spec,
            evidence_before=[entry.id for entry in self.evidence],
        )
        self.stages.append(record)
        self.status = "active"
        self.terminal = None
        self.revision += 1
        return record

    def add_evidence(self, entry: EvidenceEntry) -> bool:
        """Add a genuinely new observation and return whether it changed the ledger."""
        fingerprints = {existing.fingerprint() for existing in self.evidence}
        if entry.fingerprint() in fingerprints:
            return False
        self.evidence.append(entry)
        self.revision += 1
        return True

    def snapshot(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class TaskPlanningHandoff(BaseModel):
    """Structured input from one active Stage Task to the Task Planner."""

    goal: GoalSpec
    stage_id: str
    stage_number: int
    stage: StageSpec
    task: StageTaskSpec
    dependency_results: dict[str, str] = Field(default_factory=dict)
    evidence: list[EvidenceEntry] = Field(default_factory=list)

    def render(self, exploration_budget: int) -> str:
        payload = {
            "goal": self.goal.model_dump(mode="json"),
            "stage_id": self.stage_id,
            "stage_number": self.stage_number,
            "stage": self.stage.model_dump(mode="json"),
            "task": self.task.model_dump(mode="json"),
            "dependency_results": {
                key: value[:2000] for key, value in self.dependency_results.items()
            },
            "evidence": [
                {
                    "id": entry.id,
                    "kind": entry.kind,
                    "summary": entry.summary[:2000],
                    "stage_id": entry.stage_id,
                    "task_id": entry.task_id,
                }
                for entry in self.evidence
            ],
        }
        return (
            "STAGED TASK HANDOFF\n"
            "Authority labels: goal.user_request is USER_LITERAL; Stage and Task "
            "checks are DERIVED; evidence is OBSERVED_EVIDENCE.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
            f"At most {exploration_budget} exploration calls, then call "
            "emit_task_plan."
        )


def plan_snapshot(plan: Any) -> dict[str, Any]:
    """Serialize the legacy Task Plan without coupling persistence to dataclasses."""
    steps: list[dict[str, Any]] = []
    for step in getattr(plan, "steps", []) or []:
        verify = getattr(step, "verify", None)
        if hasattr(verify, "model_dump"):
            verify = verify.model_dump(mode="json")
        steps.append({
            "title": getattr(step, "title", ""),
            "detail": getattr(step, "detail", ""),
            "expected_output": getattr(step, "expected_output", ""),
            "verify": verify,
            "authority": getattr(step, "authority", "model_inferred"),
        })
    return {
        "overview": getattr(plan, "overview", ""),
        "steps": steps,
        "derived_verification_criteria": list(
            getattr(plan, "acceptance_criteria", []) or []
        ),
    }


__all__ = [
    "BlockGoalDecision",
    "CompleteGoalDecision",
    "EmitStageDecision",
    "EvidenceEntry",
    "GoalSpec",
    "GoalTerminalState",
    "StageDecision",
    "StageExecutionRecord",
    "StageSpec",
    "StageTaskSpec",
    "StagedPlanningState",
    "TaskExecutionRecord",
    "TaskPlanningHandoff",
    "plan_snapshot",
]
