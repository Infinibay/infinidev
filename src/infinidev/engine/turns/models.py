"""Conversation outcomes and explicitly declared work plans."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PlanItem(BaseModel):
    """One work item whose stable ID and status are declared by the agent."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str
    status: Literal["pending", "in_progress", "completed", "blocked"]


def validate_plan(plan: list[PlanItem]) -> list[PlanItem]:
    """Reject ambiguous identifiers or multiple simultaneously active items."""
    if len({item.id for item in plan}) != len(plan):
        raise ValueError("Plan item IDs must be unique")
    if sum(item.status == "in_progress" for item in plan) > 1:
        raise ValueError("At most one plan item may be in_progress")
    return plan


class EvidenceSource(BaseModel):
    """Describe which dispatched call produced an archived result."""

    model_config = ConfigDict(extra="forbid")

    tool: str = Field(min_length=1, max_length=100)
    requested_path: str | None = Field(default=None, max_length=200)
    file_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class PreviousInterruption(BaseModel):
    """Host-observed execution status of the previous interrupted turn."""

    status: Literal["cancelled", "exhausted", "failed"]
    dispatches: int = Field(ge=0, strict=True)
    stop_reason: Literal["repeated_read_evidence"] | None = None


class PreviousCheckpoint(BaseModel):
    """Saved dispatcher progress without an observed close."""

    closure: Literal["unclosed"]
    dispatches: int = Field(ge=0, strict=True)


class TestObservation(BaseModel):
    """A zero exit associated with a measured workspace revision."""

    revision: str = Field(strict=True, pattern=r"^[0-9a-f]{64}$")
    launch_cwd: str = Field(strict=True, min_length=1)
    locator: str | None = Field(default=None, strict=True, max_length=500)


class TurnState(BaseModel):
    """Retained context with transient conversational termination state."""

    plan: list[PlanItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    evidence_sources: dict[str, EvidenceSource] = Field(default_factory=dict, exclude=True)
    previous_interruption: PreviousInterruption | None = Field(default=None, exclude=True)
    previous_checkpoint: PreviousCheckpoint | None = Field(default=None, exclude=True)
    previous_run_id: str | None = Field(default=None, exclude=True)
    previous_test_observation: TestObservation | None = Field(default=None, exclude=True)
    finished: bool = Field(default=False, exclude=True)
    reply: str | None = Field(default=None, exclude=True)

    _validate_plan = field_validator("plan")(validate_plan)


class TurnResult(BaseModel):
    """A conversation outcome, independent of work Task lifecycle."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: str
    status: Literal["answered", "cancelled", "exhausted", "failed"]
    state: TurnState
    public_messages: list[str] = Field(default_factory=list)
    engine: Any | None = None
    metrics: dict[str, Annotated[int, Field(strict=True, ge=0)]] = Field(default_factory=dict)
