"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from infinidev.engine.loop.step_operation import StepOperation


class StepResult(BaseModel):
    """Parsed result from the LLM's step_complete tool call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    next_steps: list[StepOperation] = Field(default_factory=list)
    status: Literal["continue", "done", "blocked", "explore"] = "continue"
    final_answer: str | None = None

    # Post-processing metadata (set by _run_inner_loop, consumed by step_manager)
    action_tool_calls: int = 0
    behavior_tracker: Any = Field(default=None, exclude=True)


