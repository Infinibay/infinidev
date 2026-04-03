"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from infinidev.engine.loop.step_operation import StepOperation


class StepResult(BaseModel):
    """Parsed result from the LLM's step_complete tool call."""

    summary: str
    next_steps: list[StepOperation] = Field(default_factory=list)
    status: Literal["continue", "done", "blocked", "explore"] = "continue"
    final_answer: str | None = None


