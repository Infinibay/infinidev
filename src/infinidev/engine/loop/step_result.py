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
    # The model's free-text claim of how it verified the step (from the
    # step_complete schema). Captured for logging/diagnostics and as a hint
    # of WHICH check it ran — it is never the source of truth for pass/fail
    # (that is the executed StepVerification). Previously parsed-and-discarded.
    evidence_summary: str = ""

    # Post-processing metadata (set by _run_inner_loop, consumed by step_manager)
    action_tool_calls: int = 0
    behavior_tracker: Any = Field(default=None, exclude=True)
    # Whether the model emitted ANY function call this step, pseudo-tools
    # included. ``action_tool_calls`` counts only executed regular tools, so
    # it measures budget, not liveness: a step closed with think +
    # step_complete has zero of them while being perfectly well-behaved.
    # The abort for "the model cannot produce function calls" reads this.
    saw_tool_calls: bool = False


