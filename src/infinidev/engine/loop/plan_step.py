"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the agent's execution plan."""

    index: int
    title: str
    explanation: str = ""
    # Self-defined success criterion: a short, verifiable statement the model
    # commits to before running the step. Used to render <expected-output> in
    # the iteration prompt and as the post-step verification anchor. Empty if
    # the model didn't declare one (older flows / quick steps).
    expected_output: str = ""
    # Long-form guidance written upfront by the planner: exact files, changes,
    # verification approach. Rendered ONLY while the step is active to keep
    # the iteration prompt small — pending steps show their title only.
    detail: str = ""
    # True when this step came from a user-approved plan (analyst-emitted,
    # displayed to the user in chat). LoopPlan.apply_operations refuses to
    # remove or modify approved steps when the LLM tries mid-execution.
    user_approved: bool = False
    status: Literal["pending", "active", "done", "skipped"] = "pending"


