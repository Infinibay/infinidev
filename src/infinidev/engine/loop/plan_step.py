"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the agent's execution plan."""

    index: int
    title: str
    description: str = ""
    status: Literal["pending", "active", "done", "skipped"] = "pending"


