"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class StepOperation(BaseModel):
    """A structured operation to apply to the plan."""

    op: Literal["add", "modify", "remove"]
    index: int
    description: str = ""  # required for add/modify, ignored for remove


