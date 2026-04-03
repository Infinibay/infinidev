"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ActionRecord(BaseModel):
    """Structured summary of a completed step."""

    step_index: int
    summary: str
    tool_calls_count: int = 0
    files_to_preload: list[str] = Field(default_factory=list)
    changes_made: str = ""
    discovered_context: str = ""
    pending_items: str = ""
    anti_patterns: str = ""


