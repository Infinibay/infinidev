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
    # Output of the user's ``step_end_summary`` hook, if one is configured.
    # Held on the record rather than folded into ``summary`` precisely so
    # the summariser cannot rewrite or drop it: the record is what the next
    # iteration's prompt is rebuilt from, so this is the one field a user
    # can put text into and know it will still be there ten steps later.
    hook_notes: str = ""
    behavior_score: int = 0
    behavior_good: list[str] = Field(default_factory=list)
    behavior_bad: list[str] = Field(default_factory=list)
    # Deterministic evidence snapshots used by code-controlled progress
    # detection. These are deliberately separate from LLM-written prose.
    successful_edit_count: int = 0
    net_workspace_changed: bool = False
    test_outcome_fingerprints: tuple[str, ...] = ()
