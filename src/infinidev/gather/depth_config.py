"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


@dataclass
class DepthConfig:
    """Controls how strictly the engine guides the model.

    Not just "how many phases" but "how much control":
    - minimal: no phases, single LoopEngine, model is free
    - light: force-read first, model creates own plan, light nudges
    - standard: full QUESTIONS→INVESTIGATE→PLAN→EXECUTE pipeline
    - deep: full pipeline + strict guardrails (mandatory tests, anti-rewrite, auto-revert)
    """
    # Phase control
    skip_questions: bool = False
    skip_investigate: bool = False
    questions_max: int = 6
    investigate_max_tool_calls: int = 12

    # Plan control
    plan_min_steps: int = 3
    plan_max_rounds: int = 5

    # Execute control
    replan_max_rounds: int = 3
    allow_only_add_steps: bool = True  # restrict next_steps to add-only

    # Guardrails (deep mode)
    reject_write_on_existing: bool = False   # force edit_method over write_file
    require_test_before_complete: bool = False  # reject step_complete without test
    auto_revert_on_regression: bool = False  # revert if test count drops
    aggressive_summarizer: bool = False  # summarize more frequently

    # Prompt style
    prompt_suffix: str = ""  # extra text appended to execute prompts


