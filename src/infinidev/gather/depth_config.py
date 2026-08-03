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
    - deep: full pipeline with a stricter execution prompt
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

    # Prompt style
    prompt_suffix: str = ""  # extra text appended to execute prompts

