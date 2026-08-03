"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


from infinidev.gather.ticket_type import TicketType
from infinidev.gather.depth_level import DepthLevel
from infinidev.gather.depth_config import DepthConfig
from infinidev.gather.question import Question
from infinidev.gather.question_result import QuestionResult
from infinidev.gather.classification_result import ClassificationResult
from infinidev.gather.gather_brief import GatherBrief

DEPTH_CONFIGS: dict[DepthLevel, DepthConfig] = {
    DepthLevel.minimal: DepthConfig(
        skip_questions=True,
        skip_investigate=True,
        questions_max=0,
        investigate_max_tool_calls=0,
        plan_min_steps=1,
        plan_max_rounds=2,
        replan_max_rounds=1,
        prompt_suffix="",
    ),
    DepthLevel.light: DepthConfig(
        skip_questions=True,  # no separate question phase
        skip_investigate=True,  # no separate investigate phase
        questions_max=0,
        investigate_max_tool_calls=0,
        plan_min_steps=2,
        plan_max_rounds=3,
        replan_max_rounds=2,
        prompt_suffix="\nReminder: verify your changes work before calling step_complete.",
    ),
    DepthLevel.standard: DepthConfig(
        skip_questions=False,
        skip_investigate=False,
        questions_max=6,
        investigate_max_tool_calls=12,
        plan_min_steps=3,
        plan_max_rounds=5,
        replan_max_rounds=3,
        prompt_suffix="",
    ),
    DepthLevel.deep: DepthConfig(
        skip_questions=False,
        skip_investigate=False,
        questions_max=10,
        investigate_max_tool_calls=20,
        plan_min_steps=5,
        plan_max_rounds=5,
        replan_max_rounds=3,
        prompt_suffix=(
            "\nSTRICT RULES (deep mode):\n"
            "- You MUST run tests before calling step_complete\n"
            "- Use edit_file for existing files and create_file only for new files\n"
            "- Each step should change at most ONE method or function\n"
            "- If tests regress, STOP and rethink before proceeding"
        ),
    ),
}
