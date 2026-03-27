"""Data models for the information gathering phase."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TicketType(str, Enum):
    bug = "bug"
    feature = "feature"
    refactor = "refactor"
    sysadmin = "sysadmin"
    other = "other"


class DepthLevel(str, Enum):
    """How much analysis and control the engine applies."""
    minimal = "minimal"   # "Do it yourself" — single free run
    light = "light"       # "Think first" — force-read then free execute
    standard = "standard" # "Follow the process" — full phase pipeline
    deep = "deep"         # "I'm watching every move" — strict guardrails


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


DEPTH_CONFIGS: dict[DepthLevel, DepthConfig] = {
    DepthLevel.minimal: DepthConfig(
        skip_questions=True,
        skip_investigate=True,
        questions_max=0,
        investigate_max_tool_calls=0,
        plan_min_steps=1,
        plan_max_rounds=2,
        replan_max_rounds=1,
        allow_only_add_steps=False,  # model is free
        reject_write_on_existing=False,
        require_test_before_complete=False,
        auto_revert_on_regression=False,
        aggressive_summarizer=False,
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
        allow_only_add_steps=True,
        reject_write_on_existing=False,
        require_test_before_complete=False,
        auto_revert_on_regression=False,
        aggressive_summarizer=False,
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
        allow_only_add_steps=True,
        reject_write_on_existing=False,
        require_test_before_complete=False,
        auto_revert_on_regression=False,
        aggressive_summarizer=False,
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
        allow_only_add_steps=True,
        reject_write_on_existing=True,
        require_test_before_complete=True,
        auto_revert_on_regression=True,
        aggressive_summarizer=True,
        prompt_suffix=(
            "\nSTRICT RULES (deep mode):\n"
            "- You MUST run tests before calling step_complete\n"
            "- Do NOT use write_file on files that already exist — use edit_method or edit_file\n"
            "- Each step should change at most ONE method or function\n"
            "- If tests regress, STOP and rethink before proceeding"
        ),
    ),
}


@dataclass
class Question:
    """A single question to investigate about the codebase."""

    id: str
    question: str
    context_prompt: str  # Expanded prompt with {ticket_description} placeholder
    max_tool_calls: int = 15
    timeout_seconds: int = 120


class QuestionResult(BaseModel):
    """Result of answering one question."""

    question_id: str
    question_text: str
    answer: str
    tool_calls_used: int = 0
    phase: str = "fixed"  # "fixed" or "dynamic"


class ClassificationResult(BaseModel):
    """Result of ticket type + depth classification."""

    ticket_type: TicketType = TicketType.other
    reasoning: str = ""
    keywords: list[str] = Field(default_factory=list)
    depth: DepthLevel = DepthLevel.standard
    depth_reasoning: str = ""


class GatherBrief(BaseModel):
    """Complete brief from the gathering phase."""

    ticket_description: str = ""
    classification: ClassificationResult = Field(default_factory=ClassificationResult)
    fixed_answers: list[QuestionResult] = Field(default_factory=list)
    dynamic_answers: list[QuestionResult] = Field(default_factory=list)

    def render(self) -> str:
        """Render as XML text for injection into the developer system prompt."""
        parts = ["<gathered-context>"]

        # Classification
        parts.append(
            f'<classification type="{self.classification.ticket_type.value}">\n'
            f"Reasoning: {self.classification.reasoning}\n"
            f"Keywords: {', '.join(self.classification.keywords)}\n"
            f"</classification>"
        )

        # Investigation results
        all_answers = self.fixed_answers + self.dynamic_answers
        if all_answers:
            parts.append("<investigation>")
            for r in all_answers:
                # Truncate answer to ~500 tokens (~2000 chars)
                answer = r.answer[:2000]
                if len(r.answer) > 2000:
                    answer += "...[truncated]"
                parts.append(
                    f'<q id="{r.question_id}" phase="{r.phase}">\n'
                    f"Q: {r.question_text}\n"
                    f"A: {answer}\n"
                    f"</q>"
                )
            parts.append("</investigation>")

        parts.append("</gathered-context>")
        return "\n\n".join(parts)

    def summary(self) -> str:
        """Short summary for TUI display."""
        total = len(self.fixed_answers) + len(self.dynamic_answers)
        return (
            f"Gathered {total} context items "
            f"(type: {self.classification.ticket_type.value})"
        )
