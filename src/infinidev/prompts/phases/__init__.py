"""Phase strategy registry — QUESTIONS → INVESTIGATE → PLAN → EXECUTE.

Provides PhaseStrategy configs per task type with prompts, identities,
and limits for each phase.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from infinidev.prompts.phases.questions import (
    BUG_QUESTIONS, BUG_FALLBACK,
    FEATURE_QUESTIONS, FEATURE_FALLBACK,
    REFACTOR_QUESTIONS, REFACTOR_FALLBACK,
    OTHER_QUESTIONS, OTHER_FALLBACK,
)
from infinidev.prompts.phases.investigate import (
    BUG_INVESTIGATE, BUG_INVESTIGATE_IDENTITY,
    FEATURE_INVESTIGATE, FEATURE_INVESTIGATE_IDENTITY,
    REFACTOR_INVESTIGATE, REFACTOR_INVESTIGATE_IDENTITY,
    OTHER_INVESTIGATE, OTHER_INVESTIGATE_IDENTITY,
)
from infinidev.prompts.phases.plan import (
    PLANNER_IDENTITY,
    BUG_PLAN, BUG_PLAN_IDENTITY,
    FEATURE_PLAN, FEATURE_PLAN_IDENTITY,
    REFACTOR_PLAN, REFACTOR_PLAN_IDENTITY,
    OTHER_PLAN, OTHER_PLAN_IDENTITY,
)
from infinidev.prompts.phases.execute import (
    BUG_EXECUTE, BUG_EXECUTE_IDENTITY,
    FEATURE_EXECUTE, FEATURE_EXECUTE_IDENTITY,
    REFACTOR_EXECUTE, REFACTOR_EXECUTE_IDENTITY,
    OTHER_EXECUTE, OTHER_EXECUTE_IDENTITY,
)


@dataclass
class PhaseStrategy:
    """Configuration for how to run each phase for a given task type."""
    questions_prompt: str
    investigate_prompt: str
    plan_prompt: str
    execute_prompt: str
    investigate_identity: str = ""
    plan_identity: str = ""
    execute_identity: str = ""
    fallback_questions: list[str] = field(default_factory=list)
    questions_min: int = 2
    questions_max: int = 10
    investigate_max_tool_calls: int = 12
    plan_min_steps: int = 3
    plan_max_step_files: int = 2
    execute_max_tool_calls_per_step: int = 15
    auto_test: bool = True
    anti_rewrite: bool = False


STRATEGIES: dict[str, PhaseStrategy] = {
    "bug": PhaseStrategy(
        questions_prompt=BUG_QUESTIONS,
        investigate_prompt=BUG_INVESTIGATE,
        plan_prompt=BUG_PLAN,
        execute_prompt=BUG_EXECUTE,
        investigate_identity=BUG_INVESTIGATE_IDENTITY,
        plan_identity=BUG_PLAN_IDENTITY,
        execute_identity=BUG_EXECUTE_IDENTITY,
        fallback_questions=BUG_FALLBACK,
        questions_min=2,
        questions_max=6,
        investigate_max_tool_calls=12,
        plan_min_steps=2,
        execute_max_tool_calls_per_step=12,
        auto_test=True,
    ),
    "feature": PhaseStrategy(
        questions_prompt=FEATURE_QUESTIONS,
        investigate_prompt=FEATURE_INVESTIGATE,
        plan_prompt=FEATURE_PLAN,
        execute_prompt=FEATURE_EXECUTE,
        investigate_identity=FEATURE_INVESTIGATE_IDENTITY,
        plan_identity=FEATURE_PLAN_IDENTITY,
        execute_identity=FEATURE_EXECUTE_IDENTITY,
        fallback_questions=FEATURE_FALLBACK,
        questions_min=3,
        questions_max=10,
        investigate_max_tool_calls=12,
        plan_min_steps=4,
        execute_max_tool_calls_per_step=15,
        auto_test=True,
        anti_rewrite=True,
    ),
    "refactor": PhaseStrategy(
        questions_prompt=REFACTOR_QUESTIONS,
        investigate_prompt=REFACTOR_INVESTIGATE,
        plan_prompt=REFACTOR_PLAN,
        execute_prompt=REFACTOR_EXECUTE,
        investigate_identity=REFACTOR_INVESTIGATE_IDENTITY,
        plan_identity=REFACTOR_PLAN_IDENTITY,
        execute_identity=REFACTOR_EXECUTE_IDENTITY,
        fallback_questions=REFACTOR_FALLBACK,
        questions_min=2,
        questions_max=6,
        investigate_max_tool_calls=12,
        plan_min_steps=3,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=15,
        auto_test=True,
        anti_rewrite=True,
    ),
    "other": PhaseStrategy(
        questions_prompt=OTHER_QUESTIONS,
        investigate_prompt=OTHER_INVESTIGATE,
        plan_prompt=OTHER_PLAN,
        execute_prompt=OTHER_EXECUTE,
        investigate_identity=OTHER_INVESTIGATE_IDENTITY,
        plan_identity=OTHER_PLAN_IDENTITY,
        execute_identity=OTHER_EXECUTE_IDENTITY,
        fallback_questions=OTHER_FALLBACK,
        questions_min=1,
        questions_max=5,
        investigate_max_tool_calls=12,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=15,
    ),
    "sysadmin": PhaseStrategy(
        questions_prompt=OTHER_QUESTIONS,
        investigate_prompt=OTHER_INVESTIGATE,
        plan_prompt=OTHER_PLAN,
        execute_prompt=OTHER_EXECUTE,
        investigate_identity=OTHER_INVESTIGATE_IDENTITY,
        plan_identity=OTHER_PLAN_IDENTITY,
        execute_identity=OTHER_EXECUTE_IDENTITY,
        fallback_questions=OTHER_FALLBACK,
        questions_min=1,
        questions_max=5,
        investigate_max_tool_calls=12,
        plan_min_steps=1,
        plan_max_step_files=3,
        execute_max_tool_calls_per_step=15,
    ),
}


def get_strategy(task_type: str) -> PhaseStrategy:
    """Get the phase strategy for a task type. Defaults to 'feature'."""
    return STRATEGIES.get(task_type, STRATEGIES["feature"])
