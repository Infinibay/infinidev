"""Compatibility export for the task planner prompt."""

from __future__ import annotations

from infinidev.prompts.analyst.task_planner_prompt import TASK_PLANNER_SYSTEM_PROMPT


# ``run_planner`` keeps the historical import while the stage orchestrator is
# introduced. Compatibility imports of this module resolve to the Task Planner.
ANALYST_PLANNER_SYSTEM_PROMPT = TASK_PLANNER_SYSTEM_PROMPT


__all__ = ["ANALYST_PLANNER_SYSTEM_PROMPT"]
