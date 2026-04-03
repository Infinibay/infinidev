"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_operation import StepOperation
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.step_result import StepResult
from infinidev.engine.loop.opened_file import OpenedFile
from infinidev.engine.loop.loop_state import LoopState

# Default TTL for opened files (in tool calls)
OPENED_FILE_TTL = 20
# Max number of files to keep in the cache (to avoid prompt bloat)
MAX_OPENED_FILES = 10
# Max file content size to cache (larger files are not cached)
MAX_CACHE_CONTENT_SIZE = 32000  # ~8K tokens — enough for most source files

