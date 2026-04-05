"""LLM calling with manual-TC / FC-mode branching and retry."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from infinidev.engine.llm_client import (
    call_llm as _call_llm,
    is_malformed_tool_call as _is_malformed_tool_call,
    PERMANENT_ERRORS as _PERMANENT_ERRORS,
)
from infinidev.engine.engine_logging import (
    emit_log as _emit_log,
    YELLOW as _YELLOW,
    RED as _RED,
    RESET as _RESET,
)
from infinidev.engine.loop.context import build_system_prompt, build_tools_prompt_section
from infinidev.engine.loop.models import StepResult
from infinidev.engine.formats.tool_call_parser import (
    ManualToolCall as _ManualToolCall,
    parse_text_tool_calls as _parse_text_tool_calls,
)

if TYPE_CHECKING:
    from infinidev.engine.loop.execution_context import ExecutionContext


@dataclass
class ClassifiedCalls:
    """Tool calls separated by category."""

    regular: list[Any] = field(default_factory=list)
    step_complete: Any | None = None
    notes: list[Any] = field(default_factory=list)
    session_notes: list[Any] = field(default_factory=list)
    thinks: list[Any] = field(default_factory=list)


