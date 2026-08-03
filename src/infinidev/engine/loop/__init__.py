"""Loop engine package — plan-execute-summarize cycle.

Re-exports key classes for convenient imports:
    from infinidev.engine.loop import LoopEngine
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.loop.execution_context import ExecutionContext
    from infinidev.engine.loop.llm_caller import ClassifiedCalls, LLMCaller, LLMCallResult
    from infinidev.engine.loop.loop_guard import LoopGuard
    from infinidev.engine.loop.step_manager import StepManager
    from infinidev.engine.loop.tool_processor import ToolProcessor

__all__ = [
    "LoopEngine",
    "ExecutionContext",
    "LLMCaller",
    "LLMCallResult",
    "ClassifiedCalls",
    "ToolProcessor",
    "LoopGuard",
    "StepManager",
]

_EXPORTS = {
    "LoopEngine": ("infinidev.engine.loop.engine", "LoopEngine"),
    "ExecutionContext": (
        "infinidev.engine.loop.execution_context",
        "ExecutionContext",
    ),
    "LLMCaller": ("infinidev.engine.loop.llm_caller", "LLMCaller"),
    "LLMCallResult": ("infinidev.engine.loop.llm_caller", "LLMCallResult"),
    "ClassifiedCalls": ("infinidev.engine.loop.llm_caller", "ClassifiedCalls"),
    "ToolProcessor": ("infinidev.engine.loop.tool_processor", "ToolProcessor"),
    "LoopGuard": ("infinidev.engine.loop.loop_guard", "LoopGuard"),
    "StepManager": ("infinidev.engine.loop.step_manager", "StepManager"),
}


def __getattr__(name: str) -> Any:
    """Load convenience exports without importing the complete loop eagerly."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
