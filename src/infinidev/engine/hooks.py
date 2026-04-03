"""Hook system for intercepting and modifying engine behavior.

Unlike EventBus (fire-and-forget observation for UI), hooks run inline
in the execution pipeline and can modify arguments, results, or skip
execution entirely.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


from infinidev.engine.hook_event import HookEvent
from infinidev.engine.hook_context import HookContext
from infinidev.engine.hook_manager import HookManager

HookCallback = Callable[[HookContext], None]


def hook(event: HookEvent, *, priority: int = 100, name: str = ""):
    """Decorator to register a function as a hook.

    Usage::

        @hook(HookEvent.PRE_TOOL)
        def my_hook(ctx: HookContext):
            if ctx.tool_name == "execute_command":
                ctx.arguments["command"] = sanitize(ctx.arguments["command"])
    """

    def decorator(fn: HookCallback) -> HookCallback:
        hook_manager.register(event, fn, priority=priority, name=name or fn.__qualname__)
        return fn

    return decorator


hook_manager = HookManager()
