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


class HookEvent(str, Enum):
    """All hookable events in the engine lifecycle."""

    # Tool hooks
    PRE_TOOL = "pre_tool"
    POST_TOOL = "post_tool"

    # Step lifecycle
    PRE_STEP = "pre_step"
    POST_STEP = "post_step"
    STEP_TRANSITION = "step_transition"

    # Loop lifecycle
    LOOP_START = "loop_start"
    LOOP_END = "loop_end"

    # LLM call
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"


@dataclass
class HookContext:
    """Mutable bag of data passed through the hook chain.

    Each hook event populates relevant fields.  Hooks read and modify
    these freely.  Setting ``skip = True`` in a pre-hook prevents the
    default action (e.g. tool execution).

    A single ``HookContext`` instance is reused for the PRE and POST
    phases of the same operation so that hooks can stash data in
    ``metadata`` during the PRE phase and read it back in POST.
    """

    event: HookEvent
    tool_name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str | None = None
    skip: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    project_id: int = 0
    agent_id: str = ""


HookCallback = Callable[[HookContext], None]


@dataclass(order=True)
class _HookEntry:
    priority: int
    callback: HookCallback = field(compare=False)
    name: str = field(default="", compare=False)


class HookManager:
    """Thread-safe hook registry and dispatcher.

    Hooks are called in priority order (lower number = earlier).
    Default priority is 100.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._hooks: dict[HookEvent, list[_HookEntry]] = {}

    def register(
        self,
        event: HookEvent,
        callback: HookCallback,
        *,
        priority: int = 100,
        name: str = "",
    ) -> None:
        entry = _HookEntry(
            priority=priority,
            callback=callback,
            name=name or getattr(callback, "__qualname__", repr(callback)),
        )
        with self._lock:
            hooks = self._hooks.setdefault(event, [])
            hooks.append(entry)
            hooks.sort()

    def unregister(self, event: HookEvent, callback: HookCallback) -> None:
        with self._lock:
            if event in self._hooks:
                self._hooks[event] = [
                    h for h in self._hooks[event] if h.callback is not callback
                ]

    def clear(self, event: HookEvent | None = None) -> None:
        with self._lock:
            if event is None:
                self._hooks.clear()
            else:
                self._hooks.pop(event, None)

    def dispatch(self, ctx: HookContext) -> HookContext:
        """Run all hooks for ``ctx.event`` in priority order.

        Returns the (possibly modified) context.  Short-circuits if a
        hook sets ``ctx.skip = True``.
        """
        with self._lock:
            entries = list(self._hooks.get(ctx.event, []))
        for entry in entries:
            try:
                entry.callback(ctx)
            except Exception:
                logger.warning(
                    "Hook %r failed on %s",
                    entry.name,
                    ctx.event,
                    exc_info=True,
                )
            if ctx.skip:
                break
        return ctx

    @property
    def has_hooks(self) -> bool:
        with self._lock:
            return any(bool(v) for v in self._hooks.values())

    def has_hooks_for(self, event: HookEvent) -> bool:
        with self._lock:
            return bool(self._hooks.get(event))


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
