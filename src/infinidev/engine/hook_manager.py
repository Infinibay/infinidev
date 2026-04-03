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
from infinidev.engine.hook_context import HookContext
from infinidev.engine.hook_event import HookEvent


@dataclass(order=True)
class _HookEntry:
    """Internal: a registered hook callback with priority."""
    priority: int
    callback: Callable[[HookContext], None] = field(compare=False)
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


