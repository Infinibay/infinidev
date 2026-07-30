"""Bridge MCP server events into TaskRuntime events."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from infinidev.engine.runtime_state import TaskStatus


class McpRuntimeBridge:
    """Translate raw MCP events into runtime memory + task state updates.

    The bridge observes ``started``/``tool_call``/``tool_result``/``tool_error``
    /``failure`` events from one or more MCP clients and routes them through a
    ``TaskRuntime`` so the runtime can keep the chat transcript untouched while
    remembering tool activity in working memory.
    """

    def __init__(
        self,
        runtime: Any,
        *,
        on_unavailable: Callable[[str, str], None] | None = None,
    ) -> None:
        self._runtime = runtime
        self._on_unavailable = on_unavailable

    def __call__(self, event: dict[str, Any]) -> None:
        kind = event.get("event")
        server = event.get("server", "mcp")
        if kind == "started":
            self._runtime.remember(
                f"MCP server {server} started", kind="mcp_lifecycle", importance=0.3
            )
        elif kind == "tool_call":
            self._runtime.remember(
                f"{server}.{event.get('tool')}", kind="mcp_call", importance=0.4
            )
        elif kind == "tool_error":
            self._runtime.remember(
                f"{server}.{event.get('tool')} failed: {event.get('error')}",
                kind="mcp_error",
                importance=0.5,
            )
            if self._on_unavailable is not None:
                self._on_unavailable(server, event.get("tool", ""))
        elif kind == "failure":
            self._runtime.remember(
                f"{server} failure ({event.get('count')}): {event.get('error')}",
                kind="mcp_failure",
                importance=0.6,
            )
            task_id = self._runtime.state.current_task_id
            if task_id:
                for task in self._runtime.state.tasks:
                    if task.id == task_id:
                        task.status = TaskStatus.BLOCKED
                        break

    def attach(self, manager: Any) -> None:
        """Register the bridge as *manager*'s event handler."""
        setter = getattr(manager, "set_event_handler", None)
        if callable(setter):
            setter(self)
            return
        for client in getattr(manager, "_servers", {}).values():
            client._on_event = self
