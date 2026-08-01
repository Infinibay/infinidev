"""Bounded, rate-limited progress events for running tools."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from infinidev.flows.event_listeners import event_bus


_EMIT_INTERVAL_SECONDS = 0.1
_MAX_PENDING_CHARS = 64_000
_local = threading.local()


@dataclass
class _ProgressState:
    run_id: str
    project_id: int
    agent_id: str
    cancel_event: Any | None = None
    pending: str = ""
    last_emit: float = 0.0


@contextmanager
def tool_progress_context(
    run_id: str,
    project_id: int,
    agent_id: str,
    *,
    cancel_event: Any | None = None,
) -> Iterator[None]:
    """Route progress emitted by the current tool to its UI event stream."""
    previous = getattr(_local, "state", None)
    state = _ProgressState(
        run_id=run_id,
        project_id=project_id,
        agent_id=agent_id,
        cancel_event=cancel_event,
    )
    _local.state = state
    try:
        yield
    finally:
        _flush(state)
        _local.state = previous


def emit_tool_output(chunk: str) -> None:
    """Publish a coalesced output chunk for the currently executing tool."""
    state: _ProgressState | None = getattr(_local, "state", None)
    if state is None or not chunk:
        return

    state.pending = (state.pending + str(chunk))[-_MAX_PENDING_CHARS:]
    now = time.monotonic()
    if now - state.last_emit >= _EMIT_INTERVAL_SECONDS:
        _flush(state, now)


def is_tool_cancelled() -> bool:
    """Return whether the engine asked the current foreground tool to stop."""
    state: _ProgressState | None = getattr(_local, "state", None)
    event = state.cancel_event if state is not None else None
    return bool(event is not None and event.is_set())


def _flush(state: _ProgressState, now: float | None = None) -> None:
    if not state.pending:
        return
    chunk = state.pending
    state.pending = ""
    state.last_emit = now if now is not None else time.monotonic()
    event_bus.emit(
        "loop_tool_output",
        state.project_id,
        state.agent_id,
        {"tool_run_id": state.run_id, "chunk": chunk},
    )
