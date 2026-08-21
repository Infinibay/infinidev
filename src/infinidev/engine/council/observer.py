"""Observable, UI-neutral state for council sessions and their members."""

from __future__ import annotations

import threading
import uuid
from collections import deque
from copy import deepcopy
from typing import Any

from infinidev.config.settings import settings
from infinidev.flows.event_listeners import event_bus


_lock = threading.RLock()
_sessions: dict[str, dict[str, Any]] = {}
_terminal_order: deque[str] = deque()
_delivered_terminal_events: set[str] = set()
_eviction_count = 0


def _evict_terminal_history(limit: int) -> None:
    """Evict delivered terminal councils in completion order; caller holds ``_lock``."""
    global _eviction_count

    while len(_terminal_order) > limit:
        oldest_id = _terminal_order[0]
        if oldest_id not in _delivered_terminal_events:
            break
        _terminal_order.popleft()
        _delivered_terminal_events.remove(oldest_id)
        if _sessions.pop(oldest_id, None) is not None:
            _eviction_count += 1


def start_council(
    *,
    question: str,
    members: list[dict[str, str]],
    project_id: int | None,
    parent_agent_id: str = "council",
) -> str:
    """Register a council and announce its inspectable member roster."""
    council_id = f"council-{uuid.uuid4().hex[:8]}"
    state = {
        "id": council_id,
        "question": question,
        "status": "running",
        "members": {
            member["member_id"]: {
                **member,
                "status": "waiting",
                "messages": [],
            }
            for member in members
        },
        "messages": [],
    }
    with _lock:
        _sessions[council_id] = state
        snapshot = _summary(state)
    _emit_snapshot("council_started", council_id, snapshot, project_id, parent_agent_id)
    return council_id


def set_member_status(
    council_id: str,
    member_id: str,
    status: str,
    *,
    project_id: int | None,
    round_num: int | None = None,
) -> None:
    """Update one member's lifecycle state."""
    with _lock:
        state = _sessions.get(council_id)
        member = (state or {}).get("members", {}).get(member_id)
        if state is None or state["status"] != "running" or member is None:
            return
        member["status"] = status
        if round_num is not None:
            member["round"] = round_num
        snapshot = _summary(state)
    _emit_snapshot("council_agent_status", council_id, snapshot, project_id, member_id)


def add_message(
    council_id: str,
    member_id: str,
    text: str,
    *,
    project_id: int | None,
    round_num: int,
    action: str,
) -> None:
    """Append one contribution to both the debate and member transcripts."""
    if not text:
        return
    with _lock:
        state = _sessions.get(council_id)
        member = (state or {}).get("members", {}).get(member_id)
        if state is None or state["status"] != "running" or member is None:
            return
        message = {
            "member_id": member_id,
            "persona": member.get("persona", ""),
            "objective": member.get("objective", ""),
            "text": text,
            "round": round_num,
            "action": action,
        }
        state["messages"].append(message)
        member["messages"].append(message)
        member["status"] = "completed" if action == "conclude" else "waiting"
        snapshot = _summary(state)
    _emit_snapshot(
        "council_agent_message",
        council_id,
        snapshot,
        project_id,
        member_id,
        message=message,
    )


def finish_council(
    council_id: str,
    status: str,
    *,
    project_id: int | None,
) -> None:
    """Mark a council terminal, emit its final snapshot, and bound retained history."""
    with _lock:
        state = _sessions.get(council_id)
        if state is None or state["status"] != "running":
            return
        state["status"] = status
        for member in state["members"].values():
            if member["status"] in {"running", "waiting"}:
                member["status"] = "completed" if status == "completed" else status

        snapshot = _summary(state)
        _terminal_order.append(council_id)

    # Listener callbacks may inspect council state from another thread, so never
    # invoke the event bus while holding the observer lock.
    _emit_snapshot("council_finished", council_id, snapshot, project_id, "council")

    with _lock:
        if _sessions.get(council_id) is not state:
            return
        _delivered_terminal_events.add(council_id)
        limit = settings.COUNCIL_HISTORY_LIMIT
        if limit is not None:
            _evict_terminal_history(limit)


def _reconcile_council_history() -> None:
    """Apply the current retention limit; caller holds ``_lock``."""
    limit = settings.COUNCIL_HISTORY_LIMIT
    if limit is not None:
        _evict_terminal_history(limit)


def get_council(council_id: str) -> dict[str, Any] | None:
    """Return a detached snapshot safe for renderers and commands."""
    with _lock:
        _reconcile_council_history()
        state = _sessions.get(council_id)
        return deepcopy(state) if state is not None else None


def list_councils(*, include_messages: bool = True) -> list[dict[str, Any]]:
    """Return active councils first, then terminal councils by completion recency."""
    with _lock:
        _reconcile_council_history()
        terminal_ids = set(_terminal_order)
        active_states = [
            state
            for council_id, state in reversed(_sessions.items())
            if council_id not in terminal_ids
        ]
        terminal_states = [
            _sessions[council_id]
            for council_id in reversed(_terminal_order)
            if council_id in _sessions
        ]
        states = active_states + terminal_states
        if include_messages:
            return [deepcopy(state) for state in states]
        return [_summary(state) for state in states]


def running_agent_count() -> int:
    """Return the live agent count without allocating transcript snapshots."""
    with _lock:
        return sum(
            member.get("status") == "running"
            for state in _sessions.values()
            if state.get("status") == "running"
            for member in state.get("members", {}).values()
        )


def council_eviction_count() -> int:
    """Return how many terminal council transcripts were evicted since reset."""
    with _lock:
        return _eviction_count


def clear_councils() -> None:
    """Clear process-local council history (primarily for tests)."""
    global _eviction_count

    with _lock:
        _sessions.clear()
        _terminal_order.clear()
        _delivered_terminal_events.clear()
        _eviction_count = 0


def _emit_snapshot(
    event_type: str,
    council_id: str,
    snapshot: dict[str, Any],
    project_id: int | None,
    agent_id: str,
    **extra: Any,
) -> None:
    """Emit an event from a snapshot captured while holding ``_lock``."""
    event_bus.emit(
        event_type,
        project_id or 1,
        agent_id,
        {"council_id": council_id, "council": snapshot, **extra},
    )


def _summary(state: dict[str, Any]) -> dict[str, Any]:
    """Copy only bounded metadata needed by status lines and selectors."""
    return {
        "id": state["id"],
        "question": state.get("question", ""),
        "status": state.get("status", ""),
        "members": {
            member_id: {
                key: value
                for key, value in member.items()
                if key != "messages"
            }
            for member_id, member in state.get("members", {}).items()
        },
        "message_count": len(state.get("messages", [])),
    }


__all__ = [
    "add_message",
    "clear_councils",
    "council_eviction_count",
    "finish_council",
    "get_council",
    "list_councils",
    "running_agent_count",
    "set_member_status",
    "start_council",
]
