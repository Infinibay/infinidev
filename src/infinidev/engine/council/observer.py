"""Observable, UI-neutral state for council sessions and their members."""

from __future__ import annotations

import threading
import uuid
from copy import deepcopy
from typing import Any

from infinidev.flows.event_listeners import event_bus


_lock = threading.RLock()
_sessions: dict[str, dict[str, Any]] = {}


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
    _emit("council_started", council_id, project_id, parent_agent_id)
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
        if member is None:
            return
        member["status"] = status
        if round_num is not None:
            member["round"] = round_num
    _emit("council_agent_status", council_id, project_id, member_id)


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
        if state is None or member is None:
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
    _emit("council_agent_message", council_id, project_id, member_id, message=message)


def finish_council(
    council_id: str,
    status: str,
    *,
    project_id: int | None,
) -> None:
    """Mark a council terminal without discarding its inspectable history."""
    with _lock:
        state = _sessions.get(council_id)
        if state is None:
            return
        state["status"] = status
        for member in state["members"].values():
            if member["status"] in {"running", "waiting"}:
                member["status"] = "completed" if status == "completed" else status
    _emit("council_finished", council_id, project_id, "council")


def get_council(council_id: str) -> dict[str, Any] | None:
    """Return a detached snapshot safe for renderers and commands."""
    with _lock:
        state = _sessions.get(council_id)
        return deepcopy(state) if state is not None else None


def list_councils() -> list[dict[str, Any]]:
    """Return all council snapshots, newest first."""
    with _lock:
        return [deepcopy(state) for state in reversed(_sessions.values())]


def clear_councils() -> None:
    """Clear process-local council history (primarily for tests)."""
    with _lock:
        _sessions.clear()


def _emit(
    event_type: str,
    council_id: str,
    project_id: int | None,
    agent_id: str,
    **extra: Any,
) -> None:
    state = get_council(council_id)
    if state is None:
        return
    event_bus.emit(
        event_type,
        project_id or 1,
        agent_id,
        {"council_id": council_id, "council": state, **extra},
    )


__all__ = [
    "add_message",
    "clear_councils",
    "finish_council",
    "get_council",
    "list_councils",
    "set_member_status",
    "start_council",
]
