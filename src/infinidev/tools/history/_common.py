"""Shared helpers for the history_* tools."""

from __future__ import annotations

import json
from typing import Any


def snippet_of(payload: Any, max_chars: int = 400) -> str:
    """Compact single-line rendering of an event payload for search hits."""
    try:
        text = json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(payload)
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - 1] + "…"
    return text


def project_event(event: dict[str, Any], *, full_payload: bool = False) -> dict[str, Any]:
    """Normalize a store row for tool output.

    ``full_payload`` controls whether the whole payload is included (read /
    trace) or only a snippet (search hits). Visibility is surfaced so the
    caller can see why a payload may be withheld.
    """
    visibility = event.get("visibility", "public")
    data: dict[str, Any] = {
        "event_id": event.get("event_id"),
        "run_id": event.get("run_id"),
        "session_id": event.get("session_id"),
        "sequence": event.get("sequence"),
        "timestamp": event.get("timestamp"),
        "actor": event.get("actor"),
        "event_type": event.get("event_type"),
        "parent_event_id": event.get("parent_event_id"),
        "goal_revision": event.get("goal_revision"),
        "node_id": event.get("node_id"),
        "visibility": visibility,
    }
    if full_payload:
        # archive_only rows are audit-only: expose the type and metadata but
        # withhold the payload unless a later policy deliberately opens it.
        if visibility == "archive_only":
            data["payload"] = {"_withheld": "archive_only visibility"}
        else:
            data["payload"] = event.get("payload", {})
    else:
        data["payload_snippet"] = snippet_of(event.get("payload", {}))
    return data


__all__ = ["project_event", "snippet_of"]
