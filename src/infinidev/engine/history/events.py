"""Execution-event vocabulary for the engine event log.

The event log (``execution_events``) is the append-only canonical record of
what each task engine did; the ``graph_nodes`` / ``graph_edges`` projections
that later phases add are rebuilt FROM these events, never the reverse.

This module owns the stable names. Two rules keep the log trustworthy:

* **Additive only.** New event types are added; existing names and payload
  keys are never renamed. Consumers key off the literal strings below.
* **Versioned payloads.** Every row carries ``schema_version`` so a reader
  that predates a payload change can still round-trip the row.

See docs/GRAPH_ENGINE_BETA_DESIGN.md §10 for the full event catalog.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


Visibility = Literal["public", "archive_only"]

#: Default visibility. Public events are searchable and readable by the
#: history tools. ``archive_only`` rows persist for audit but are hidden
#: from retrieval until a later phase deliberately surfaces them.
VISIBILITY_PUBLIC: str = "public"
VISIBILITY_ARCHIVE_ONLY: str = "archive_only"


# ── Run lifecycle ────────────────────────────────────────────────────────────
RUN_STARTED = "run_started"
ENGINE_SELECTED = "engine_selected"
TASK_PROFILE_RESOLVED = "task_profile_resolved"
ENGINE_SWITCHED = "engine_switched"
RUN_PAUSED = "run_paused"
RUN_RESUMED = "run_resumed"
RUN_CANCELLED = "run_cancelled"
RUN_COMPLETED = "run_completed"
RUN_BLOCKED = "run_blocked"
RUN_FAILED = "run_failed"

# ── Intent / goal ────────────────────────────────────────────────────────────
GOAL_REVISED = "goal_revised"
GOAL_RESOLVED = "goal_resolved"

# ── Graph (reserved for the Graph beta phases; kept stable now) ──────────────
GRAPH_PATCHED = "graph_patched"
NODE_ACTIVATED = "node_activated"
NODE_CHECKPOINTED = "node_checkpointed"
NODE_RESOLVED = "node_resolved"
NODE_INVALIDATED = "node_invalidated"
EVIDENCE_ATTACHED = "evidence_attached"

# ── Staged projection (Stage / Task lifecycle) ───────────────────────────────
STAGE_OPENED = "stage_opened"
STAGE_CLOSED = "stage_closed"
TASK_STARTED = "task_started"
TASK_CLOSED = "task_closed"

# ── Tool execution ───────────────────────────────────────────────────────────
TOOL_REQUESTED = "tool_requested"
TOOL_STARTED = "tool_started"
TOOL_PROGRESSED = "tool_progressed"
TOOL_FINISHED = "tool_finished"

# ── Recovery / summarisation ─────────────────────────────────────────────────
DIGEST_CREATED = "digest_created"

#: Every event type the current build may write. Readers use this to validate
#: filters; writers are not restricted to it (forward compatibility).
KNOWN_EVENT_TYPES: frozenset[str] = frozenset({
    RUN_STARTED, ENGINE_SELECTED, ENGINE_SWITCHED, TASK_PROFILE_RESOLVED,
    RUN_PAUSED, RUN_RESUMED, RUN_CANCELLED,
    RUN_COMPLETED, RUN_BLOCKED, RUN_FAILED,
    GOAL_REVISED, GOAL_RESOLVED,
    GRAPH_PATCHED, NODE_ACTIVATED, NODE_CHECKPOINTED,
    NODE_RESOLVED, NODE_INVALIDATED, EVIDENCE_ATTACHED,
    STAGE_OPENED, STAGE_CLOSED, TASK_STARTED, TASK_CLOSED,
    TOOL_REQUESTED, TOOL_STARTED, TOOL_PROGRESSED, TOOL_FINISHED,
    DIGEST_CREATED,
})

#: Terminal event types — exactly one closes a run. ``run_paused`` is NOT
#: terminal: a paused run is resumable and keeps the same run_id.
TERMINAL_EVENT_TYPES: frozenset[str] = frozenset({
    RUN_COMPLETED, RUN_BLOCKED, RUN_CANCELLED, RUN_FAILED,
})


class ExecutionEvent(BaseModel):
    """One immutable row of the execution event log.

    Field names mirror the ``execution_events`` columns 1:1 so projection
    and replay code can round-trip without a mapping layer. ``payload`` is
    the decoded ``payload_json``; ``content_hash`` binds the row to the
    exact payload that was written.
    """

    event_id: str
    run_id: str
    session_id: str
    sequence: int
    timestamp: float
    actor: str = "system"
    event_type: str
    parent_event_id: str | None = None
    goal_revision: int | None = None
    node_id: str | None = None
    visibility: str = VISIBILITY_PUBLIC
    payload: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1
    content_hash: str | None = None


__all__ = [
    "DIGEST_CREATED",
    "ENGINE_SELECTED",
    "ENGINE_SWITCHED",
    "TASK_PROFILE_RESOLVED",
    "EVIDENCE_ATTACHED",
    "ExecutionEvent",
    "GOAL_RESOLVED",
    "GOAL_REVISED",
    "GRAPH_PATCHED",
    "KNOWN_EVENT_TYPES",
    "NODE_ACTIVATED",
    "NODE_CHECKPOINTED",
    "NODE_INVALIDATED",
    "NODE_RESOLVED",
    "RUN_BLOCKED",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_PAUSED",
    "RUN_RESUMED",
    "RUN_STARTED",
    "STAGE_CLOSED",
    "STAGE_OPENED",
    "TASK_CLOSED",
    "TASK_STARTED",
    "TERMINAL_EVENT_TYPES",
    "TOOL_FINISHED",
    "TOOL_PROGRESSED",
    "TOOL_REQUESTED",
    "TOOL_STARTED",
    "VISIBILITY_ARCHIVE_ONLY",
    "VISIBILITY_PUBLIC",
]
