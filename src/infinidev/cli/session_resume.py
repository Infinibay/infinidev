"""Session resume — the `-c`/`--continue` and `--resume` machinery.

Infinidev's context is cheap by construction: raw tool output is
discarded and every turn rebuilds the prompt from compact summaries.
So "continue yesterday's work" does NOT mean replaying a heavy
transcript (the Claude `-c` cost) — it means *reusing the same
``session_id``* so the existing machinery re-engages:

  * ``conversation_turns`` — the chat agent already reads history by
    session_id, so prior turns reappear automatically.
  * ``findings`` — already cross-session, project-scoped.
  * ContextRank scores — already cached per session.
  * ``session_notes`` — now persisted (see db.service) and re-loaded by
    the LoopEngine on first ``execute``.

This module only resolves *which* session_id to reuse and, on the
first resumed turn, asks the chat agent to replay the full history
once (the user opted into "historial completo al modelo").
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from infinidev.db.service import (
    get_last_session,
    list_recent_sessions,
    register_session,
    get_all_turns,
)


def current_workspace() -> str:
    """The directory whose sessions `-c` should scope to."""
    return os.environ.get("INFINIDEV_WORKSPACE") or os.getcwd()


def resolve_continue_session(workspace_path: str | None = None) -> dict | None:
    """Pick the session that bare ``-c`` should resume.

    Prefers the most-recent session for this workspace; falls back to
    the most-recent session anywhere (so `-c` still does something
    useful from a directory that has no history of its own). Returns
    None only when the DB has no sessions at all.
    """
    ws = workspace_path or current_workspace()
    return get_last_session(ws) or get_last_session(None)


def session_label(row: dict) -> str:
    """Human one-liner for a session in the picker / banner."""
    title = (row.get("title") or "(untitled)").strip().replace("\n", " ")
    if len(title) > 60:
        title = title[:57] + "..."
    turns = row.get("turn_count") or 0
    when = _relative_time(row.get("last_active_at"))
    return f"{title}  ·  {turns} turns  ·  {when}"


def recent_sessions(workspace_path: str | None = None, limit: int = 20) -> list[dict]:
    """Sessions for the `--resume` picker (workspace-scoped, newest first)."""
    return list_recent_sessions(workspace_path or current_workspace(), limit=limit)


def begin_resumed_session(session_id: str, workspace_path: str | None = None) -> list[tuple[str, str]]:
    """Mark ``session_id`` as resumed and return its turns for repaint.

    Side effects: refreshes the ``sessions`` row (``last_active_at``)
    and queues a one-shot full-history replay for the model. The
    returned ``(role, content)`` pairs are for repainting the UI
    scrollback — that costs zero tokens.
    """
    register_session(session_id, workspace_path or current_workspace())
    # Defer the import so a missing chat_agent (unlikely) never blocks resume.
    from infinidev.engine.orchestration.chat_agent import request_full_history_once
    request_full_history_once(session_id)
    return get_all_turns(session_id)


def begin_fresh_session(session_id: str, workspace_path: str | None = None) -> None:
    """Register a brand-new session in the registry (no replay)."""
    register_session(session_id, workspace_path or current_workspace())


def _relative_time(ts: str | None) -> str:
    """Best-effort '2h ago' style label from a SQLite timestamp string."""
    if not ts:
        return "unknown"
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
        try:
            dt = datetime.strptime(ts, fmt).replace(tzinfo=timezone.utc)
            break
        except ValueError:
            continue
    else:
        return ts
    delta = datetime.now(timezone.utc) - dt
    secs = int(delta.total_seconds())
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{secs // 60}m ago"
    if secs < 86400:
        return f"{secs // 3600}h ago"
    return f"{secs // 86400}d ago"
