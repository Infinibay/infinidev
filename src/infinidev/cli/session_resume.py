"""Session resume — the `-c`/`--continue` and `--resume` machinery.

Infinidev keeps model context compact while persisting the complete
structured transcript separately. "Continue yesterday's work" reuses
the same ``session_id`` so the existing machinery re-engages:

  * ``conversation_turns`` — the chat agent already reads history by
    session_id, so prior turns reappear automatically.
  * ``findings`` — already cross-session, project-scoped.
  * ContextRank scores — already cached per session.
  * ``session_notes`` — now persisted (see db.service) and re-loaded by
    the LoopEngine on first ``execute``.

  * ``session_messages`` — UI messages, tool calls/results, reasoning,
    diffs, and renderer metadata for an exact repaint.
  * ``session_runtime_state`` — task description, plan steps, and
    sidebar state.

This module resolves *which* session_id to reuse and queues one compact
model-facing resume checkpoint. The full history is returned only for local
UI repaint.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
import os
from pathlib import Path

from infinidev.db.service import (
    delete_session,
    get_all_turns,
    get_last_session,
    get_session_messages,
    get_session_runtime_state,
    get_sessions_storage_bytes,
    list_recent_sessions,
    register_session,
    rename_session,
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
    workspace = _workspace_label(row.get("workspace_path"))
    turns = row.get("turn_count") or 0
    size = _format_storage_bytes(row.get("storage_bytes") or 0)
    when = _relative_time(row.get("last_active_at"))
    return f"{title}  ·  {workspace}  ·  {turns} turns  ·  ~{size}  ·  {when}"


def recent_sessions(
    workspace_path: str | None = None, limit: int | None = None
) -> list[dict]:
    """Sessions for `--resume`, with current-workspace entries first.

    Unlike bare ``--continue``, the explicit picker exposes every session in
    the active workspace database. Keeping current-workspace history first
    preserves the useful local default while still allowing records associated
    with another workspace path in that database to be resumed.
    """
    workspace = workspace_path or current_workspace()
    local_sessions = list_recent_sessions(workspace, limit=None)
    local_ids = {session["session_id"] for session in local_sessions}
    other_sessions = [
        session
        for session in list_recent_sessions(None, limit=None)
        if session["session_id"] not in local_ids
    ]
    sessions = local_sessions + other_sessions
    selected = sessions if limit is None else sessions[:limit]
    storage_by_session = get_sessions_storage_bytes(
        [session["session_id"] for session in selected]
    )
    for session in selected:
        session["storage_bytes"] = storage_by_session.get(session["session_id"], 0)
    return selected


def name_session(session_id: str, title: str) -> str | None:
    """Persist a normalized display name for an existing session."""
    normalized = " ".join(title.split())[:80]
    if not normalized or not rename_session(session_id, normalized):
        return None
    return normalized


def pick_recent_session(
    prompt: Callable[[str], str],
    echo: Callable[[str], None],
    workspace_path: str | None = None,
) -> dict | None:
    """Interactively list, rename, delete, and select any recent session.

    A number resumes that session, ``rename NUMBER NAME`` updates its durable
    display name, and ``delete NUMBER`` removes it after confirmation. An empty
    response starts fresh.
    """
    sessions = recent_sessions(workspace_path)
    if not sessions:
        return None

    while True:
        echo("Recent sessions:")
        for index, session in enumerate(sessions, 1):
            echo(f"  {index}. {session_label(session)}")
        raw = prompt(
            "Resume which? (number, 'rename NUMBER NAME', 'delete NUMBER', "
            "or Enter for fresh)"
        ).strip()
        if not raw:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(sessions):
            return sessions[int(raw) - 1]

        command = raw.split(maxsplit=2)
        if len(command) == 3 and command[0].lower() == "rename" and command[1].isdigit():
            index = int(command[1])
            if 1 <= index <= len(sessions):
                session = sessions[index - 1]
                normalized = name_session(session["session_id"], command[2])
                if normalized:
                    session["title"] = normalized
                    echo("Session renamed.")
                    continue
        if len(command) == 2 and command[0].lower() == "delete" and command[1].isdigit():
            index = int(command[1])
            if 1 <= index <= len(sessions):
                session = sessions[index - 1]
                title = (session.get("title") or session["session_id"][:8]).strip()
                confirmed = prompt(
                    f"Permanently delete '{title}' and its session data? [y/N]"
                ).strip().lower()
                if confirmed not in {"y", "yes"}:
                    echo("Deletion cancelled.")
                    continue
                if delete_session(session["session_id"]):
                    sessions.pop(index - 1)
                    echo("Session deleted.")
                    if not sessions:
                        echo("No recent sessions remain; starting fresh.")
                        return None
                    continue
                echo("Session could not be deleted.")
                continue
        echo("Invalid choice.")


def begin_resumed_session(session_id: str, workspace_path: str | None = None) -> list[tuple[str, str]]:
    """Mark ``session_id`` as resumed and return its turns for repaint.

    Side effects: refreshes the ``sessions`` row (``last_active_at``)
    and queues a one-shot compact Step checkpoint for the model. The
    returned ``(role, content)`` pairs are for repainting the UI
    scrollback — that costs zero tokens.
    """
    # Refresh without passing the launch directory: an existing session keeps the
    # workspace where it was created, even when selected from another project.
    register_session(session_id)
    # Defer the import so a missing chat_agent (unlikely) never blocks resume.
    from infinidev.engine.orchestration.chat_agent import request_resume_context_once
    request_resume_context_once(session_id)
    return get_all_turns(
        session_id,
        limit=None,
        max_chars_per_turn=None,
    )


def resumed_session_state(session_id: str) -> dict:
    """Return the structured transcript and latest runtime snapshot."""
    runtime = get_session_runtime_state(session_id)
    return {
        "messages": get_session_messages(session_id),
        "task_description": runtime.get("task_description", ""),
        "plan_steps": runtime.get("plan_steps", []),
        "ui_state": runtime.get("ui_state", {}),
        "staged_planning": runtime.get("staged_planning", {}),
    }


def begin_fresh_session(session_id: str, workspace_path: str | None = None) -> None:
    """Register a brand-new session in the registry (no replay)."""
    register_session(session_id, workspace_path or current_workspace())


def _workspace_label(workspace_path: str | None) -> str:
    """Return a compact, still-identifying workspace path for picker rows."""
    if not workspace_path:
        return "unknown workspace"

    path = Path(workspace_path).expanduser()
    try:
        relative = path.relative_to(Path.home())
    except ValueError:
        label = str(path)
    else:
        label = "~" if not relative.parts else f"~/{relative}"

    max_length = 40
    if len(label) > max_length:
        return f"…{label[-(max_length - 1):]}"
    return label


def _format_storage_bytes(size: int) -> str:
    """Format a logical payload-byte estimate compactly for picker rows."""
    value = float(max(size, 0))
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{int(value)} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return "0 B"


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
