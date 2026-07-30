"""Durable event log backed by the existing SQLite conversation store."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_table() -> None:
    """Create the runtime_events table if it does not exist yet."""

    def _create(conn):
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS runtime_events ("
            " id TEXT PRIMARY KEY,"
            " session_id TEXT NOT NULL,"
            " task_id TEXT,"
            " event TEXT NOT NULL,"
            " payload TEXT NOT NULL,"
            " created_at TEXT NOT NULL"
            ");"
            "CREATE INDEX IF NOT EXISTS runtime_events_session_idx "
            "ON runtime_events(session_id, created_at);"
        )
        # execute_with_retry leaves committing to the caller.
        conn.commit()

    execute_with_retry(_create)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to a JSON-serializable structure."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            return repr(value)
    return repr(value)


def store_event(
    session_id: str,
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    task_id: str | None = None,
) -> str:
    """Persist *event* and return its assigned id.

    Best-effort: failures are logged but never raised so the runtime
    never sinks a task because the audit log couldn't be written.
    """
    if not session_id or not event:
        return ""
    try:
        _ensure_table()
        event_id = str(uuid.uuid4())
        payload_json = json.dumps(_json_safe(payload or {}), ensure_ascii=False)

        def _insert(conn):
            conn.execute(
                "INSERT INTO runtime_events ("
                " id, session_id, task_id, event, payload, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?)",
                (event_id, session_id, task_id, event, payload_json, _now()),
            )
            conn.commit()

        execute_with_retry(_insert)
        return event_id
    except Exception:
        logger.debug("runtime_events insert failed", exc_info=True)
        return ""


def list_events(
    session_id: str,
    *,
    limit: int = 200,
    task_id: str | None = None,
) -> list[dict[str, Any]]:
    """Return the most recent events for *session_id*, oldest first."""
    if not session_id:
        return []
    try:
        _ensure_table()

        def _select(conn):
            if task_id:
                return conn.execute(
                    "SELECT id, task_id, event, payload, created_at "
                    "FROM runtime_events WHERE session_id = ? AND task_id = ? "
                    "ORDER BY created_at ASC LIMIT ?",
                    (session_id, task_id, int(limit)),
                ).fetchall()
            return conn.execute(
                "SELECT id, task_id, event, payload, created_at "
                "FROM runtime_events WHERE session_id = ? "
                "ORDER BY created_at ASC LIMIT ?",
                (session_id, int(limit)),
            ).fetchall()

        rows = execute_with_retry(_select)
        return [_row_to_dict(row) for row in rows]
    except Exception:
        logger.debug("runtime_events list failed", exc_info=True)
        return []


def _row_to_dict(row: Any) -> dict[str, Any]:
    try:
        payload = json.loads(row[3]) if row[3] else {}
    except json.JSONDecodeError:
        payload = {"raw": row[3]}
    return {
        "id": row[0],
        "task_id": row[1],
        "event": row[2],
        "payload": payload,
        "created_at": row[4],
    }
