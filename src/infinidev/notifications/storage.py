"""SQLite-backed storage for programmable notifications.

Stored at ``~/.infinidev/notifications.db`` so the same notifications
are visible across projects — they describe cross-cutting concerns
(long-running watchers, daily reminders, post-test hooks), not anything
project-specific. This matches the precedent set by
``~/.infinidev/feedback.db`` for cross-project harness data.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

from infinidev.notifications.models import (
    ChannelConfig,
    Notification,
    TriggerSpec,
)

logger = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path.home() / ".infinidev" / "notifications.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS notifications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    name            TEXT NOT NULL UNIQUE,
    enabled         INTEGER NOT NULL DEFAULT 1,
    trigger_json    TEXT NOT NULL,
    channel_json    TEXT NOT NULL,
    title           TEXT NOT NULL DEFAULT '',
    template        TEXT NOT NULL DEFAULT '{name} fired at {fired_at}',
    created_at      REAL NOT NULL,
    last_fired_at   REAL,
    fire_count      INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_notifications_enabled ON notifications(enabled);

CREATE TABLE IF NOT EXISTS notification_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    notification_id INTEGER NOT NULL,
    fired_at        REAL NOT NULL,
    status          TEXT NOT NULL,
    error           TEXT,
    payload_json    TEXT,
    FOREIGN KEY (notification_id) REFERENCES notifications(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_history_notification ON notification_history(notification_id);
CREATE INDEX IF NOT EXISTS idx_history_fired_at ON notification_history(fired_at);
"""

_SCRIPT_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_script_state (
    notification_id INTEGER PRIMARY KEY,
    last_exit_code  INTEGER,
    last_stdout     TEXT,
    last_checked    REAL,
    last_match      INTEGER,
    FOREIGN KEY (notification_id) REFERENCES notifications(id) ON DELETE CASCADE
);
"""

_FILE_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS notification_file_state (
    notification_id INTEGER PRIMARY KEY,
    last_signature  TEXT NOT NULL,
    last_checked    REAL NOT NULL,
    FOREIGN KEY (notification_id) REFERENCES notifications(id) ON DELETE CASCADE
);
"""


def _default_db_path() -> Path:
    """Resolve the user-level DB path, creating the directory if needed."""
    path = DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class NotificationStore:
    """Thread-safe SQLite layer for notifications.

    All public methods take an internal lock — both the daemon scheduler
    thread and the tool thread may call into the store concurrently.
    """

    def __init__(self, db_path: os.PathLike[str] | str | None = None) -> None:
        self._path = Path(db_path) if db_path is not None else _default_db_path()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @property
    def path(self) -> Path:
        return self._path

    # ── Connection management ───────────────────────────────────────────
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=10.0, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.executescript(_SCRIPT_STATE_SCHEMA)
            conn.executescript(_FILE_STATE_SCHEMA)

    # ── CRUD ───────────────────────────────────────────────────────────
    def create(
        self,
        name: str,
        trigger: TriggerSpec,
        channel: ChannelConfig,
        *,
        title: str = "",
        template: str = "{name} fired at {fired_at}",
        enabled: bool = True,
    ) -> Notification:
        with self._lock, self._connect() as conn:
            now = time.time()
            try:
                cur = conn.execute(
                    """
                    INSERT INTO notifications
                      (name, enabled, trigger_json, channel_json,
                       title, template, created_at, fire_count)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 0)
                    """,
                    (
                        name,
                        1 if enabled else 0,
                        trigger.to_json(),
                        channel.to_json(),
                        title,
                        template,
                        now,
                    ),
                )
            except sqlite3.IntegrityError as e:
                raise ValueError(f"Notification named {name!r} already exists") from e
            new_id = cur.lastrowid
            conn.commit()
            row = conn.execute(
                "SELECT * FROM notifications WHERE id = ?", (new_id,)
            ).fetchone()
            return Notification.from_row(row)

    def update_enabled(self, notification_id: int, enabled: bool) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "UPDATE notifications SET enabled = ? WHERE id = ?",
                (1 if enabled else 0, notification_id),
            )
            conn.commit()
            return cur.rowcount > 0

    def delete(self, notification_id: int) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM notifications WHERE id = ?", (notification_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def delete_by_name(self, name: str) -> bool:
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM notifications WHERE name = ?", (name,)
            )
            conn.commit()
            return cur.rowcount > 0

    def get(self, notification_id: int) -> Notification | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notifications WHERE id = ?", (notification_id,)
            ).fetchone()
            return Notification.from_row(row) if row else None

    def get_by_name(self, name: str) -> Notification | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notifications WHERE name = ?", (name,)
            ).fetchone()
            return Notification.from_row(row) if row else None

    def list_all(self) -> list[Notification]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications ORDER BY id ASC"
            ).fetchall()
            return [Notification.from_row(r) for r in rows]

    def list_enabled(self) -> list[Notification]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM notifications WHERE enabled = 1 ORDER BY id ASC"
            ).fetchall()
            return [Notification.from_row(r) for r in rows]

    # ── Fire bookkeeping ───────────────────────────────────────────────
    def record_fire(
        self,
        notification_id: int,
        status: str,
        error: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self._lock, self._connect() as conn:
            now = time.time()
            conn.execute(
                """
                INSERT INTO notification_history
                  (notification_id, fired_at, status, error, payload_json)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    notification_id,
                    now,
                    status,
                    error,
                    json.dumps(payload or {}),
                ),
            )
            conn.execute(
                """
                UPDATE notifications
                   SET last_fired_at = ?, fire_count = fire_count + 1
                 WHERE id = ?
                """,
                (now, notification_id),
            )
            conn.commit()

    def history(
        self,
        notification_id: int | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as conn:
            if notification_id is None:
                rows = conn.execute(
                    """
                    SELECT h.*, n.name AS notification_name
                      FROM notification_history h
                      JOIN notifications n ON n.id = h.notification_id
                     ORDER BY h.fired_at DESC
                     LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT h.*, n.name AS notification_name
                      FROM notification_history h
                      JOIN notifications n ON n.id = h.notification_id
                     WHERE h.notification_id = ?
                     ORDER BY h.fired_at DESC
                     LIMIT ?
                    """,
                    (notification_id, limit),
                ).fetchall()
        out = []
        for r in rows:
            try:
                payload = json.loads(r["payload_json"] or "{}")
            except (TypeError, ValueError):
                payload = {}
            out.append(
                {
                    "id": r["id"],
                    "notification_id": r["notification_id"],
                    "notification_name": r["notification_name"],
                    "fired_at": r["fired_at"],
                    "status": r["status"],
                    "error": r["error"],
                    "payload": payload,
                }
            )
        return out

    # ── Trigger state caches ───────────────────────────────────────────
    def get_script_state(self, notification_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notification_script_state WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "last_exit_code": row["last_exit_code"],
            "last_stdout": row["last_stdout"],
            "last_checked": row["last_checked"],
            "last_match": bool(row["last_match"]),
        }

    def set_script_state(
        self,
        notification_id: int,
        exit_code: int | None,
        stdout: str,
        checked: float,
        matched: bool,
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notification_script_state
                  (notification_id, last_exit_code, last_stdout, last_checked, last_match)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(notification_id) DO UPDATE SET
                  last_exit_code = excluded.last_exit_code,
                  last_stdout = excluded.last_stdout,
                  last_checked = excluded.last_checked,
                  last_match = excluded.last_match
                """,
                (
                    notification_id,
                    exit_code,
                    stdout[:8192],
                    checked,
                    1 if matched else 0,
                ),
            )
            conn.commit()

    def get_file_state(self, notification_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM notification_file_state WHERE notification_id = ?",
                (notification_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "last_signature": row["last_signature"],
            "last_checked": row["last_checked"],
        }

    def set_file_state(
        self, notification_id: int, signature: str, checked: float
    ) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO notification_file_state
                  (notification_id, last_signature, last_checked)
                VALUES (?, ?, ?)
                ON CONFLICT(notification_id) DO UPDATE SET
                  last_signature = excluded.last_signature,
                  last_checked = excluded.last_checked
                """,
                (notification_id, signature, checked),
            )
            conn.commit()

    # ── Test helpers ───────────────────────────────────────────────────
    def reset(self) -> None:
        """Wipe every row. Used by tests; never call from production."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM notification_history")
            conn.execute("DELETE FROM notification_file_state")
            conn.execute("DELETE FROM notification_script_state")
            conn.execute("DELETE FROM notifications")
            conn.commit()


_default_store: NotificationStore | None = None
_default_store_lock = threading.Lock()


def get_default_store() -> NotificationStore:
    """Lazily construct the process-wide default store."""
    global _default_store
    if _default_store is None:
        with _default_store_lock:
            if _default_store is None:
                _default_store = NotificationStore()
    return _default_store


def reset_default_store_for_tests() -> None:
    """Clear the cached default store (tests only)."""
    global _default_store
    with _default_store_lock:
        _default_store = None