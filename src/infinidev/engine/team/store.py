"""Transactional team state and an append-only, attributed activity log."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from infinidev.tools.base.db import execute_with_retry


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


class TeamStore:
    """Keep one board per workspace/session; serialize competing updates in SQLite."""

    def __init__(self, team_id: str) -> None:
        self.team_id = team_id

        def create(conn):
            conn.executescript(
                "CREATE TABLE IF NOT EXISTS research_teams ("
                " id TEXT PRIMARY KEY, state TEXT NOT NULL);"
                "CREATE TABLE IF NOT EXISTS research_team_events ("
                " id INTEGER PRIMARY KEY AUTOINCREMENT, team_id TEXT NOT NULL,"
                " kind TEXT NOT NULL, author TEXT NOT NULL, recipient TEXT,"
                " ticket_id TEXT, reply_to INTEGER, supersedes INTEGER,"
                " content TEXT NOT NULL, refs TEXT NOT NULL, created_at TEXT NOT NULL);"
                "CREATE INDEX IF NOT EXISTS research_team_events_team "
                "ON research_team_events(team_id, id);"
                "CREATE UNIQUE INDEX IF NOT EXISTS research_team_note_revision "
                "ON research_team_events(team_id, supersedes) WHERE supersedes IS NOT NULL;"
            )
            conn.commit()

        execute_with_retry(create)

    def snapshot(self) -> dict[str, Any]:
        def read(conn):
            row = conn.execute(
                "SELECT state FROM research_teams WHERE id = ?", (self.team_id,),
            ).fetchone()
            return json.loads(row[0]) if row else {}

        return execute_with_retry(read)

    def update(self, change: Callable[[dict, Callable[..., int]], Any]) -> Any:
        """Apply a fresh-state mutation and its events in the same transaction."""
        def write(conn):
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT state FROM research_teams WHERE id = ?", (self.team_id,),
            ).fetchone()
            state = json.loads(row[0]) if row else {}

            def emit(kind, author, content, *, recipient=None, ticket_id=None,
                     reply_to=None, supersedes=None, refs=()):
                cursor = conn.execute(
                    "INSERT INTO research_team_events (team_id, kind, author, recipient,"
                    " ticket_id, reply_to, supersedes, content, refs, created_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (self.team_id, kind, author, recipient, ticket_id, reply_to,
                     supersedes, content, json.dumps(list(refs)), now()),
                )
                return int(cursor.lastrowid)

            result = change(state, emit)
            conn.execute(
                "INSERT INTO research_teams (id, state) VALUES (?, ?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state",
                (self.team_id, json.dumps(state, ensure_ascii=False)),
            )
            conn.commit()
            return result

        return execute_with_retry(write)

    def events(self, *, after: int = 0, limit: int = 50, kind: str | None = None,
               recipient: str | None = None, ticket_id: str | None = None) -> list[dict]:
        """Read a bounded page in sequence order, including supersession metadata."""
        where = ["e.team_id = ?", "e.id > ?"]
        args: list[Any] = [self.team_id, after]
        if kind:
            where.append("e.kind = ?")
            args.append(kind)
        if recipient:
            where.append("(e.recipient IS NULL OR e.recipient IN (?, 'all'))")
            args.append(recipient)
        if ticket_id:
            where.append("e.ticket_id = ?")
            args.append(ticket_id)
        args.append(max(1, min(100, limit)))

        def read(conn):
            rows = conn.execute(
                "SELECT e.id, e.kind, e.author, e.recipient, e.ticket_id, e.reply_to,"
                " e.supersedes, e.content, e.refs, e.created_at,"
                " (SELECT n.id FROM research_team_events n WHERE n.team_id=e.team_id"
                " AND n.supersedes=e.id) AS superseded_by"
                " FROM research_team_events e WHERE " + " AND ".join(where)
                + " ORDER BY e.id LIMIT ?", args,
            ).fetchall()
            fields = ("id", "kind", "author", "recipient", "ticket_id", "reply_to",
                      "supersedes", "content", "refs", "created_at", "superseded_by")
            result = [dict(zip(fields, row)) for row in rows]
            for item in result:
                item["refs"] = json.loads(item["refs"])
            return result

        return execute_with_retry(read)

    def event(self, event_id: int) -> dict:
        rows = self.events(after=event_id - 1, limit=1)
        if not rows or rows[0]["id"] != event_id:
            raise ValueError("Unknown event in this team")
        return rows[0]
