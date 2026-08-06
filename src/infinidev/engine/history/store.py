"""Append-only execution event store backed by SQLite.

The store is the persistence side of the event log
(docs/GRAPH_ENGINE_BETA_DESIGN.md §10): ``engine_runs`` holds one row per
orchestrated task run; ``execution_events`` is the append-only canonical
record. All writes go through ``execute_with_retry`` so WAL contention gets
the project-standard exponential backoff.

Single-writer contract: one coordinator owns a run and appends its events
sequentially, so per-run sequences are assigned inside the same transaction
that inserts the row. When a later phase adds the Graph reducer, it becomes
the only writer for graph-mutating events; nothing about this interface
changes.

Failures raise — callers in the pipeline wrap the store in best-effort
guards because the event log must never sink a task run.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import time
import uuid
from typing import Any

from infinidev.code_intel._db import execute_with_retry, sanitize_fts5_query
from infinidev.engine.history.events import (
    VISIBILITY_ARCHIVE_ONLY,
    VISIBILITY_PUBLIC,
)
from infinidev.engine.history.redaction import redact_payload

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

#: Run status vocabulary stored in ``engine_runs.status``.
RUN_RUNNING = "running"
RUN_COMPLETED = "completed"
RUN_BLOCKED = "blocked"
RUN_CANCELLED = "cancelled"
RUN_FAILED = "failed"


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:16]}"


def new_event_id() -> str:
    return f"evt_{uuid.uuid4().hex[:16]}"


def _canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _row_to_event(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
    data = dict(row)
    raw_payload = data.pop("payload_json", "{}")
    try:
        data["payload"] = json.loads(raw_payload) if raw_payload else {}
    except (TypeError, ValueError):
        data["payload"] = {"_raw": raw_payload}
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Runs
# ─────────────────────────────────────────────────────────────────────────────


def create_run(
    *,
    session_id: str,
    engine: str,
    mode: str = "",
    goal_title: str = "",
    goal_request: str = "",
    project_id: int | None = None,
    parent_run_id: str | None = None,
    selection: dict[str, Any] | None = None,
    run_id: str | None = None,
) -> str:
    """Insert one ``engine_runs`` row and return its run_id."""
    rid = run_id or new_run_id()
    selection_payload = redact_payload(selection or {})
    clean_goal_title = redact_payload(goal_title or "")
    clean_goal_request = redact_payload(goal_request or "")

    def _insert(conn: sqlite3.Connection) -> str:
        conn.execute(
            """INSERT INTO engine_runs
               (run_id, session_id, project_id, parent_run_id, engine, mode,
                goal_title, goal_request, status, selection_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rid, session_id, project_id, parent_run_id, engine, mode,
                clean_goal_title, clean_goal_request, RUN_RUNNING,
                json.dumps(selection_payload, ensure_ascii=False),
            ),
        )
        conn.commit()
        return rid

    return execute_with_retry(_insert)


def finish_run(
    run_id: str,
    status: str,
    *,
    digest: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
) -> None:
    """Close a run: terminal status + digest + metrics."""

    def _update(conn: sqlite3.Connection) -> None:
        conn.execute(
            """UPDATE engine_runs
               SET status = ?, digest_json = ?, metrics_json = ?,
                   finished_at = CURRENT_TIMESTAMP
               WHERE run_id = ?""",
            (
                status,
                json.dumps(redact_payload(digest or {}), ensure_ascii=False),
                json.dumps(metrics or {}, ensure_ascii=False),
                run_id,
            ),
        )
        conn.commit()

    execute_with_retry(_update)


def update_run_engine(run_id: str, engine: str) -> None:
    """Record a mid-run engine switch on the run row."""

    def _update(conn: sqlite3.Connection) -> None:
        conn.execute(
            "UPDATE engine_runs SET engine = ? WHERE run_id = ?",
            (engine, run_id),
        )
        conn.commit()

    execute_with_retry(_update)


def get_run(run_id: str) -> dict[str, Any] | None:
    def _select(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM engine_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        data = dict(row)
        for key in ("selection_json", "digest_json", "metrics_json"):
            raw = data.get(key)
            try:
                data[key] = json.loads(raw) if raw else {}
            except (TypeError, ValueError):
                data[key] = {}
        return data

    try:
        return execute_with_retry(_select)
    except sqlite3.Error:
        logger.debug("get_run failed for %s", run_id, exc_info=True)
        return None


def latest_run_for_session(session_id: str) -> dict[str, Any] | None:
    def _select(conn: sqlite3.Connection) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT run_id FROM engine_runs WHERE session_id = ? "
            "ORDER BY started_at DESC, rowid DESC LIMIT 1",
            (session_id,),
        ).fetchone()
        return dict(row) if row is not None else None

    try:
        info = execute_with_retry(_select)
    except sqlite3.Error:
        return None
    return get_run(info["run_id"]) if info else None


# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────


def append_event(
    run_id: str,
    session_id: str,
    event_type: str,
    payload: dict[str, Any] | None = None,
    *,
    actor: str = "system",
    parent_event_id: str | None = None,
    goal_revision: int | None = None,
    node_id: str | None = None,
    visibility: str = VISIBILITY_PUBLIC,
) -> str:
    """Append one event and return its event_id.

    The per-run sequence is assigned inside the inserting transaction, and
    the payload is redacted and content-hashed before it touches disk.
    """
    event_id = new_event_id()
    clean_payload = redact_payload(payload or {})
    payload_json = json.dumps(clean_payload, ensure_ascii=False)
    content_hash = _canonical_hash(clean_payload if isinstance(clean_payload, dict) else {"value": clean_payload})
    if visibility not in (VISIBILITY_PUBLIC, VISIBILITY_ARCHIVE_ONLY):
        visibility = VISIBILITY_PUBLIC

    def _insert(conn: sqlite3.Connection) -> str:
        row = conn.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM execution_events "
            "WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        sequence = int(row[0]) if row is not None else 1
        conn.execute(
            """INSERT INTO execution_events
               (event_id, run_id, session_id, sequence, timestamp, actor,
                event_type, parent_event_id, goal_revision, node_id,
                visibility, payload_json, schema_version, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                event_id, run_id, session_id, sequence, time.time(), actor,
                event_type, parent_event_id, goal_revision, node_id,
                visibility, payload_json, SCHEMA_VERSION, content_hash,
            ),
        )
        conn.commit()
        return event_id

    return execute_with_retry(_insert)


def list_run_events(
    run_id: str,
    *,
    include_archive: bool = False,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        clause = "run_id = ?"
        params: list[Any] = [run_id]
        if not include_archive:
            clause += " AND visibility != ?"
            params.append(VISIBILITY_ARCHIVE_ONLY)
        sql = (
            "SELECT * FROM execution_events WHERE " + clause +
            " ORDER BY sequence"
        )
        if limit:
            sql += " LIMIT ?"
            params.append(limit)
        return [_row_to_event(row) for row in conn.execute(sql, params)]

    try:
        return execute_with_retry(_select)
    except sqlite3.Error:
        logger.debug("list_run_events failed for %s", run_id, exc_info=True)
        return []


def search_events(
    *,
    query: str = "",
    session_id: str | None = None,
    run_id: str | None = None,
    event_type: str | None = None,
    node_id: str | None = None,
    after: float | None = None,
    before: float | None = None,
    include_archive: bool = False,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Hybrid search: FTS5 over payloads plus structured filters.

    With an empty *query* this browses the most recent matching events.
    Falls back to LIKE matching when the FTS table is unavailable so an
    old database still answers.
    """
    filters: list[str] = []
    params: list[Any] = []
    if session_id:
        filters.append("e.session_id = ?")
        params.append(session_id)
    if run_id:
        filters.append("e.run_id = ?")
        params.append(run_id)
    if event_type:
        filters.append("e.event_type = ?")
        params.append(event_type)
    if node_id:
        filters.append("e.node_id = ?")
        params.append(node_id)
    if after is not None:
        filters.append("e.timestamp >= ?")
        params.append(after)
    if before is not None:
        filters.append("e.timestamp <= ?")
        params.append(before)
    if not include_archive:
        filters.append("e.visibility != ?")
        params.append(VISIBILITY_ARCHIVE_ONLY)
    where = (" WHERE " + " AND ".join(filters)) if filters else ""

    text = (query or "").strip()

    def _fts(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        if text:
            safe = sanitize_fts5_query(text)
            sql = (
                "SELECT e.*, bm25(execution_events_fts) AS score "
                "FROM execution_events e "
                "JOIN execution_events_fts f ON e.id = f.rowid "
                f"WHERE execution_events_fts MATCH ?{(' AND ' + ' AND '.join(filters)) if filters else ''} "
                "ORDER BY score LIMIT ?"
            )
            rows = conn.execute(sql, [safe, *params, limit]).fetchall()
        else:
            sql = (
                "SELECT e.*, NULL AS score FROM execution_events e"
                f"{where} ORDER BY e.id DESC LIMIT ?"
            )
            rows = conn.execute(sql, [*params, limit]).fetchall()
        results = []
        for row in rows:
            data = _row_to_event(row)
            data["score"] = row["score"] if "score" in row.keys() else None
            data["match_reason"] = "fts" if text else "browse"
            results.append(data)
        return results

    def _like(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        clause = list(filters)
        bound = list(params)
        if text:
            clause.append("e.payload_json LIKE ?")
            bound.append(f"%{text}%")
        where_like = (" WHERE " + " AND ".join(clause)) if clause else ""
        sql = (
            "SELECT e.*, NULL AS score FROM execution_events e"
            f"{where_like} ORDER BY e.id DESC LIMIT ?"
        )
        rows = conn.execute(sql, [*bound, limit]).fetchall()
        results = []
        for row in rows:
            data = _row_to_event(row)
            data["score"] = None
            data["match_reason"] = "like"
            results.append(data)
        return results

    try:
        return execute_with_retry(_fts)
    except sqlite3.OperationalError:
        logger.debug("FTS search unavailable; falling back to LIKE", exc_info=True)
        return execute_with_retry(_like)


def read_events(
    event_ids: list[str],
    *,
    window_before: int = 0,
    window_after: int = 0,
) -> list[dict[str, Any]]:
    """Fetch events by id, optionally with a sequence window around each."""

    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        marks = ",".join("?" for _ in event_ids)
        rows = conn.execute(
            f"SELECT * FROM execution_events WHERE event_id IN ({marks})",
            list(event_ids),
        ).fetchall()
        picked: dict[tuple[str, int], sqlite3.Row] = {}
        for row in rows:
            picked[(row["run_id"], row["sequence"])] = row
        if window_before or window_after:
            spans: list[Any] = []
            for row in rows:
                spans.extend((
                    row["run_id"],
                    row["sequence"] - window_before,
                    row["sequence"] + window_after,
                ))
            if spans:
                clauses = " OR ".join(
                    "(run_id = ? AND sequence BETWEEN ? AND ?)"
                    for _ in range(len(spans) // 3)
                )
                for row in conn.execute(
                    "SELECT * FROM execution_events WHERE " + clauses,
                    spans,
                ).fetchall():
                    picked[(row["run_id"], row["sequence"])] = row
        ordered = sorted(picked.values(), key=lambda r: (r["run_id"], r["sequence"]))
        return [_row_to_event(row) for row in ordered]

    if not event_ids:
        return []
    return execute_with_retry(_select)


def events_around(
    run_id: str,
    sequence: int,
    *,
    window_before: int = 0,
    window_after: int = 0,
) -> list[dict[str, Any]]:
    """Events in *run_id* within a sequence window around *sequence*."""

    def _select(conn: sqlite3.Connection) -> list[dict[str, Any]]:
        rows = conn.execute(
            "SELECT * FROM execution_events WHERE run_id = ? "
            "AND sequence BETWEEN ? AND ? ORDER BY sequence",
            (run_id, sequence - window_before, sequence + window_after),
        ).fetchall()
        return [_row_to_event(row) for row in rows]

    return execute_with_retry(_select)


def trace_chain(
    start_event_id: str,
    *,
    max_depth: int = 100,
) -> list[dict[str, Any]]:
    """Walk parent_event_id links from *start_event_id* back to the root.

    Returns the chain root-first so callers can read it as the causal story
    that led to the starting event.
    """
    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    current_id: str | None = start_event_id

    def _one(conn: sqlite3.Connection, event_id: str) -> dict[str, Any] | None:
        row = conn.execute(
            "SELECT * FROM execution_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return _row_to_event(row) if row is not None else None

    while current_id and current_id not in seen and len(chain) < max_depth:
        seen.add(current_id)
        event = execute_with_retry(lambda conn, _id=current_id: _one(conn, _id))
        if event is None:
            break
        chain.append(event)
        current_id = event.get("parent_event_id")
    chain.reverse()
    return chain


__all__ = [
    "RUN_BLOCKED",
    "RUN_CANCELLED",
    "RUN_COMPLETED",
    "RUN_FAILED",
    "RUN_RUNNING",
    "SCHEMA_VERSION",
    "append_event",
    "create_run",
    "events_around",
    "finish_run",
    "get_run",
    "latest_run_for_session",
    "list_run_events",
    "new_event_id",
    "new_run_id",
    "read_events",
    "search_events",
    "trace_chain",
    "update_run_engine",
]
