"""Database service for Infinidev CLI."""

import json
import logging
import os
import re
import sqlite3
from typing import Any

from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ALLOWED_COL_TYPES = {"TEXT", "INTEGER", "REAL", "BLOB", "TIMESTAMP", "DATETIME", "NUMERIC"}

# Canonical schema lives in schema.sql next to this module (the same file the
# Rust crate mirrors via include_str!). Fresh DBs are provisioned from it
# verbatim, so the DDL has a single source of truth.
_SCHEMA_SQL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.sql")


def _load_schema_sql() -> str:
    with open(_SCHEMA_SQL_PATH, "r", encoding="utf-8") as f:
        return f.read()


def execute_with_retry(func, db_path=None, max_retries=None, base_delay=None):
    """Execute a DB operation with retry logic.

    Delegates to the canonical implementation in tools/base/db.py which
    includes jitter, proper pragmas, and connection management.
    """
    from infinidev.tools.base.db import execute_with_retry as _canonical
    return _canonical(func, db_path=db_path, max_retries=max_retries, base_delay=base_delay)

def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column to a table if it doesn't already exist.

    Validates identifiers/type as defense in depth: today all callers pass
    literals, but SQLite has no parameter binding for DDL, so we have to
    interpolate. Rejecting anything that isn't a bare SQL identifier (or
    a type from a fixed whitelist) means future misuse fails loud rather
    than turning into a SQL-injection vector.
    """
    if not _IDENT_RE.match(table):
        raise ValueError(f"invalid table name: {table!r}")
    if not _IDENT_RE.match(column):
        raise ValueError(f"invalid column name: {column!r}")
    # Allow "TYPE" or "TYPE DEFAULT ..." — split on whitespace and check base type.
    base_type = col_type.strip().split()[0].upper() if col_type.strip() else ""
    if base_type not in _ALLOWED_COL_TYPES:
        raise ValueError(f"invalid column type: {col_type!r}")
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists


def init_db():
    """Initialize the SQLite database with essential tables."""
    def _init(conn):
        # Old databases need indexed columns before schema.sql can create the
        # corresponding indexes. Missing tables on a fresh database are a
        # harmless no-op here and are created immediately below.
        _migrate_add_column(conn, "ci_symbols", "embedding_space", "TEXT")
        _migrate_add_column(conn, "ci_files", "embedding_space", "TEXT")
        # Replace the historical catch-all update trigger. Derived embedding
        # refreshes must not rewrite the external-content FTS index.
        conn.execute("DROP TRIGGER IF EXISTS ci_symbols_au")

        # Fresh databases are fully provisioned from schema.sql (the single
        # source of truth). On an existing DB every CREATE ... IF NOT EXISTS
        # is a no-op.
        conn.executescript(_load_schema_sql())

        # ── Column back-fills for pre-existing databases ──────────────────
        # No-ops on a fresh DB (schema.sql already includes these columns),
        # but required to bring an OLD DB up to date: CREATE TABLE IF NOT
        # EXISTS will not add a column to a table that already exists.
        _migrate_add_column(conn, "findings", "session_id", "TEXT")
        _migrate_add_column(conn, "findings", "validation_method", "TEXT")
        _migrate_add_column(conn, "findings", "reproducibility_score", "REAL")
        _migrate_add_column(conn, "findings", "updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        # Anchored memory: each lesson/rule/landmine can be tied to a concrete
        # code location so it fires automatically when the agent touches that
        # anchor (see tool_executor._MEMORY_HANDLERS). All nullable.
        _migrate_add_column(conn, "findings", "anchor_file", "TEXT")
        _migrate_add_column(conn, "findings", "anchor_symbol", "TEXT")
        _migrate_add_column(conn, "findings", "anchor_tool", "TEXT")
        _migrate_add_column(conn, "findings", "anchor_error", "TEXT")
        _migrate_add_column(conn, "artifacts", "session_id", "TEXT")
        _migrate_add_column(conn, "artifacts", "type", "TEXT DEFAULT 'artifact'")
        _migrate_add_column(conn, "ci_files", "parser_version", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "ci_symbols", "embedding", "BLOB")
        _migrate_add_column(conn, "ci_symbols", "embedding_text", "TEXT")
        _migrate_add_column(conn, "ci_symbols", "embedding_space", "TEXT")
        _migrate_add_column(conn, "ci_files", "embedding", "BLOB")
        _migrate_add_column(conn, "ci_files", "embedding_text", "TEXT")
        _migrate_add_column(conn, "ci_files", "embedding_space", "TEXT")
        _migrate_add_column(conn, "findings", "embedding_space", "TEXT")
        _migrate_add_column(conn, "library_docs", "embedding_space", "TEXT")
        _migrate_add_column(conn, "cr_contexts", "embedding_space", "TEXT")
        _migrate_add_column(conn, "cr_interactions", "was_error", "INTEGER DEFAULT 0")
        _migrate_add_column(conn, "cr_session_scores", "productivity", "REAL DEFAULT 1.0")
        _migrate_add_column(conn, "cr_session_scores", "was_edited", "INTEGER DEFAULT 0")
        # Generated-image operations are isolated by the complete reviewed route.
        # These values are non-secret; credential_id is a SHA-256 fingerprint.
        _migrate_add_column(
            conn, "image_generation_operations", "endpoint", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn, "image_generation_operations", "transport", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn, "image_generation_operations", "adapter", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn, "image_generation_operations", "mechanism", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn, "image_generation_operations", "operation", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn, "image_generation_operations", "revision", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn,
            "image_generation_operations",
            "credential_type",
            "TEXT NOT NULL DEFAULT ''",
        )
        _migrate_add_column(
            conn, "image_generation_operations", "account_id", "TEXT NOT NULL DEFAULT ''"
        )
        _migrate_add_column(
            conn,
            "image_generation_operations",
            "generation_project_id",
            "TEXT NOT NULL DEFAULT ''",
        )
        _migrate_add_column(
            conn, "image_generation_operations", "credential_id", "TEXT NOT NULL DEFAULT ''"
        )

        # Seed a default project if none exists.
        row = conn.execute("SELECT id FROM projects LIMIT 1").fetchone()
        if not row:
            conn.execute(
                "INSERT INTO projects (name, description) "
                "VALUES ('Default Project', 'Autogenerated project for CLI')"
            )
        conn.commit()

    execute_with_retry(_init)


def store_conversation_turn(
    session_id: str, role: str, content: str, summary: str | None = None
) -> None:
    """Store a conversation turn in the database.

    Also keeps the ``sessions`` registry fresh: every turn bumps
    ``last_active_at`` and ``turn_count`` so ``-c`` can find the
    most-recently-active session. The UPDATE is a no-op if no session
    row exists yet (e.g. a turn stored before ``register_session`` ran),
    so this never fails on legacy callers.
    """
    def _insert(conn):
        conn.execute(
            "INSERT INTO conversation_turns (session_id, role, content, summary) VALUES (?, ?, ?, ?)",
            (session_id, role, content, summary),
        )
        conn.execute(
            "UPDATE sessions SET last_active_at = strftime('%Y-%m-%d %H:%M:%f','now'), "
            "turn_count = turn_count + 1 WHERE session_id = ?",
            (session_id,),
        )
        # Backfill the title from the first user message if still blank.
        if role == "user" and content:
            conn.execute(
                "UPDATE sessions SET title = ? "
                "WHERE session_id = ? AND (title IS NULL OR title = '')",
                (content.strip()[:80], session_id),
            )
        conn.commit()
    execute_with_retry(_insert)


def register_session(
    session_id: str,
    workspace_path: str | None = None,
    project_id: int = 1,
) -> None:
    """Create (or refresh) the ``sessions`` row for a session.

    Called once at CLI/TUI startup. Idempotent — resuming an existing
    session re-touches ``last_active_at`` without clobbering its title
    or turn_count.
    """
    def _upsert(conn):
        conn.execute(
            """
            INSERT INTO sessions (session_id, project_id, workspace_path, last_active_at)
            VALUES (?, ?, ?, strftime('%Y-%m-%d %H:%M:%f','now'))
            ON CONFLICT(session_id) DO UPDATE SET
                last_active_at = strftime('%Y-%m-%d %H:%M:%f','now'),
                workspace_path = COALESCE(excluded.workspace_path, sessions.workspace_path)
            """,
            (session_id, project_id, workspace_path),
        )
        conn.commit()
    execute_with_retry(_upsert)


def get_last_session(workspace_path: str | None = None) -> dict | None:
    """Return the most-recently-active session, or None.

    When ``workspace_path`` is given, only sessions from that directory
    are considered (the ``-c`` "continue this project" semantics). With
    no match there, returns None so the caller can fall back to a fresh
    session rather than resurrecting unrelated work.
    """
    def _query(conn):
        if workspace_path:
            row = conn.execute(
                "SELECT * FROM sessions WHERE workspace_path = ? "
                "ORDER BY last_active_at DESC LIMIT 1",
                (workspace_path,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT * FROM sessions ORDER BY last_active_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None
    return execute_with_retry(_query)


def list_recent_sessions(
    workspace_path: str | None = None, limit: int = 20
) -> list[dict]:
    """Return recent sessions (newest first) for the resume picker.

    Sessions with zero turns are skipped — they're empty shells from a
    launch that never sent a message. When ``workspace_path`` is given,
    scoping is preferred but NOT exclusive: if the directory has fewer
    than ``limit`` sessions we still only show that directory's work
    (cross-directory noise is worse than a short list).
    """
    def _query(conn):
        if workspace_path:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE workspace_path = ? AND turn_count > 0 "
                "ORDER BY last_active_at DESC LIMIT ?",
                (workspace_path, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM sessions WHERE turn_count > 0 "
                "ORDER BY last_active_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]
    return execute_with_retry(_query) or []


def persist_session_note(session_id: str, note_text: str) -> None:
    """Persist one session note so a resumed session can re-load it."""
    if not (session_id and note_text):
        return
    def _insert(conn):
        conn.execute(
            "INSERT INTO session_notes (session_id, note_text) VALUES (?, ?)",
            (session_id, note_text),
        )
        conn.commit()
    execute_with_retry(_insert)


def record_objective_verdict(
    *,
    session_id: str | None,
    step_index: int | None,
    title: str,
    kind: str,
    spec: str,
    verdict: str,
    detail: str = "",
    project_id: int = 1,
    agent_run_id: str | None = None,
) -> None:
    """Persist one objective-verification verdict to the durable ledger.

    Best-effort: the table is additive (schema.sql, created on next init),
    and a write failure must never break the review phase — callers wrap
    this so a missing/locked DB just means no ledger row.
    """
    def _insert(conn):
        conn.execute(
            "INSERT INTO objective_verdicts "
            "(project_id, session_id, agent_run_id, step_index, title, kind, spec, verdict, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, session_id, agent_run_id, step_index, title, kind, spec, verdict, detail),
        )
        conn.commit()
    execute_with_retry(_insert)


def get_objective_verdicts(
    session_id: str, *, agent_run_id: str | None = None, limit: int = 100
) -> list[dict]:
    """Return recent objective verdicts (newest first) for a session.

    The queryable read side of the ledger: powers "which objectives ended
    unmet" inspection and future resume-aware re-verification.
    """
    def _query(conn):
        if agent_run_id:
            rows = conn.execute(
                "SELECT step_index, title, kind, spec, verdict, detail, created_at "
                "FROM objective_verdicts WHERE session_id = ? AND agent_run_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (session_id, agent_run_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT step_index, title, kind, spec, verdict, detail, created_at "
                "FROM objective_verdicts WHERE session_id = ? "
                "ORDER BY created_at DESC, id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]
    return execute_with_retry(_query) or []


def get_session_notes(session_id: str, limit: int = 50) -> list[str]:
    """Return persisted session notes (oldest first) for a session."""
    def _query(conn):
        rows = conn.execute(
            "SELECT note_text FROM session_notes WHERE session_id = ? "
            "ORDER BY created_at ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [r["note_text"] for r in rows if r["note_text"]]
    return execute_with_retry(_query) or []


def store_session_message(
    session_id: str,
    message: dict[str, Any],
    *,
    message_id: int | None = None,
) -> int | None:
    """Insert or update one structured, UI-visible session message.

    ``conversation_turns`` remains the compact model-facing history. This
    ledger preserves renderer data such as tool arguments/results, diffs,
    reasoning messages, and critic metadata so ``-c`` can rebuild the actual
    transcript. Private runtime-only keys are intentionally omitted.
    """
    if not session_id or not isinstance(message, dict):
        return None

    from infinidev.config.secrets import redact

    public_message = {
        key: value
        for key, value in message.items()
        if not str(key).startswith("_")
    }
    payload = redact(json.dumps(public_message, ensure_ascii=False, default=str))

    def _upsert(conn):
        if message_id is not None:
            cursor = conn.execute(
                "UPDATE session_messages SET message_json = ? "
                "WHERE id = ? AND session_id = ?",
                (payload, message_id, session_id),
            )
            if cursor.rowcount:
                conn.commit()
                return message_id
        cursor = conn.execute(
            "INSERT INTO session_messages (session_id, message_json) VALUES (?, ?)",
            (session_id, payload),
        )
        conn.commit()
        return int(cursor.lastrowid)

    return execute_with_retry(_upsert)


def get_session_messages(session_id: str) -> list[dict[str, Any]]:
    """Return the complete structured transcript for ``session_id``."""
    if not session_id:
        return []

    def _query(conn):
        rows = conn.execute(
            "SELECT id, message_json FROM session_messages "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
        messages: list[dict[str, Any]] = []
        for row in rows:
            try:
                message = json.loads(row["message_json"])
            except (TypeError, json.JSONDecodeError):
                logger.warning("Skipping malformed session message id=%s", row["id"])
                continue
            if not isinstance(message, dict):
                continue
            message["_resume_message_id"] = row["id"]
            messages.append(message)
        return messages

    return execute_with_retry(_query) or []


def persist_session_runtime_state(
    session_id: str,
    *,
    task_description: str = "",
    plan_steps: list[dict[str, Any]] | None = None,
    ui_state: dict[str, Any] | None = None,
) -> None:
    """Persist the latest task/plan/sidebar snapshot for session resume."""
    if not session_id:
        return

    plan_json = json.dumps(plan_steps or [], ensure_ascii=False, default=str)
    requested_ui_state = dict(ui_state or {})

    def _upsert(conn):
        existing = conn.execute(
            "SELECT ui_state_json FROM session_runtime_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if existing and "staged_planning" not in requested_ui_state:
            try:
                prior_ui = json.loads(existing["ui_state_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                prior_ui = {}
            if isinstance(prior_ui, dict) and "staged_planning" in prior_ui:
                requested_ui_state["staged_planning"] = prior_ui["staged_planning"]
        ui_json = json.dumps(requested_ui_state, ensure_ascii=False, default=str)
        conn.execute(
            """
            INSERT INTO session_runtime_state
                (session_id, task_description, plan_steps_json, ui_state_json, updated_at)
            VALUES (?, ?, ?, ?, strftime('%Y-%m-%d %H:%M:%f','now'))
            ON CONFLICT(session_id) DO UPDATE SET
                task_description = excluded.task_description,
                plan_steps_json = excluded.plan_steps_json,
                ui_state_json = excluded.ui_state_json,
                updated_at = excluded.updated_at
            """,
            (session_id, task_description, plan_json, ui_json),
        )
        conn.commit()

    execute_with_retry(_upsert)


def get_session_runtime_state(session_id: str) -> dict[str, Any]:
    """Load the durable task/plan/sidebar snapshot for ``session_id``."""
    if not session_id:
        return {}

    def _query(conn):
        row = conn.execute(
            "SELECT task_description, plan_steps_json, ui_state_json "
            "FROM session_runtime_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if not row:
            return {}

        try:
            plan_steps = json.loads(row["plan_steps_json"] or "[]")
        except (TypeError, json.JSONDecodeError):
            plan_steps = []
        try:
            ui_state = json.loads(row["ui_state_json"] or "{}")
        except (TypeError, json.JSONDecodeError):
            ui_state = {}
        return {
            "task_description": row["task_description"] or "",
            "plan_steps": plan_steps if isinstance(plan_steps, list) else [],
            "ui_state": ui_state if isinstance(ui_state, dict) else {},
            "staged_planning": (
                ui_state.get("staged_planning", {})
                if isinstance(ui_state, dict)
                else {}
            ),
        }

    return execute_with_retry(_query) or {}


def persist_staged_planning_state(
    session_id: str,
    state: dict[str, Any],
    *,
    task_description: str = "",
) -> None:
    """Merge a durable staged-planning snapshot into the session state."""
    current = get_session_runtime_state(session_id)
    ui_state = dict(current.get("ui_state") or {})
    ui_state["staged_planning"] = state
    persist_session_runtime_state(
        session_id,
        task_description=str(current.get("task_description") or task_description),
        plan_steps=list(current.get("plan_steps") or []),
        ui_state=ui_state,
    )


def get_all_turns(
    session_id: str,
    limit: int | None = 200,
    max_chars_per_turn: int | None = 2000,
) -> list[tuple[str, str]]:
    """Return conversation turns oldest first as role/content pairs.

    Passing None disables the corresponding limit. The normal defaults keep
    callers bounded, while the TUI resume path opts out because rendering
    local scrollback costs no model-context tokens and must preserve the
    complete prior chat.
    """
    def _query(conn):
        # Exclude hidden work-summary turns: this powers the UI scrollback
        # repaint, and those turns are internal hand-off notes the user is
        # never meant to see. The model still gets them via
        # get_recent_turns_full().
        query = (
            "SELECT role, content FROM conversation_turns WHERE session_id = ? "
            "AND role != 'work_summary' ORDER BY created_at ASC, id ASC"
        )
        params: list[Any] = [session_id]
        if limit is not None:
            query += " LIMIT ?"
            params.append(max(0, limit))
        rows = conn.execute(query, params).fetchall()

        results: list[tuple[str, str]] = []
        for row in rows:
            content = row["content"] or ""
            if not content:
                continue
            if max_chars_per_turn is not None and len(content) > max_chars_per_turn:
                head = content[: max_chars_per_turn // 2]
                tail = content[-(max_chars_per_turn // 2):]
                content = f"{head}\n\n[...truncated middle...]\n\n{tail}"
            results.append((row["role"], content))
        return results
    return execute_with_retry(_query) or []


def get_recent_turns_full(
    session_id: str, limit: int = 6, max_chars_per_turn: int = 2000
) -> list[tuple[str, str]]:
    """Return the most recent turns as ``(role, content)`` pairs.

    Unlike :func:`get_recent_summaries` (which returns the truncated
    200-char ``summary`` snapshot used by the loop engine's compact
    history), this returns the *full* content of each turn, capped
    per-turn at ``max_chars_per_turn`` so a single huge assistant reply
    can't blow the caller's prompt budget.

    Used by the pre-analysis preamble: deciding whether a user message
    is "answerable from memory" requires actually seeing what the
    agent just said, not a 200-char fragment of it. The preamble would
    otherwise hallucinate elaborations of recommendations it can't
    actually see.
    """
    def _query(conn):
        rows = conn.execute(
            """\
            SELECT role, content
            FROM conversation_turns
            WHERE session_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        results: list[tuple[str, str]] = []
        for row in reversed(rows):
            content = row["content"] or ""
            if not content:
                continue
            if len(content) > max_chars_per_turn:
                # Keep head + tail so the model sees the opening
                # framing AND the closing recommendations, not just
                # the first half.
                head = content[: max_chars_per_turn // 2]
                tail = content[-(max_chars_per_turn // 2) :]
                content = f"{head}\n\n[...truncated middle...]\n\n{tail}"
            results.append((row["role"], content))
        return results
    return execute_with_retry(_query) or []


def get_recent_summaries(session_id: str, limit: int = 10) -> list[str]:
    """Return the most recent conversation summaries for a session."""
    def _query(conn):
        rows = conn.execute(
            """\
            SELECT role, summary, content
            FROM conversation_turns
            WHERE session_id = ? AND role != 'work_summary'
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (session_id, limit),
        ).fetchall()
        results = []
        for row in reversed(rows):
            text = row["summary"] or (row["content"][:200] if row["content"] else "")
            if text:
                results.append(f"[{row['role']}] {text}")
        return results
    return execute_with_retry(_query) or []


def get_all_findings(project_id: int = 1, limit: int = 200) -> list[dict]:
    """Return all findings for browsing in the TUI."""
    def _query(conn):
        rows = conn.execute(
            """\
            SELECT id, topic, content, finding_type, confidence, status, created_at
            FROM findings
            WHERE project_id = ?
            ORDER BY
                CASE finding_type WHEN 'project_context' THEN 0 ELSE 1 END,
                updated_at DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [
            {
                "id": row["id"],
                "topic": row["topic"],
                "content": row["content"],
                "finding_type": row["finding_type"],
                "confidence": row["confidence"],
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]
    try:
        return execute_with_retry(_query) or []
    except Exception:
        logger.warning("get_all_findings failed", exc_info=True)
        return []


def get_project_knowledge(project_id: int = 1, limit: int = 15) -> list[dict]:
    """Return the most relevant project knowledge findings.

    Fetches ``project_context`` findings first (always loaded), then recent
    high-confidence findings of other types.  Returns a compact list of dicts
    with ``topic``, ``content``, ``finding_type``, and ``confidence``.
    """
    def _query(conn):
        # 1. All project_context findings (structural knowledge)
        ctx_rows = conn.execute(
            """\
            SELECT id, topic, content, finding_type, confidence, status, created_at
            FROM findings
            WHERE project_id = ? AND finding_type = 'project_context'
              AND status IN ('active', 'provisional')
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()

        remaining = limit - len(ctx_rows)
        other_rows: list = []
        if remaining > 0:
            # 2. Recent high-confidence findings of other types
            other_rows = conn.execute(
                """\
                SELECT id, topic, content, finding_type, confidence, status, created_at
                FROM findings
                WHERE project_id = ? AND finding_type != 'project_context'
                  AND status IN ('active', 'provisional')
                  AND confidence >= 0.6
                ORDER BY confidence DESC, updated_at DESC
                LIMIT ?
                """,
                (project_id, remaining),
            ).fetchall()

        results = []
        for row in list(ctx_rows) + list(other_rows):
            results.append({
                "id": row["id"],
                "topic": row["topic"],
                "content": row["content"],
                "finding_type": row["finding_type"],
                "confidence": row["confidence"],
                "status": row["status"],
                "created_at": row["created_at"],
            })
        return results

    try:
        return execute_with_retry(_query) or []
    except Exception:
        logger.warning("get_project_knowledge failed", exc_info=True)
        return []


# ── Exploration Trees ─────────────────────────────────────────────────────────


def store_exploration_tree(
    project_id: int,
    problem: str,
    tree_json: str,
    *,
    session_id: str | None = None,
    agent_id: str | None = None,
    synthesis: str | None = None,
    status: str = "running",
    total_nodes: int = 0,
    total_tool_calls: int = 0,
    total_tokens: int = 0,
) -> int:
    """Store an exploration tree. Returns the row ID."""
    def _insert(conn):
        cursor = conn.execute(
            """\
            INSERT INTO exploration_trees
                (project_id, session_id, agent_id, problem, tree_json,
                 synthesis, status, total_nodes, total_tool_calls, total_tokens,
                 completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    CASE WHEN ? IN ('completed', 'exhausted', 'error') THEN CURRENT_TIMESTAMP ELSE NULL END)
            """,
            (project_id, session_id, agent_id, problem, tree_json,
             synthesis, status, total_nodes, total_tool_calls, total_tokens,
             status),
        )
        conn.commit()
        return cursor.lastrowid
    return execute_with_retry(_insert)


def get_exploration_tree(tree_id: int) -> dict | None:
    """Retrieve an exploration tree by ID."""
    def _query(conn):
        row = conn.execute(
            "SELECT * FROM exploration_trees WHERE id = ?", (tree_id,)
        ).fetchone()
        if row is None:
            return None
        return dict(row)
    return execute_with_retry(_query)


def get_recent_explorations(project_id: int = 1, limit: int = 10) -> list[dict]:
    """Return recent exploration trees for a project."""
    def _query(conn):
        rows = conn.execute(
            """\
            SELECT id, problem, status, total_nodes, total_tool_calls,
                   total_tokens, created_at, completed_at
            FROM exploration_trees
            WHERE project_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (project_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]
    try:
        return execute_with_retry(_query) or []
    except Exception:
        logger.warning("get_recent_explorations failed", exc_info=True)
        return []


# ── Anchored memory retrieval ─────────────────────────────────────────────


def get_anchored_findings(
    *,
    project_id: int = 1,
    anchor_file: str | None = None,
    anchor_symbol: str | None = None,
    anchor_tool: str | None = None,
    anchor_error: str | None = None,
    limit: int = 3,
) -> list[dict]:
    """Return findings that match ANY of the supplied anchors.

    Used by the tool executor to surface lessons/rules/landmines when
    the agent touches a file, symbol, tool, or error pattern they
    were anchored to. ``OR`` semantics across the anchor kinds — a
    finding matches if it's anchored to the file OR the symbol OR
    the tool OR the error, whichever is relevant for the caller.

    Only findings with ``finding_type`` in (``lesson``, ``rule``,
    ``landmine``) are eligible. Ordered by confidence DESC then
    recency DESC, capped at ``limit``. Returns the typical finding
    dict shape plus the anchor fields so the caller can explain
    which anchor triggered the match.
    """
    def _query(conn):
        conditions: list[str] = []
        params: list = [project_id]
        if anchor_file:
            conditions.append("anchor_file = ?")
            params.append(anchor_file)
        if anchor_symbol:
            conditions.append("anchor_symbol = ?")
            params.append(anchor_symbol)
        if anchor_tool:
            conditions.append("anchor_tool = ?")
            params.append(anchor_tool)
        if anchor_error:
            conditions.append("anchor_error = ?")
            params.append(anchor_error)
        if not conditions:
            return []
        where = "(" + " OR ".join(conditions) + ")"
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT id, topic, content, finding_type, confidence,
                   anchor_file, anchor_symbol, anchor_tool, anchor_error,
                   created_at
            FROM findings
            WHERE project_id = ?
              AND status IN ('active', 'provisional')
              AND finding_type IN ('lesson', 'rule', 'landmine')
              AND {where}
            ORDER BY confidence DESC, updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
        return [dict(r) for r in rows]

    try:
        return execute_with_retry(_query) or []
    except Exception:
        logger.warning("get_anchored_findings failed", exc_info=True)
        return []
