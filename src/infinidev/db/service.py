"""Database service for Infinidev CLI."""

import sqlite3
import logging
import os
from typing import Any, Callable
from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

def execute_with_retry(func: Callable[[sqlite3.Connection], Any]) -> Any:
    """Execute a DB operation with retry logic."""
    import time
    for attempt in range(settings.MAX_RETRIES):
        try:
            with sqlite3.connect(settings.DB_PATH) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.row_factory = sqlite3.Row
                return func(conn)
        except sqlite3.OperationalError as e:
            if ("locked" in str(e) or "busy" in str(e)) and attempt < settings.MAX_RETRIES - 1:
                time.sleep(settings.RETRY_BASE_DELAY * (2 ** attempt))
                continue
            raise
    return None

def _migrate_add_column(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    """Add a column to a table if it doesn't already exist."""
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
    except sqlite3.OperationalError:
        pass  # Column already exists


def init_db():
    """Initialize the SQLite database with essential tables."""
    def _init(conn):
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                name                  TEXT NOT NULL,
                description           TEXT,
                status                TEXT NOT NULL DEFAULT 'new',
                created_at            DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS findings (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id            INTEGER NOT NULL REFERENCES projects(id),
                session_id            TEXT,
                agent_id              TEXT,
                agent_run_id          TEXT,
                topic                 TEXT,
                content               TEXT,
                status                TEXT DEFAULT 'active',
                finding_type          TEXT DEFAULT 'observation',
                confidence            REAL DEFAULT 0.5,
                sources_json          TEXT DEFAULT '[]',
                tags_json             TEXT DEFAULT '[]',
                artifact_id           INTEGER,
                validation_method     TEXT,
                reproducibility_score REAL,
                embedding             BLOB,
                created_at            DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at            DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # FTS5 virtual table for full-text search on findings
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS findings_fts USING fts5(
                topic, content, content=findings, content_rowid=id
            )
        """)
        # Triggers to keep findings_fts in sync with findings
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS findings_ai AFTER INSERT ON findings BEGIN
                INSERT INTO findings_fts(rowid, topic, content)
                VALUES (new.id, new.topic, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS findings_ad AFTER DELETE ON findings BEGIN
                INSERT INTO findings_fts(findings_fts, rowid, topic, content)
                VALUES ('delete', old.id, old.topic, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS findings_au AFTER UPDATE ON findings BEGIN
                INSERT INTO findings_fts(findings_fts, rowid, topic, content)
                VALUES ('delete', old.id, old.topic, old.content);
                INSERT INTO findings_fts(rowid, topic, content)
                VALUES (new.id, new.topic, new.content);
            END
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifacts (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id),
                session_id    TEXT,
                type          TEXT DEFAULT 'artifact',
                name          TEXT,
                file_path     TEXT,
                description   TEXT,
                content       TEXT,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # FTS5 virtual table for full-text search on artifacts
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS artifacts_fts USING fts5(
                name, description, content, content=artifacts, content_rowid=id
            )
        """)
        # Triggers to keep artifacts_fts in sync with artifacts
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS artifacts_ai AFTER INSERT ON artifacts BEGIN
                INSERT INTO artifacts_fts(rowid, name, description, content)
                VALUES (new.id, new.name, new.description, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS artifacts_ad AFTER DELETE ON artifacts BEGIN
                INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description, content)
                VALUES ('delete', old.id, old.name, old.description, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS artifacts_au AFTER UPDATE ON artifacts BEGIN
                INSERT INTO artifacts_fts(artifacts_fts, rowid, name, description, content)
                VALUES ('delete', old.id, old.name, old.description, old.content);
                INSERT INTO artifacts_fts(rowid, name, description, content)
                VALUES (new.id, new.name, new.description, new.content);
            END
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS web_cache (
                url TEXT NOT NULL,
                format TEXT NOT NULL DEFAULT 'markdown',
                content TEXT,
                fetched_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(url, format)
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversation_turns(session_id, created_at)
        """)
        # Library documentation storage with FTS5 search
        conn.execute("""
            CREATE TABLE IF NOT EXISTS library_docs (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                library_name   TEXT NOT NULL,
                language       TEXT NOT NULL DEFAULT 'unknown',
                version        TEXT NOT NULL DEFAULT 'latest',
                section_title  TEXT NOT NULL,
                section_order  INTEGER NOT NULL DEFAULT 0,
                content        TEXT NOT NULL,
                embedding      BLOB,
                source_urls    TEXT DEFAULT '[]',
                created_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at     DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(library_name, language, version, section_title)
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_library_docs_lookup
                ON library_docs(library_name, language, version)
        """)
        conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS library_docs_fts USING fts5(
                section_title, content, content=library_docs, content_rowid=id
            )
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS library_docs_ai AFTER INSERT ON library_docs BEGIN
                INSERT INTO library_docs_fts(rowid, section_title, content)
                VALUES (new.id, new.section_title, new.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS library_docs_ad AFTER DELETE ON library_docs BEGIN
                INSERT INTO library_docs_fts(library_docs_fts, rowid, section_title, content)
                VALUES ('delete', old.id, old.section_title, old.content);
            END
        """)
        conn.execute("""
            CREATE TRIGGER IF NOT EXISTS library_docs_au AFTER UPDATE ON library_docs BEGIN
                INSERT INTO library_docs_fts(library_docs_fts, rowid, section_title, content)
                VALUES ('delete', old.id, old.section_title, old.content);
                INSERT INTO library_docs_fts(rowid, section_title, content)
                VALUES (new.id, new.section_title, new.content);
            END
        """)

        # Migrate existing databases: add columns that may be missing
        _migrate_add_column(conn, "findings", "session_id", "TEXT")
        _migrate_add_column(conn, "findings", "validation_method", "TEXT")
        _migrate_add_column(conn, "findings", "reproducibility_score", "REAL")
        _migrate_add_column(conn, "findings", "updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        _migrate_add_column(conn, "artifacts", "session_id", "TEXT")
        _migrate_add_column(conn, "artifacts", "type", "TEXT DEFAULT 'artifact'")

        # Create a default project if none exists
        row = conn.execute("SELECT id FROM projects LIMIT 1").fetchone()
        if not row:
            conn.execute("INSERT INTO projects (name, description) VALUES ('Default Project', 'Autogenerated project for CLI')")
        conn.commit()

    execute_with_retry(_init)


def store_conversation_turn(
    session_id: str, role: str, content: str, summary: str | None = None
) -> None:
    """Store a conversation turn in the database."""
    def _insert(conn):
        conn.execute(
            "INSERT INTO conversation_turns (session_id, role, content, summary) VALUES (?, ?, ?, ?)",
            (session_id, role, content, summary),
        )
        conn.commit()
    execute_with_retry(_insert)


def get_recent_summaries(session_id: str, limit: int = 10) -> list[str]:
    """Return the most recent conversation summaries for a session."""
    def _query(conn):
        rows = conn.execute(
            """\
            SELECT role, summary, content
            FROM conversation_turns
            WHERE session_id = ?
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
            SELECT topic, content, finding_type, confidence
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
                SELECT topic, content, finding_type, confidence
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
                "topic": row["topic"],
                "content": row["content"],
                "finding_type": row["finding_type"],
                "confidence": row["confidence"],
            })
        return results

    try:
        return execute_with_retry(_query) or []
    except Exception:
        return []
