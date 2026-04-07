"""Database service for Infinidev CLI."""

import sqlite3
import logging
import os
from typing import Any
from infinidev.config.settings import settings

logger = logging.getLogger(__name__)

def execute_with_retry(func, db_path=None, max_retries=None, base_delay=None):
    """Execute a DB operation with retry logic.

    Delegates to the canonical implementation in tools/base/db.py which
    includes jitter, proper pragmas, and connection management.
    """
    from infinidev.tools.base.db import execute_with_retry as _canonical
    return _canonical(func, db_path=db_path, max_retries=max_retries, base_delay=base_delay)

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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS exploration_trees (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id       INTEGER NOT NULL REFERENCES projects(id),
                session_id       TEXT,
                agent_id         TEXT,
                problem          TEXT NOT NULL,
                tree_json        TEXT NOT NULL,
                synthesis        TEXT,
                status           TEXT DEFAULT 'running',
                total_nodes      INTEGER DEFAULT 0,
                total_tool_calls INTEGER DEFAULT 0,
                total_tokens     INTEGER DEFAULT 0,
                created_at       DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at     DATETIME
            )
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS artifact_changes (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id),
                agent_run_id  TEXT,
                file_path     TEXT NOT NULL,
                action        TEXT NOT NULL DEFAULT 'modified',
                before_hash   TEXT,
                after_hash    TEXT,
                size_bytes    INTEGER,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS status_updates (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id),
                agent_id      TEXT,
                agent_run_id  TEXT,
                message       TEXT,
                progress      REAL,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS branches (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL REFERENCES projects(id),
                task_id       TEXT,
                repo_name     TEXT,
                branch_name   TEXT NOT NULL,
                base_branch   TEXT,
                status        TEXT DEFAULT 'active',
                created_by    TEXT,
                created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Migrate existing databases: add columns that may be missing
        _migrate_add_column(conn, "findings", "session_id", "TEXT")
        _migrate_add_column(conn, "findings", "validation_method", "TEXT")
        _migrate_add_column(conn, "findings", "reproducibility_score", "REAL")
        _migrate_add_column(conn, "findings", "updated_at", "DATETIME DEFAULT CURRENT_TIMESTAMP")
        _migrate_add_column(conn, "artifacts", "session_id", "TEXT")
        _migrate_add_column(conn, "artifacts", "type", "TEXT DEFAULT 'artifact'")

        # ── Code Intelligence tables ──────────────────────────────────────

        conn.execute("""\
            CREATE TABLE IF NOT EXISTS ci_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                language TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                symbol_count INTEGER DEFAULT 0,
                indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, file_path)
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_files_path ON ci_files(project_id, file_path)")

        conn.execute("""\
            CREATE TABLE IF NOT EXISTS ci_symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                qualified_name TEXT DEFAULT '',
                kind TEXT NOT NULL,
                line_start INTEGER NOT NULL,
                line_end INTEGER,
                column_start INTEGER DEFAULT 0,
                signature TEXT DEFAULT '',
                type_annotation TEXT DEFAULT '',
                docstring TEXT DEFAULT '',
                parent_symbol TEXT DEFAULT '',
                visibility TEXT DEFAULT 'public',
                is_async BOOLEAN DEFAULT FALSE,
                is_static BOOLEAN DEFAULT FALSE,
                is_abstract BOOLEAN DEFAULT FALSE,
                language TEXT NOT NULL
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_symbols_name ON ci_symbols(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_symbols_kind ON ci_symbols(kind, name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_symbols_file ON ci_symbols(project_id, file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_symbols_qualified ON ci_symbols(qualified_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_symbols_parent ON ci_symbols(parent_symbol)")

        conn.execute("""\
            CREATE VIRTUAL TABLE IF NOT EXISTS ci_symbols_fts USING fts5(
                name, qualified_name, signature, docstring,
                content=ci_symbols, content_rowid=id
            )""")

        # FTS triggers for ci_symbols
        conn.executescript("""\
            CREATE TRIGGER IF NOT EXISTS ci_symbols_ai AFTER INSERT ON ci_symbols BEGIN
                INSERT INTO ci_symbols_fts(rowid, name, qualified_name, signature, docstring)
                VALUES (new.id, new.name, new.qualified_name, new.signature, new.docstring);
            END;
            CREATE TRIGGER IF NOT EXISTS ci_symbols_ad AFTER DELETE ON ci_symbols BEGIN
                INSERT INTO ci_symbols_fts(ci_symbols_fts, rowid, name, qualified_name, signature, docstring)
                VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.docstring);
            END;
            CREATE TRIGGER IF NOT EXISTS ci_symbols_au AFTER UPDATE ON ci_symbols BEGIN
                INSERT INTO ci_symbols_fts(ci_symbols_fts, rowid, name, qualified_name, signature, docstring)
                VALUES ('delete', old.id, old.name, old.qualified_name, old.signature, old.docstring);
                INSERT INTO ci_symbols_fts(rowid, name, qualified_name, signature, docstring)
                VALUES (new.id, new.name, new.qualified_name, new.signature, new.docstring);
            END;
        """)

        conn.execute("""\
            CREATE TABLE IF NOT EXISTS ci_references (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                name TEXT NOT NULL,
                line INTEGER NOT NULL,
                column_num INTEGER DEFAULT 0,
                context TEXT DEFAULT '',
                ref_kind TEXT DEFAULT 'usage',
                resolved_file TEXT DEFAULT '',
                resolved_line INTEGER,
                language TEXT NOT NULL
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_refs_name ON ci_references(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_refs_file ON ci_references(project_id, file_path)")

        conn.execute("""\
            CREATE TABLE IF NOT EXISTS ci_imports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                source TEXT NOT NULL,
                name TEXT NOT NULL,
                alias TEXT DEFAULT '',
                line INTEGER NOT NULL,
                is_wildcard BOOLEAN DEFAULT FALSE,
                resolved_file TEXT DEFAULT '',
                language TEXT NOT NULL
            )""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_imports_file ON ci_imports(project_id, file_path)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_imports_name ON ci_imports(name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_imports_source ON ci_imports(source)")

        # Code intelligence: diagnostics from heuristic analysis
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ci_diagnostics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                line INTEGER NOT NULL,
                severity TEXT NOT NULL,
                check_name TEXT NOT NULL,
                message TEXT NOT NULL,
                fix_suggestion TEXT DEFAULT '',
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ci_diag_file ON ci_diagnostics(project_id, file_path)")

        # Code intelligence: per-method body fingerprints for fuzzy
        # similarity search across the project. Populated by the
        # ``code_intel.method_index`` module immediately after
        # ``store_file_symbols`` runs, so it stays in sync with the
        # symbol index without a separate background pass.
        #
        # Why a separate table instead of columns on ci_symbols:
        #   * Most ci_symbols rows are imports, fields, properties —
        #     things with no body to fingerprint. Adding two TEXT
        #     columns to ci_symbols would make every row pay for a
        #     feature 90% of rows can't use.
        #   * Similarity search has its own index requirements
        #     (idx_method_bodies_hash for exact-dup, idx_method_bodies_qual
        #     for "fetch THIS method"), unrelated to the symbol indexes.
        #   * The table can be cleared and rebuilt independently of
        #     ci_symbols if the normalization scheme changes.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS ci_method_bodies (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id    INTEGER NOT NULL,
                file_path     TEXT NOT NULL,
                qualified_name TEXT NOT NULL,
                kind          TEXT NOT NULL,    -- 'function' | 'method'
                line_start    INTEGER NOT NULL,
                line_end      INTEGER NOT NULL,
                body_size     INTEGER NOT NULL, -- in lines (after stripping comments)
                body_hash     TEXT NOT NULL,    -- normalized sha256[:16] for exact-dup
                body_norm     TEXT NOT NULL,    -- space-separated normalized tokens (for Jaccard)
                language      TEXT NOT NULL,
                indexed_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(project_id, file_path, qualified_name)
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_method_bodies_hash "
            "ON ci_method_bodies(project_id, body_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_method_bodies_qual "
            "ON ci_method_bodies(project_id, qualified_name)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_method_bodies_file "
            "ON ci_method_bodies(project_id, file_path)"
        )

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
        return []
