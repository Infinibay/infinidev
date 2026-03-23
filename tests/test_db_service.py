"""Tests for DB service: execute_with_retry, CRUD, DBConnection."""

import sqlite3
from unittest.mock import MagicMock, patch

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import (
    init_db,
    execute_with_retry as service_retry,
    store_conversation_turn,
    get_recent_summaries,
)
from infinidev.tools.base.db import (
    DBConnection,
    execute_with_retry,
    sanitize_fts5_query,
    parse_query_or_terms,
)


# ── execute_with_retry ───────────────────────────────────────────────────────


class TestExecuteWithRetry:
    """Tests for retry logic."""

    def test_success_first_try(self, temp_db):
        """Normal operation returns result."""
        result = execute_with_retry(lambda c: c.execute("SELECT 1").fetchone()[0])
        assert result == 1

    def test_retry_on_locked(self, temp_db):
        """Transient lock triggers retry and eventual success."""
        call_count = 0

        def _flaky(conn):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise sqlite3.OperationalError("database is locked")
            return conn.execute("SELECT 1").fetchone()[0]

        result = execute_with_retry(_flaky, max_retries=5, base_delay=0.001)
        assert result == 1
        assert call_count == 3

    def test_max_retries_exceeded(self, temp_db):
        """All attempts fail raises final exception."""
        def _always_locked(conn):
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(sqlite3.OperationalError, match="locked"):
            execute_with_retry(_always_locked, max_retries=3, base_delay=0.001)

    def test_non_retryable_error_raises_immediately(self, temp_db):
        """Non-lock errors propagate without retry."""
        call_count = 0

        def _bad_sql(conn):
            nonlocal call_count
            call_count += 1
            raise sqlite3.OperationalError("no such table: nonexistent")

        with pytest.raises(sqlite3.OperationalError, match="no such table"):
            execute_with_retry(_bad_sql, max_retries=5, base_delay=0.001)
        assert call_count == 1  # No retries


# ── DBConnection context manager ────────────────────────────────────────────


class TestDBConnection:
    """Tests for DBConnection context manager."""

    def test_commits_on_success(self, temp_db):
        """Insert inside context manager persists after exit."""
        with DBConnection() as conn:
            conn.execute(
                "INSERT INTO projects (name, description) VALUES (?, ?)",
                ("test-proj", "desc"),
            )
        # Verify it persisted
        row = execute_with_retry(
            lambda c: c.execute("SELECT name FROM projects WHERE name = 'test-proj'").fetchone()
        )
        assert row is not None

    def test_rolls_back_on_error(self, temp_db):
        """Exception inside context manager rolls back changes."""
        try:
            with DBConnection() as conn:
                conn.execute(
                    "INSERT INTO projects (name, description) VALUES (?, ?)",
                    ("rollback-proj", "desc"),
                )
                raise ValueError("simulated error")
        except ValueError:
            pass

        row = execute_with_retry(
            lambda c: c.execute("SELECT name FROM projects WHERE name = 'rollback-proj'").fetchone()
        )
        assert row is None


# ── Findings ─────────────────────────────────────────────────────────────────


class TestFindings:
    """Tests for findings CRUD."""

    def test_insert_and_retrieve_finding(self, temp_db):
        """Insert a finding and retrieve it."""
        def _insert(conn):
            conn.execute(
                """INSERT INTO findings (project_id, topic, content, finding_type, confidence)
                   VALUES (1, 'test topic', 'test content', 'observation', 0.9)"""
            )
            conn.commit()

        execute_with_retry(_insert)

        row = execute_with_retry(
            lambda c: c.execute("SELECT * FROM findings WHERE topic = 'test topic'").fetchone()
        )
        assert row is not None
        assert dict(row)["content"] == "test content"

    def test_fts5_search_findings(self, temp_db):
        """FTS5 search matches findings by content."""
        def _insert(conn):
            conn.execute(
                """INSERT INTO findings (project_id, topic, content, finding_type)
                   VALUES (1, 'security audit', 'SQL injection vulnerability found', 'observation')"""
            )
            conn.commit()

        execute_with_retry(_insert)

        rows = execute_with_retry(
            lambda c: c.execute(
                "SELECT * FROM findings_fts WHERE findings_fts MATCH '\"SQL injection\"'"
            ).fetchall()
        )
        assert len(rows) >= 1

    def test_empty_project_returns_empty(self, temp_db):
        """No findings for project returns empty list."""
        rows = execute_with_retry(
            lambda c: c.execute("SELECT * FROM findings WHERE project_id = 999").fetchall()
        )
        assert len(rows) == 0


# ── New tables from bug fix ──────────────────────────────────────────────────


class TestNewTables:
    """Tests for tables added in the bug fix: artifact_changes, status_updates, branches."""

    def test_artifact_changes_table_exists(self, temp_db):
        """artifact_changes table is created by init_db."""
        row = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='artifact_changes'"
            ).fetchone()
        )
        assert row is not None

    def test_status_updates_table_exists(self, temp_db):
        """status_updates table is created by init_db."""
        row = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='status_updates'"
            ).fetchone()
        )
        assert row is not None

    def test_branches_table_exists(self, temp_db):
        """branches table is created by init_db."""
        row = execute_with_retry(
            lambda c: c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='branches'"
            ).fetchone()
        )
        assert row is not None

    def test_insert_artifact_change(self, temp_db):
        """Can insert into artifact_changes."""
        def _insert(conn):
            conn.execute(
                """INSERT INTO artifact_changes
                   (project_id, agent_run_id, file_path, action, before_hash, after_hash, size_bytes)
                   VALUES (1, 'run-1', '/test.py', 'created', NULL, 'abc123', 100)"""
            )
            conn.commit()

        execute_with_retry(_insert)
        row = execute_with_retry(
            lambda c: c.execute("SELECT * FROM artifact_changes WHERE file_path = '/test.py'").fetchone()
        )
        assert row is not None
        assert dict(row)["action"] == "created"


# ── FTS5 query sanitisation ──────────────────────────────────────────────────


class TestSanitizeFts5Query:
    """Tests for FTS5 query builder."""

    def test_simple_query(self):
        """hello world → quoted terms."""
        result = sanitize_fts5_query("hello world")
        assert '"hello"' in result
        assert '"world"' in result

    def test_or_query(self):
        """security | auth → OR-joined."""
        result = sanitize_fts5_query("security | auth")
        assert "OR" in result

    def test_quoted_phrase(self):
        """Quoted phrase preserved."""
        result = sanitize_fts5_query('"exact phrase"')
        assert '"exact phrase"' in result

    def test_empty_query(self):
        """Empty returns empty quoted string."""
        result = sanitize_fts5_query("")
        assert result == '""'


# ── parse_query_or_terms ─────────────────────────────────────────────────────


class TestParseQueryOrTerms:
    """Tests for OR-term splitting."""

    def test_pipe_split(self):
        """a | b → ['a', 'b']."""
        result = parse_query_or_terms("a | b")
        assert result == ["a", "b"]

    def test_or_split(self):
        """a OR b → ['a', 'b']."""
        result = parse_query_or_terms("a OR b")
        assert result == ["a", "b"]

    def test_single_term(self):
        """hello → ['hello']."""
        result = parse_query_or_terms("hello")
        assert result == ["hello"]

    def test_empty_returns_original(self):
        """Empty query returns list with empty string."""
        result = parse_query_or_terms("")
        assert result == [""]
