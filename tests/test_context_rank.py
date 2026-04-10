"""Tests for the ContextRank system — logger, ranker, and prompt integration."""

from __future__ import annotations

import time

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import init_db
from infinidev.code_intel._db import execute_with_retry


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def cr_db(temp_db):
    """Temp database with ContextRank tables initialized."""
    return temp_db


# ── Logger tests ──────────────────────────────────────────────────────


class TestInteractionLogger:
    def test_log_context_stores_row(self, cr_db):
        from infinidev.engine.context_rank.logger import log_context

        ctx_id = log_context("sess-1", "task-1", "task_input", "fix auth bug")
        assert ctx_id is not None

        def _check(conn):
            row = conn.execute(
                "SELECT * FROM cr_contexts WHERE id = ?", (ctx_id,),
            ).fetchone()
            return row
        row = execute_with_retry(_check)
        assert row["context_type"] == "task_input"
        assert row["content"] == "fix auth bug"
        assert row["session_id"] == "sess-1"
        assert row["task_id"] == "task-1"
        assert row["iteration"] is None  # task_input has no iteration

    def test_log_context_with_step(self, cr_db):
        from infinidev.engine.context_rank.logger import log_context

        ctx_id = log_context(
            "sess-1", "task-1", "step_title",
            "Read auth middleware", iteration=2, step_index=1,
        )

        def _check(conn):
            return conn.execute(
                "SELECT * FROM cr_contexts WHERE id = ?", (ctx_id,),
            ).fetchone()
        row = execute_with_retry(_check)
        assert row["context_type"] == "step_title"
        assert row["iteration"] == 2
        assert row["step_index"] == 1

    def test_log_interaction_stores_row(self, cr_db):
        from infinidev.engine.context_rank.logger import log_interaction

        log_interaction(
            "sess-1", "task-1", None, 0,
            "file_read", "src/auth.py", "file", 1.0,
        )

        def _check(conn):
            rows = conn.execute(
                "SELECT * FROM cr_interactions WHERE target = 'src/auth.py'",
            ).fetchall()
            return rows
        rows = execute_with_retry(_check)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "file_read"
        assert rows[0]["weight"] == 1.0

    def test_log_tool_call_classifies_correctly(self, cr_db):
        from infinidev.engine.context_rank.logger import log_tool_call

        log_tool_call(
            "sess-1", "task-1", None, 0,
            "replace_lines", {"path": "src/auth.py", "start": 1, "end": 5},
        )

        def _check(conn):
            return conn.execute(
                "SELECT * FROM cr_interactions WHERE target = 'src/auth.py'",
            ).fetchall()
        rows = execute_with_retry(_check)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "file_write"
        assert rows[0]["weight"] == 2.0

    def test_log_tool_call_ignores_untracked_tools(self, cr_db):
        from infinidev.engine.context_rank.logger import log_tool_call

        log_tool_call(
            "sess-1", "task-1", None, 0,
            "step_complete", {"summary": "done"},
        )

        def _check(conn):
            return conn.execute(
                "SELECT COUNT(*) as cnt FROM cr_interactions",
            ).fetchone()
        row = execute_with_retry(_check)
        assert row["cnt"] == 0

    def test_classify_tool_call_all_types(self):
        from infinidev.engine.context_rank.logger import classify_tool_call

        # File read
        result = classify_tool_call("read_file", {"path": "a.py"})
        assert result == ("file_read", "a.py", "file", 1.0)

        # Symbol write
        result = classify_tool_call("edit_symbol", {"qualified_name": "Foo.bar"})
        assert result == ("symbol_write", "Foo.bar", "symbol", 2.5)

        # Finding
        result = classify_tool_call("record_finding", {"topic": "auth uses RS256"})
        assert result == ("finding_create", "auth uses RS256", "finding", 1.5)

        # Untracked
        result = classify_tool_call("think", {"text": "hmm"})
        assert result is None

    def test_snapshot_session_scores(self, cr_db):
        from infinidev.engine.context_rank.logger import (
            log_interaction, snapshot_session_scores,
        )

        # Log a few interactions
        log_interaction("sess-1", "task-1", None, 0, "file_read", "a.py", "file", 1.0)
        log_interaction("sess-1", "task-1", None, 1, "file_write", "a.py", "file", 2.0)
        log_interaction("sess-1", "task-1", None, 1, "file_read", "b.py", "file", 1.0)

        snapshot_session_scores("sess-1", "task-1")

        def _check(conn):
            return conn.execute(
                "SELECT target, score, access_count FROM cr_session_scores "
                "ORDER BY score DESC",
            ).fetchall()
        rows = execute_with_retry(_check)
        assert len(rows) == 2
        # a.py: 1.0 + 2.0 = 3.0
        assert rows[0]["target"] == "a.py"
        assert rows[0]["score"] == 3.0
        assert rows[0]["access_count"] == 2
        # b.py: 1.0
        assert rows[1]["target"] == "b.py"
        assert rows[1]["score"] == 1.0


# ── Ranker tests ──────────────────────────────────────────────────────


class TestContextRanker:
    def test_reactive_scoring(self, cr_db):
        from infinidev.engine.context_rank.logger import log_interaction
        from infinidev.engine.context_rank.ranker import _compute_reactive_scores

        # Log interactions at different iterations
        log_interaction("sess-1", "task-1", None, 0, "file_read", "old.py", "file", 1.0)
        log_interaction("sess-1", "task-1", None, 5, "file_write", "new.py", "file", 2.0)
        log_interaction("sess-1", "task-1", None, 5, "file_read", "new.py", "file", 1.0)

        scores = _compute_reactive_scores("sess-1", "task-1", current_iteration=5)
        assert "new.py" in scores
        assert "old.py" in scores
        # new.py should score higher (more recent, more weight, more accesses)
        assert scores["new.py"][0] > scores["old.py"][0]

    def test_alpha_adaptation(self):
        from infinidev.engine.context_rank.ranker import _compute_alpha

        # Iteration 0, no signal → pure prediction
        assert _compute_alpha(0, 0) == 0.0
        # Iteration 0, some signal → still mostly prediction
        assert _compute_alpha(0, 5) == 0.0
        # Iteration 8 with signal → capped at 0.85
        alpha = _compute_alpha(8, 10)
        assert 0.8 <= alpha <= 0.85
        # Few reactive signals halves alpha
        alpha_weak = _compute_alpha(4, 2)
        alpha_strong = _compute_alpha(4, 10)
        assert alpha_weak < alpha_strong

    def test_rank_with_no_data(self, cr_db):
        from infinidev.engine.context_rank.ranker import rank

        result = rank("fix something", "sess-1", "task-1", 0)
        assert result.empty

    def test_rank_reactive_only(self, cr_db):
        from infinidev.engine.context_rank.logger import log_interaction
        from infinidev.engine.context_rank.ranker import rank

        # Simulate tool calls
        log_interaction("sess-1", "task-1", None, 0, "file_read", "a.py", "file", 1.0)
        log_interaction("sess-1", "task-1", None, 1, "file_write", "a.py", "file", 2.0)
        log_interaction("sess-1", "task-1", None, 2, "symbol_write", "Foo.bar", "symbol", 2.5)

        result = rank("fix auth bug", "sess-1", "task-1", 3)
        # Should have files and symbols
        assert len(result.files) >= 1
        assert result.files[0].target == "a.py"
        assert len(result.symbols) >= 1
        assert result.symbols[0].target == "Foo.bar"

    def test_top_k_limits_output(self, cr_db):
        from infinidev.engine.context_rank.logger import log_interaction
        from infinidev.engine.context_rank.ranker import rank

        # Log many files
        for i in range(20):
            log_interaction(
                "sess-1", "task-1", None, 0,
                "file_read", f"file_{i}.py", "file", 1.0,
            )

        result = rank("test", "sess-1", "task-1", 1, top_k_files=3)
        assert len(result.files) <= 3


# ── Models tests ──────────────────────────────────────────────────────


class TestModels:
    def test_context_rank_result_empty(self):
        from infinidev.engine.context_rank.models import ContextRankResult
        result = ContextRankResult()
        assert result.empty

    def test_context_rank_result_not_empty(self):
        from infinidev.engine.context_rank.models import ContextRankResult, RankedItem
        result = ContextRankResult(
            files=[RankedItem(target="a.py", target_type="file", score=1.0)],
        )
        assert not result.empty


# ── Hooks tests ───────────────────────────────────────────────────────


class TestContextRankHooks:
    def test_hooks_disabled_by_default(self, cr_db):
        from infinidev.engine.context_rank.hooks import ContextRankHooks

        original = settings.CONTEXT_RANK_LOGGING_ENABLED
        settings.CONTEXT_RANK_LOGGING_ENABLED = False
        try:
            hooks = ContextRankHooks()
            hooks.start("sess-1", "task-1", "do something")
            assert not hooks._enabled

            # These should be no-ops
            hooks.on_step_activated("Read files", "Read all py files", 0, 0)
            hooks.on_tool_call("read_file", '{"path": "a.py"}', 0)
            hooks.finish()
        finally:
            settings.CONTEXT_RANK_LOGGING_ENABLED = original

    def test_hooks_enabled_logs_data(self, cr_db):
        from infinidev.engine.context_rank.hooks import ContextRankHooks

        original = settings.CONTEXT_RANK_LOGGING_ENABLED
        settings.CONTEXT_RANK_LOGGING_ENABLED = True
        try:
            hooks = ContextRankHooks()
            hooks.start("sess-1", "task-1", "fix auth bug")
            assert hooks._enabled
            assert hooks._task_context_id is not None

            hooks.on_tool_call("read_file", '{"path": "src/auth.py"}', 0)

            # Verify data was logged
            def _check(conn):
                return conn.execute(
                    "SELECT COUNT(*) as cnt FROM cr_interactions",
                ).fetchone()
            row = execute_with_retry(_check)
            assert row["cnt"] == 1
        finally:
            settings.CONTEXT_RANK_LOGGING_ENABLED = original

    def test_hooks_string_arguments_parsed(self, cr_db):
        from infinidev.engine.context_rank.hooks import ContextRankHooks

        original = settings.CONTEXT_RANK_LOGGING_ENABLED
        settings.CONTEXT_RANK_LOGGING_ENABLED = True
        try:
            hooks = ContextRankHooks()
            hooks.start("sess-1", "task-1", "test")
            # Arguments as JSON string (how they come from the LLM)
            hooks.on_tool_call("replace_lines", '{"path": "x.py", "start": 1}', 0)

            def _check(conn):
                row = conn.execute(
                    "SELECT * FROM cr_interactions WHERE target = 'x.py'",
                ).fetchone()
                return row
            row = execute_with_retry(_check)
            assert row is not None
            assert row["event_type"] == "file_write"
        finally:
            settings.CONTEXT_RANK_LOGGING_ENABLED = original


# ── Prompt rendering tests ────────────────────────────────────────────


class TestPromptRendering:
    def test_render_context_rank_empty(self):
        from infinidev.engine.loop.context import _render_context_rank
        from infinidev.engine.context_rank.models import ContextRankResult

        assert _render_context_rank(None) == ""
        assert _render_context_rank(ContextRankResult()) == ""

    def test_render_context_rank_with_data(self):
        from infinidev.engine.loop.context import _render_context_rank
        from infinidev.engine.context_rank.models import ContextRankResult, RankedItem

        result = ContextRankResult(
            files=[
                RankedItem(target="src/auth.py", target_type="file", score=4.2, reason="edited 3x"),
                RankedItem(target="src/tokens.py", target_type="file", score=2.1, reason="predicted"),
            ],
            symbols=[
                RankedItem(target="Auth.verify", target_type="symbol", score=3.0, reason="accessed 2x"),
            ],
        )

        rendered = _render_context_rank(result)
        assert "<context-rank>" in rendered
        assert "</context-rank>" in rendered
        assert "src/auth.py" in rendered
        assert "4.2" in rendered
        assert "Auth.verify" in rendered
        assert "Files (by relevance):" in rendered
        assert "Symbols:" in rendered
        # No findings section when empty
        assert "Findings:" not in rendered
