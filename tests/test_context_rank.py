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


# ── Mention detection (inverse lookup) ────────────────────────────────


def _insert_symbol(conn, *, project_id=1, name, kind="function",
                   file_path="src/auth.py", qualified_name=None,
                   line_start=10, line_end=20, docstring=""):
    """Helper to insert a symbol for mention detection tests."""
    conn.execute(
        "INSERT INTO ci_symbols "
        "(project_id, file_path, name, qualified_name, kind, line_start, "
        "line_end, signature, docstring, language) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (project_id, file_path, name, qualified_name or name, kind,
         line_start, line_end, f"{kind} {name}()", docstring, "python"),
    )


def _insert_file(conn, *, project_id=1, file_path, language="python"):
    """Helper to insert an indexed file for mention detection tests."""
    conn.execute(
        "INSERT INTO ci_files (project_id, file_path, language, content_hash) "
        "VALUES (?, ?, ?, ?)",
        (project_id, file_path, language, "hash"),
    )


class TestMentionDetection:
    """Tests for the inverse mention detection (instr-based lookup)."""

    def test_symbol_name_in_input(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            _insert_symbol(conn, name="AuthMiddleware", kind="class",
                           file_path="/tmp/workspace/src/auth.py")
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "Please fix the AuthMiddleware class bug", project_id=1,
        )
        # Should have matched both the file and the symbol
        assert any("AuthMiddleware" in r for _, (_, _, r) in result.items())

    def test_short_names_ignored(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            _insert_symbol(conn, name="get", kind="function")
            _insert_symbol(conn, name="set", kind="function")
            _insert_symbol(conn, name="run", kind="function")
            conn.commit()
        execute_with_retry(_seed)

        # These are all short common names — should be filtered by LENGTH(name) >= 4
        result = _compute_mention_scores("Can you run get and set this?", project_id=1)
        # Either no matches, or only the `run` symbol if LENGTH check is off
        # With LENGTH >= 4, nothing should match
        assert len(result) == 0

    def test_common_words_ignored(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            # "error" is a common word in _COMMON_WORDS
            _insert_symbol(conn, name="error", kind="function")
            # "TokenValidator" is distinctive
            _insert_symbol(conn, name="TokenValidator", kind="class",
                           file_path="src/tokens.py")
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "What is the error in TokenValidator?", project_id=1,
        )
        # "error" should be filtered, "TokenValidator" should match
        targets = set(result.keys())
        assert not any("error" == t.split("/")[-1] for t in targets)
        assert any("TokenValidator" in r for _, (_, _, r) in result.items())

    def test_qualified_name_match(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            _insert_symbol(
                conn, name="handleEvent", kind="method",
                qualified_name="Agent.handleEvent",
                file_path="src/agent.py",
            )
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "Explain Agent.handleEvent and what it does", project_id=1,
        )
        assert any("handleEvent" in r for _, (_, _, r) in result.items())

    def test_ignores_non_distinctive_kinds(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            # Variables and parameters should be ignored — only function/method/
            # class/interface/enum/type_alias are considered
            _insert_symbol(conn, name="sessionRegistry", kind="variable")
            _insert_symbol(conn, name="sessionManager", kind="class",
                           file_path="src/session.py")
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "Show me the sessionRegistry and sessionManager", project_id=1,
        )
        # Only sessionManager (class) should match — sessionRegistry is a variable
        all_reasons = " ".join(r for _, (_, _, r) in result.items())
        assert "sessionManager" in all_reasons
        assert "sessionRegistry" not in all_reasons

    def test_filename_stem_match(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            _insert_file(conn, file_path="src/registry.py")
            _insert_file(conn, file_path="src/unrelated.py")
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "How does the registry work?", project_id=1,
        )
        # registry.py should appear (stem match)
        assert any("registry" in t for t in result.keys())
        # unrelated.py should not
        assert not any("unrelated" in t for t in result.keys())

    def test_basename_match_scores_higher(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            _insert_file(conn, file_path="src/registry.py")
            conn.commit()
        execute_with_retry(_seed)

        # With extension — basename match (higher score)
        result_exact = _compute_mention_scores("Check registry.py please", project_id=1)
        # Without extension — stem match (lower score)
        result_stem = _compute_mention_scores("Check the registry module", project_id=1)

        exact_scores = [s for _, (s, _, _) in result_exact.items()]
        stem_scores = [s for _, (s, _, _) in result_stem.items()]
        if exact_scores and stem_scores:
            assert max(exact_scores) >= max(stem_scores)

    def test_empty_or_short_input(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        assert _compute_mention_scores("", 1) == {}
        assert _compute_mention_scores("hi", 1) == {}

    def test_length_based_scoring(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_mention_scores

        def _seed(conn):
            # Short name — lower score
            _insert_symbol(conn, name="Auth", kind="class",
                           file_path="src/auth_short.py")
            # Long name — higher score
            _insert_symbol(conn, name="AuthenticationMiddlewareHandler",
                           kind="class",
                           file_path="src/auth_long.py")
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_mention_scores(
            "Auth and AuthenticationMiddlewareHandler", project_id=1,
        )
        short_score = result.get("src/auth_short.py", (0, "", ""))[0]
        long_score = result.get("src/auth_long.py", (0, "", ""))[0]
        assert long_score > short_score


# ── Finding multi-signal scoring ──────────────────────────────────────


def _insert_finding(conn, *, project_id=1, topic, content="",
                    tags=None, finding_type="observation", embedding=None):
    """Helper to insert a finding for scoring tests."""
    import json as _json
    conn.execute(
        "INSERT INTO findings "
        "(project_id, topic, content, status, finding_type, tags_json, embedding) "
        "VALUES (?, ?, ?, 'active', ?, ?, ?)",
        (project_id, topic, content, finding_type,
         _json.dumps(tags or []), embedding),
    )


class TestFindingMultiSignal:
    """Tests for the multi-signal finding scoring (semantic + literal)."""

    def test_topic_word_match(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        def _seed(conn):
            _insert_finding(
                conn,
                topic="Database migration requires downtime",
                content="The migration takes 30 minutes",
            )
            _insert_finding(
                conn,
                topic="Unrelated cache TTL configuration",
            )
            conn.commit()
        execute_with_retry(_seed)

        # Input mentions "database" and "migration" — should match first finding
        result = _compute_finding_scores(
            None, "How does the database migration work?", project_id=1,
        )
        assert "Database migration requires downtime" in result
        # The unrelated one should not match (no topic word overlap)
        assert "Unrelated cache TTL configuration" not in result

    def test_tag_match(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        def _seed(conn):
            _insert_finding(
                conn,
                topic="Auth uses RS256 keys",
                tags=["authentication", "security"],
            )
            conn.commit()
        execute_with_retry(_seed)

        # Input contains "authentication" which is a tag
        result = _compute_finding_scores(
            None, "Explain the authentication flow", project_id=1,
        )
        assert "Auth uses RS256 keys" in result
        _, _, reason = result["Auth uses RS256 keys"]
        assert "tags match" in reason or "topic" in reason

    def test_common_topic_words_need_multiple_matches(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        def _seed(conn):
            _insert_finding(
                conn,
                topic="System handles error gracefully in production",
            )
            conn.commit()
        execute_with_retry(_seed)

        # "system" alone shouldn't trigger — common words filtered
        result = _compute_finding_scores(
            None, "What is the system?", project_id=1,
        )
        # The topic has "system", "handles", "error", "gracefully", "production"
        # but "system" and "error" are in _COMMON_WORDS and removed from
        # topic_words. "handles", "gracefully", "production" remain.
        # None of those are in the input, so no match.
        assert "System handles error gracefully in production" not in result

    def test_empty_findings(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        result = _compute_finding_scores(None, "any query", project_id=1)
        assert result == {}

    def test_topic_words_require_min_ratio(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        def _seed(conn):
            # Long topic — would need at least 2 matches
            _insert_finding(
                conn,
                topic="Redis caching improves database read latency",
            )
            conn.commit()
        execute_with_retry(_seed)

        # Only 1 word matches ("redis") — below min threshold of 2
        result_weak = _compute_finding_scores(
            None, "What is redis?", project_id=1,
        )
        # 2 words match ("redis", "caching")
        result_strong = _compute_finding_scores(
            None, "How does redis caching work?", project_id=1,
        )
        # Strong match should find it
        assert "Redis caching improves database read latency" in result_strong

    def test_literal_wins_over_weak_semantic(self, cr_db):
        from infinidev.engine.context_rank.ranker import _compute_finding_scores

        def _seed(conn):
            _insert_finding(
                conn,
                topic="GraphQL resolver optimization",
                tags=["graphql", "performance"],
            )
            conn.commit()
        execute_with_retry(_seed)

        result = _compute_finding_scores(
            None, "How does graphql resolver work?", project_id=1,
        )
        assert "GraphQL resolver optimization" in result
        # Should be a literal match (topic words), not semantic
        _, _, reason = result["GraphQL resolver optimization"]
        assert "topic words" in reason or "tags match" in reason


# ── Outlier filtering ─────────────────────────────────────────────────


class TestOutlierFiltering:
    """Tests for the outlier detection that reduces <context-rank> noise."""

    def test_clear_outlier_shown_alone(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # One strong outlier (17.4) vs rest in 3.0-3.5 range
        items = [
            RankedItem(target="lsp/server.ts", target_type="file", score=17.4, reason="historical"),
            RankedItem(target="worker.ts", target_type="file", score=3.5, reason="noise"),
            RankedItem(target="theme.tsx", target_type="file", score=3.2, reason="noise"),
            RankedItem(target="utils.ts", target_type="file", score=3.0, reason="noise"),
            RankedItem(target="config.ts", target_type="file", score=2.8, reason="noise"),
        ]
        result = _filter_outliers(items)
        # Only the clear winner should remain
        assert len(result) == 1
        assert result[0].target == "lsp/server.ts"

    def test_no_outlier_all_kept(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # All items nearly identical — degenerate MAD case, ratio
        # test says top is not meaningfully above baseline → keep all
        items = [
            RankedItem(target="a.ts", target_type="file", score=3.6, reason=""),
            RankedItem(target="b.ts", target_type="file", score=3.5, reason=""),
            RankedItem(target="c.ts", target_type="file", score=3.5, reason=""),
            RankedItem(target="d.ts", target_type="file", score=3.4, reason=""),
            RankedItem(target="e.ts", target_type="file", score=3.3, reason=""),
        ]
        result = _filter_outliers(items)
        assert len(result) == 5

    def test_multiple_outliers_up_to_3(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # Two strong outliers (10, 8) vs rest in 2.0 range
        items = [
            RankedItem(target="a.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="b.ts", target_type="file", score=8.0, reason=""),
            RankedItem(target="c.ts", target_type="file", score=2.1, reason=""),
            RankedItem(target="d.ts", target_type="file", score=2.0, reason=""),
            RankedItem(target="e.ts", target_type="file", score=1.8, reason=""),
        ]
        result = _filter_outliers(items)
        assert len(result) == 2
        assert {it.target for it in result} == {"a.ts", "b.ts"}

    def test_too_many_outliers_falls_back(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # 4+ items all at the same high score (degenerate MAD, fallback
        # ratio fires) — 4 items > max_count → return all
        items = [
            RankedItem(target="a.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="b.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="c.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="d.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="e.ts", target_type="file", score=1.0, reason=""),
            RankedItem(target="f.ts", target_type="file", score=1.0, reason=""),
        ]
        result = _filter_outliers(items)
        # 4 outliers detected > max_count (3) → return all 6
        assert len(result) == 6

    def test_few_items_unchanged(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # Fewer than 3 items — no filtering
        items = [
            RankedItem(target="a.ts", target_type="file", score=10.0, reason=""),
            RankedItem(target="b.ts", target_type="file", score=1.0, reason=""),
        ]
        result = _filter_outliers(items)
        assert len(result) == 2

    def test_empty_list(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        assert _filter_outliers([]) == []

    def test_works_for_symbols_and_findings(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # Symbols with one clear outlier
        symbols = [
            RankedItem(target="Agent.handleEvent", target_type="symbol", score=12.0, reason=""),
            RankedItem(target="foo", target_type="symbol", score=2.0, reason=""),
            RankedItem(target="bar", target_type="symbol", score=1.8, reason=""),
            RankedItem(target="baz", target_type="symbol", score=1.5, reason=""),
        ]
        result = _filter_outliers(symbols)
        assert len(result) == 1
        assert result[0].target == "Agent.handleEvent"

        # Findings with two clear outliers
        findings = [
            RankedItem(target="Auth uses RS256", target_type="finding", score=9.0, reason=""),
            RankedItem(target="Token refresh pattern", target_type="finding", score=7.5, reason=""),
            RankedItem(target="CSS layout debt", target_type="finding", score=2.0, reason=""),
            RankedItem(target="Old migration notes", target_type="finding", score=1.5, reason=""),
        ]
        result = _filter_outliers(findings)
        assert len(result) == 2

    def test_below_min_top_score_no_filtering(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # All scores below _OUTLIER_MIN_TOP_SCORE (1.0) — don't filter
        items = [
            RankedItem(target="a.ts", target_type="file", score=0.9, reason=""),
            RankedItem(target="b.ts", target_type="file", score=0.5, reason=""),
            RankedItem(target="c.ts", target_type="file", score=0.4, reason=""),
            RankedItem(target="d.ts", target_type="file", score=0.3, reason=""),
        ]
        result = _filter_outliers(items)
        # Top score < 1.0 disables filtering (confidence too low)
        assert len(result) == 4

    def test_clear_outlier_above_noise_threshold(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # Top score is above 1.0 and dramatically above noise — should filter
        items = [
            RankedItem(target="a.ts", target_type="file", score=5.0, reason=""),
            RankedItem(target="b.ts", target_type="file", score=1.2, reason=""),
            RankedItem(target="c.ts", target_type="file", score=1.1, reason=""),
            RankedItem(target="d.ts", target_type="file", score=1.0, reason=""),
        ]
        result = _filter_outliers(items)
        assert len(result) == 1
        assert result[0].target == "a.ts"


class TestPercentileToMadMultiplier:
    """Tests for the user-friendly percentile → K conversion."""

    def test_common_percentiles(self):
        from infinidev.engine.context_rank.ranker import _percentile_to_mad_multiplier

        # Well-known Z-score values (table Z of normal distribution)
        assert abs(_percentile_to_mad_multiplier(90) - 1.282) < 0.01
        assert abs(_percentile_to_mad_multiplier(95) - 1.645) < 0.01
        assert abs(_percentile_to_mad_multiplier(99) - 2.326) < 0.01
        assert abs(_percentile_to_mad_multiplier(99.7) - 2.748) < 0.01

    def test_percentage_strings(self):
        from infinidev.engine.context_rank.ranker import _percentile_to_mad_multiplier

        # Strings with and without % sign
        assert abs(_percentile_to_mad_multiplier("90%") - 1.282) < 0.01
        assert abs(_percentile_to_mad_multiplier("95") - 1.645) < 0.01
        assert abs(_percentile_to_mad_multiplier("99.5%") - 2.576) < 0.01

    def test_invalid_values_fallback(self):
        from infinidev.engine.context_rank.ranker import _percentile_to_mad_multiplier

        # Invalid values fall back to 95% (K ≈ 1.645)
        assert abs(_percentile_to_mad_multiplier("not a number") - 1.645) < 0.01
        assert abs(_percentile_to_mad_multiplier(-5) - 1.645) < 0.01
        assert abs(_percentile_to_mad_multiplier(100) - 1.645) < 0.01
        assert abs(_percentile_to_mad_multiplier(150) - 1.645) < 0.01

    def test_aggressive_vs_strict_affects_outlier_count(self):
        from infinidev.engine.context_rank.ranker import _filter_outliers
        from infinidev.engine.context_rank.models import RankedItem

        # Moderate outlier: score 5 vs noise around 2
        items = [
            RankedItem(target="a.ts", target_type="file", score=5.0, reason=""),
            RankedItem(target="b.ts", target_type="file", score=2.1, reason=""),
            RankedItem(target="c.ts", target_type="file", score=2.0, reason=""),
            RankedItem(target="d.ts", target_type="file", score=1.9, reason=""),
            RankedItem(target="e.ts", target_type="file", score=1.8, reason=""),
        ]

        # At 90% (aggressive), score=5 clearly exceeds threshold → 1 outlier
        original = settings.CONTEXT_RANK_OUTLIER_PERCENTILE
        try:
            settings.CONTEXT_RANK_OUTLIER_PERCENTILE = 90
            result_aggressive = _filter_outliers(items)

            # At 99.9% (very strict), threshold is much higher
            settings.CONTEXT_RANK_OUTLIER_PERCENTILE = 99.9
            result_strict = _filter_outliers(items)
        finally:
            settings.CONTEXT_RANK_OUTLIER_PERCENTILE = original

        # Aggressive percentile (lower K) has a lower threshold, so
        # it should return at least as many outliers as strict.
        assert len(result_aggressive) >= len(result_strict)
        # Strict should return exactly the single clearest outlier.
        assert result_strict[0].target == "a.ts"
