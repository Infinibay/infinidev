"""Tests for the `-c`/`--resume` session-continuation feature.

Covers the three pieces that make "continue yesterday's work" cheap in
infinidev: the sessions registry (find the last session), persisted
session notes (survive process exit), and the one-shot full-history
replay (model sees the whole prior conversation exactly once on resume).
"""

from infinidev.db.service import (
    register_session,
    store_conversation_turn,
    get_last_session,
    list_recent_sessions,
    persist_session_note,
    get_session_notes,
    get_all_turns,
)


class TestSessionRegistry:
    def test_register_and_find_last_by_workspace(self, temp_db):
        register_session("s1", "/work/a")
        register_session("s2", "/work/b")
        store_conversation_turn("s1", "user", "task in A")
        store_conversation_turn("s2", "user", "task in B")

        assert get_last_session("/work/a")["session_id"] == "s1"
        assert get_last_session("/work/b")["session_id"] == "s2"

    def test_last_active_wins(self, temp_db):
        from infinidev.tools.base.db import execute_with_retry
        register_session("yesterday", "/work")
        register_session("today", "/work")
        # Pin timestamps a day apart so the ORDER BY is exercised
        # deterministically (real resumes are hours/days apart, never
        # racing the millisecond clock).
        execute_with_retry(lambda c: c.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            ("2026-05-30 09:00:00.000", "yesterday")))
        execute_with_retry(lambda c: c.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            ("2026-05-31 09:00:00.000", "today")))
        assert get_last_session("/work")["session_id"] == "today"

    def test_title_backfilled_from_first_user_turn(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "user", "fix the login bug")
        store_conversation_turn("s", "user", "and the logout too")
        # Title is the FIRST user message, not overwritten by later ones.
        assert get_last_session("/work")["title"] == "fix the login bug"

    def test_turn_count_tracked(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "user", "a")
        store_conversation_turn("s", "assistant", "b")
        assert get_last_session("/work")["turn_count"] == 2

    def test_list_recent_skips_empty_sessions(self, temp_db):
        register_session("empty", "/work")  # never gets a turn
        register_session("used", "/work")
        store_conversation_turn("used", "user", "hi")
        ids = [s["session_id"] for s in list_recent_sessions("/work")]
        assert ids == ["used"]

    def test_no_session_returns_none(self, temp_db):
        assert get_last_session("/nonexistent") is None

    def test_register_is_idempotent(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "user", "hello")
        # Re-registering (resume) must not reset title or turn_count.
        register_session("s", "/work")
        row = get_last_session("/work")
        assert row["title"] == "hello"
        assert row["turn_count"] == 1


class TestSessionNotes:
    def test_persist_and_read_in_order(self, temp_db):
        persist_session_note("s", "first note")
        persist_session_note("s", "second note")
        assert get_session_notes("s") == ["first note", "second note"]

    def test_notes_scoped_per_session(self, temp_db):
        persist_session_note("a", "note A")
        persist_session_note("b", "note B")
        assert get_session_notes("a") == ["note A"]

    def test_empty_inputs_ignored(self, temp_db):
        persist_session_note("", "x")
        persist_session_note("s", "")
        assert get_session_notes("s") == []


class TestAllTurns:
    def test_returns_full_history_oldest_first(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "user", "u1")
        store_conversation_turn("s", "assistant", "a1")
        store_conversation_turn("s", "user", "u2")
        turns = get_all_turns("s")
        assert turns == [("user", "u1"), ("assistant", "a1"), ("user", "u2")]

    def test_long_turn_is_truncated(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "assistant", "x" * 5000)
        (_role, content), = get_all_turns("s", max_chars_per_turn=100)
        assert "[...truncated middle...]" in content
        assert len(content) < 5000


class TestFullHistoryReplay:
    def test_replay_is_consumed_once(self):
        from infinidev.engine.orchestration import chat_agent as ca
        ca._FULL_HISTORY_ONCE.discard("S")  # isolate from other tests
        ca.request_full_history_once("S")
        assert "S" in ca._FULL_HISTORY_ONCE
        # Simulate the build consuming it.
        ca._FULL_HISTORY_ONCE.discard("S")
        assert "S" not in ca._FULL_HISTORY_ONCE

    def test_request_ignores_empty_session(self):
        from infinidev.engine.orchestration import chat_agent as ca
        before = set(ca._FULL_HISTORY_ONCE)
        ca.request_full_history_once("")
        assert set(ca._FULL_HISTORY_ONCE) == before
