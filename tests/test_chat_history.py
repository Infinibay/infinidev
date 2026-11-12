"""Chat-history persistence, retrieval, and session retention."""

from __future__ import annotations

import json

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import (
    delete_session,
    get_session_message_window,
    prune_old_sessions,
    register_session,
    search_session_messages,
    store_session_message,
)


@pytest.fixture
def db(temp_db):
    """Initialised temporary database with settings.DB_PATH patched."""
    return temp_db


def _sessions() -> list[str]:
    return [
        row["session_id"]
        for row in _rows(
            "SELECT session_id FROM sessions ORDER BY last_active_at DESC, rowid DESC"
        )
    ]


class TestTranscriptSearchText:
    def test_search_text_is_projected_from_message_fields(self, db):
        register_session("s1", "/work")
        message_id = store_session_message(
            "s1",
            {
                "role": "assistant",
                "content": "we should use uv run pytest",
                "tool": "execute_command",
                "args": {"command": "uv run pytest -q"},
            },
        )
        hits = search_session_messages("s1", "uv run pytest")
        assert [hit["id"] for hit in hits] == [message_id]
        assert "execute_command" in hits[0]["snippet"]

    def test_every_token_must_match(self, db):
        register_session("s1", "/work")
        first = store_session_message("s1", {"role": "user", "content": "alpha beta"})
        store_session_message("s1", {"role": "user", "content": "alpha gamma"})
        assert [hit["id"] for hit in search_session_messages("s1", "alpha beta")] == [first]
        assert len(search_session_messages("s1", "alpha")) == 2

    def test_like_wildcards_are_literal(self, db):
        register_session("s1", "/work")
        store_session_message("s1", {"role": "user", "content": "100% sure"})
        assert search_session_messages("s1", "100% sure")
        assert search_session_messages("s1", "100x sure") == []

    def test_search_text_is_redacted_like_the_payload(self, db, monkeypatch):
        import infinidev.config.secrets as secrets

        monkeypatch.setattr(secrets, "configured_secrets", lambda: ["super-secret-token"])
        monkeypatch.setattr(secrets, "mask_secret", lambda value: "[REDACTED]")
        register_session("s1", "/work")
        store_session_message(
            "s1", {"role": "user", "content": "use super-secret-token here"}
        )
        assert search_session_messages("s1", "super-secret-token") == []
        assert "[REDACTED]" in search_session_messages("s1", "use")[0]["snippet"]

    def test_empty_query_returns_nothing(self, db):
        register_session("s1", "/work")
        store_session_message("s1", {"role": "user", "content": "hello"})
        assert search_session_messages("s1", "") == []
        assert search_session_messages("", "hello") == []


def _rows(sql: str, params: tuple = ()) -> list[dict]:
    import sqlite3

    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


class TestMessageWindow:
    def test_window_returns_anchor_with_neighbours(self, db):
        register_session("s1", "/work")
        ids = [
            store_session_message("s1", {"role": "user", "content": f"msg {index}"})
            for index in range(6)
        ]
        window = get_session_message_window("s1", ids[2], before=2, after=1)
        assert [row["id"] for row in window] == ids[0:4]
        anchors = [row for row in window if row["is_anchor"]]
        assert [row["id"] for row in anchors] == [ids[2]]

    def test_window_is_clipped_at_the_transcript_edges(self, db):
        register_session("s1", "/work")
        ids = [
            store_session_message("s1", {"role": "user", "content": f"msg {index}"})
            for index in range(2)
        ]
        window = get_session_message_window("s1", ids[0], before=5, after=5)
        assert [row["id"] for row in window] == ids

    def test_window_does_not_leak_across_sessions(self, db):
        register_session("s1", "/work")
        register_session("s2", "/work")
        store_session_message("s1", {"role": "user", "content": "one"})
        other = store_session_message("s2", {"role": "user", "content": "two"})
        assert [row["id"] for row in get_session_message_window("s1", other)] == []


class TestSessionRetention:
    def test_registering_a_session_keeps_only_the_newest(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 5)
        for index in range(7):
            register_session(f"s{index}", "/work")
        assert _sessions() == ["s6", "s5", "s4", "s3", "s2"]

    def test_pruning_removes_owned_rows_not_just_the_registry(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 1)
        register_session("old", "/work")
        store_session_message("old", {"role": "user", "content": "old chat"})
        register_session("new", "/work")
        assert _rows("SELECT id FROM session_messages WHERE session_id = 'old'") == []
        assert search_session_messages("old", "old chat") == []

    def test_other_workspaces_are_untouched(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 1)
        register_session("a1", "/work/a")
        register_session("a2", "/work/a")
        register_session("b1", "/work/b")
        assert sorted(_sessions()) == ["a2", "b1"]

    def test_resumed_session_survives_its_own_prune(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 2)
        register_session("s1", "/work")
        register_session("s2", "/work")
        register_session("s3", "/work")
        # Resuming the oldest survivor must not delete it while registering.
        register_session("s1", "/work")
        assert sorted(_sessions()) == ["s1", "s3"]

    def test_retention_count_zero_disables_pruning(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 0)
        for index in range(4):
            register_session(f"s{index}", "/work")
        assert len(_sessions()) == 4

    def test_prune_is_idempotent_and_reports_nothing_when_bounded(self, db):
        register_session("s1", "/work")
        assert prune_old_sessions(keep=5, workspace_path="/work") == []

    def test_prune_deletes_sessions_without_a_workspace_row(self, db):
        register_session("orphan-1", None)
        register_session("orphan-2", None)
        register_session("in-work", "/work")
        assert prune_old_sessions(keep=1, workspace_path=None) == ["orphan-1"]
        assert "in-work" in _sessions()

    def test_registering_without_a_workspace_prunes_nothing(self, db, monkeypatch):
        # A session whose workspace is unknown has no safe retention scope.
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 1)
        register_session("unknown-1", None)
        register_session("unknown-2", None)
        assert sorted(_sessions()) == ["unknown-1", "unknown-2"]

    def test_resume_prunes_only_its_own_workspace(self, db, monkeypatch):
        # session_resume.begin_resumed_session registers without a path, so the
        # scope must come from the session row, not from the launch directory.
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 2)
        for session_id, workspace in (
            ("a1", "/work/a"), ("a2", "/work/a"), ("a3", "/work/a"),
            ("b1", "/work/b"), ("b2", "/work/b"),
            ("n1", None), ("n2", None),
        ):
            register_session(session_id, workspace)
        # /work/a kept only its two newest; the other scopes are untouched.
        assert sorted(_sessions()) == ["a2", "a3", "b1", "b2", "n1", "n2"]

        register_session("a2")  # resume, no workspace argument
        # Pruning a2's own workspace deletes nothing (it is a survivor), and the
        # unrelated scopes — including the workspace-less ones — stay intact.
        assert sorted(_sessions()) == ["a2", "a3", "b1", "b2", "n1", "n2"]

    def test_delete_session_still_works(self, db, monkeypatch):
        monkeypatch.setattr(settings, "SESSION_RETENTION_COUNT", 5)
        register_session("s1", "/work")
        store_session_message("s1", {"role": "user", "content": "hi"})
        assert delete_session("s1") is True
        assert _sessions() == []


class TestChatHistorySearchTool:
    @staticmethod
    def _payload(tool, **kwargs) -> dict:
        return json.loads(tool._run(**kwargs))

    def test_text_hit_returns_anchor_and_neighbours(self, db, bound_tool):
        from infinidev.tools.history import ChatHistorySearchTool

        register_session("test-session", "/work")
        ids = [
            store_session_message(
                "test-session",
                {"role": "user", "content": f"msg {index} about retention"},
            )
            for index in range(5)
        ]
        tool = bound_tool(ChatHistorySearchTool)
        out = self._payload(tool, query="about retention", before=1, after=1)

        assert out["mode"] == "text"
        assert out["count"] == 5
        anchor = out["results"][0]
        assert anchor["message_id"] == ids[4]
        assert [row["message_id"] for row in anchor["context"]] == ids[3:]
        assert [row["is_anchor"] for row in anchor["context"]] == [False, True]

    def test_message_id_reopens_a_known_message(self, db, bound_tool):
        from infinidev.tools.history import ChatHistorySearchTool

        register_session("test-session", "/work")
        ids = [
            store_session_message("test-session", {"role": "user", "content": f"m{index}"})
            for index in range(3)
        ]
        tool = bound_tool(ChatHistorySearchTool)
        out = self._payload(tool, message_id=ids[1], before=1, after=1)

        assert out["mode"] == "window"
        assert [row["message_id"] for row in out["results"][0]["context"]] == ids

    def test_semantic_mode_ranks_and_persists_vectors(self, db, bound_tool, monkeypatch):
        import numpy as np

        from infinidev.tools.history import ChatHistorySearchTool

        vectors = {
            "[user] we chose sqlite wal mode": np.array([1.0, 0.0]),
            "[user] the colour of the logo is blue": np.array([0.0, 1.0]),
            "we chose sqlite wal mode": np.array([1.0, 0.0]),
        }

        def embed(texts):
            return [vectors.get(text, np.array([0.3, 0.3])) for text in texts]

        monkeypatch.setattr("infinidev.tools.base.dedup._get_embed_fn", lambda: embed)
        register_session("test-session", "/work")
        store_session_message("test-session", {"role": "user", "content": "we chose sqlite wal mode"})
        store_session_message("test-session", {"role": "user", "content": "the colour of the logo is blue"})

        tool = bound_tool(ChatHistorySearchTool)
        out = self._payload(tool, query="we chose sqlite wal mode", mode="semantic", threshold=0.9)

        assert out["mode"] == "semantic"
        assert out["count"] == 1
        assert "sqlite" in out["results"][0]["snippet"]
        assert out["results"][0]["similarity"] == pytest.approx(1.0, abs=1e-4)
        stored = _rows(
            "SELECT embedding, embedding_space FROM session_messages ORDER BY id"
        )
        assert all(row["embedding"] is not None for row in stored)
        assert len({row["embedding_space"] for row in stored}) == 1

    def test_missing_query_is_an_error(self, db, bound_tool):
        from infinidev.tools.history import ChatHistorySearchTool

        register_session("test-session", "/work")
        tool = bound_tool(ChatHistorySearchTool)
        assert "error" in tool._run(query="").lower()

    def test_tool_is_registered_and_read_only(self):
        from infinidev.tools import get_tools_for_role

        names = {
            tool.name: tool
            for tool in get_tools_for_role("developer", supports_vision=False)
        }
        assert "chat_history_search" in names
        assert names["chat_history_search"].is_read_only is True
        chat_names = {
            tool.name for tool in get_tools_for_role("chat_agent", supports_vision=False)
        }
        assert "chat_history_search" in chat_names


class TestClassicModeTranscript:
    def test_classic_turns_land_in_the_searchable_ledger(self, db):
        from infinidev.cli.main import _record_classic_transcript

        register_session("classic", "/work")
        _record_classic_transcript("classic", "user", "why did we pick WAL mode")
        _record_classic_transcript("classic", "assistant", "because of concurrent readers")
        _record_classic_transcript("classic", "user", "")

        hits = search_session_messages("classic", "WAL mode")
        assert len(hits) == 1
        assert len(search_session_messages("classic", "concurrent readers")) == 1
        # An empty turn must not create a row.
        assert len(_rows("SELECT id FROM session_messages WHERE session_id = 'classic'")) == 2


class TestSchemaMigration:
    def test_legacy_rows_are_backfilled_into_the_search_index(self, db):
        # A row written before the feature existed: payload only, no text.
        register_session("legacy", "/work")
        from infinidev.tools.base.db import execute_with_retry

        def _seed(conn):
            conn.execute(
                "INSERT INTO session_messages (session_id, message_json, search_text) "
                "VALUES ('legacy', '{\"role\": \"user\", \"content\": \"decided to ship 0.7\"}', '')"
            )
            # A payload with no textual fields must not stall the backfill.
            conn.execute(
                "INSERT INTO session_messages (session_id, message_json, search_text) "
                "VALUES ('legacy', '{\"type\": \"spinner\"}', '')"
            )
            conn.commit()

        execute_with_retry(_seed)
        assert search_session_messages("legacy", "ship 0.7") == []

        from infinidev.db.service import init_db

        init_db()
        hits = search_session_messages("legacy", "ship 0.7")
        assert len(hits) == 1
        assert "decided to ship" in hits[0]["snippet"]

    def test_legacy_database_gains_the_retrieval_columns(self, db):
        # Simulate a pre-feature DB: the columns are back-filled by the next
        # init without losing existing rows.
        register_session("s1", "/work")
        store_session_message("s1", {"role": "user", "content": "before"})
        from infinidev.db.service import init_db

        init_db()
        columns = {row["name"] for row in _rows("PRAGMA table_info(session_messages)")}
        assert {"search_text", "embedding", "embedding_space"} <= columns
        # The pre-existing row keeps its payload and becomes searchable.
        assert json.loads(_rows("SELECT message_json FROM session_messages")[0]["message_json"])
        assert search_session_messages("s1", "before")


class TestConversationRetrievalRegressions:
    def test_text_search_matches_spanish_case_insensitively(self, db):
        register_session("s1", "/work")
        message_id = store_session_message(
            "s1", {"role": "user", "content": "Miércoles cerrado, reparación rápida"}
        )
        assert [row["id"] for row in search_session_messages("s1", "MIÉRCOLES")] == [message_id]
        assert search_session_messages("s1", "REPARACIÓN")

    def test_decision_after_long_preamble_is_searchable(self, db):
        register_session("s1", "/work")
        message_id = store_session_message(
            "s1", {"role": "user", "content": "intro " * 900 + "SQLite sin login"}
        )
        assert [row["id"] for row in search_session_messages("s1", "SQLite")] == [message_id]

    def test_anchor_excerpt_keeps_the_matching_decision(self, db, bound_tool):
        from infinidev.tools.history import ChatHistorySearchTool

        register_session("test-session", "/work")
        store_session_message(
            "test-session",
            {"role": "user", "content": "intro " * 250 + "SQLite, sin login. Miércoles cerrado."},
        )
        out = json.loads(bound_tool(ChatHistorySearchTool)._run(query="SQLite"))
        anchor = next(row for row in out["results"][0]["context"] if row["is_anchor"])
        assert "SQLite, sin login. Miércoles cerrado." in anchor["text"]

    def test_existing_truncated_projection_is_backfilled(self, db):
        from infinidev.db.service import _backfill_session_message_search_text
        from infinidev.tools.base.db import execute_with_retry

        register_session("s1", "/work")
        message_id = store_session_message(
            "s1", {"role": "user", "content": "intro " * 900 + "SQLite sin login"}
        )
        execute_with_retry(lambda conn: conn.execute(
            "UPDATE session_messages SET search_text = substr(search_text, 1, 4000) WHERE id = ?",
            (message_id,),
        ))
        execute_with_retry(_backfill_session_message_search_text)
        assert search_session_messages("s1", "SQLite")

    def test_edit_invalidates_cached_semantic_vector(self, db):
        from infinidev.db.service import store_session_message_embeddings

        register_session("s1", "/work")
        message_id = store_session_message("s1", {"role": "user", "content": "old"})
        store_session_message_embeddings([(message_id, b"cached-vector")], "old-space")
        store_session_message("s1", {"role": "user", "content": "new"}, message_id=message_id)

        row = _rows("SELECT embedding, embedding_space FROM session_messages")[0]
        assert row["embedding"] is None
        assert row["embedding_space"] == ""

    def test_unicode_casefold_expansion_preserves_match_in_excerpt(self, db, bound_tool):
        from infinidev.tools.history import ChatHistorySearchTool

        register_session("test-session", "/work")
        store_session_message("test-session", {
            "role": "user", "content": "intro " * 250 + "Straße 42 es la dirección elegida.",
        })
        out = json.loads(bound_tool(ChatHistorySearchTool)._run(query="STRASSE"))
        assert "Straße 42" in out["results"][0]["snippet"]
        anchor = next(row for row in out["results"][0]["context"] if row["is_anchor"])
        assert "Straße 42" in anchor["text"]
