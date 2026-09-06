"""Tests for the `-c`/`--resume` session-continuation feature.

Covers the three pieces that make "continue yesterday's work" cheap in
infinidev: the sessions registry (find the last session), persisted
session notes (survive process exit), and the one-shot resume checkpoint
(the model sees only bounded active work).
"""

import os

import pytest

from infinidev.db.service import (
    delete_session,
    get_all_turns,
    get_last_session,
    get_session_messages,
    get_session_notes,
    get_session_storage_bytes,
    get_session_runtime_state,
    list_recent_sessions,
    persist_loop_resume_checkpoint,
    persist_session_note,
    persist_session_runtime_state,
    persist_staged_planning_state,
    register_session,
    rename_session,
    store_conversation_turn,
    store_session_message,
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

    def test_recent_sessions_can_return_complete_history(self, temp_db):
        for index in range(25):
            session_id = f"session-{index}"
            register_session(session_id, "/work")
            store_conversation_turn(session_id, "user", f"task {index}")

        assert len(list_recent_sessions("/work")) == 20
        assert len(list_recent_sessions("/work", limit=None)) == 25

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

    def test_rename_session_persists_normalized_title(self, temp_db):
        register_session("s", "/work")
        store_conversation_turn("s", "user", "automatic title")

        assert rename_session("s", "  Release   planning\nnotes  ") is True
        assert get_last_session("/work")["title"] == "Release planning notes"
        assert rename_session("missing", "name") is False
        assert rename_session("s", "   ") is False

    def test_session_label_includes_compact_workspace_context(self, monkeypatch):
        from pathlib import Path

        from infinidev.cli.session_resume import session_label

        monkeypatch.setattr("pathlib.Path.home", lambda: Path("/home/dev"))
        row = {
            "title": "Release planning",
            "workspace_path": "/home/dev/projects/infinidev",
            "turn_count": 3,
            "last_active_at": None,
        }

        assert session_label(row) == (
            "Release planning  ·  ~/projects/infinidev  ·  3 turns  ·  ~0 B  ·  unknown"
        )

    def test_session_label_distinguishes_same_title_across_workspaces(self):
        from infinidev.cli.session_resume import session_label

        common = {"title": "Fix login", "turn_count": 1, "last_active_at": None}
        first = session_label({**common, "workspace_path": "/work/api"})
        second = session_label({**common, "workspace_path": "/work/web"})

        assert first != second
        assert "/work/api" in first
        assert "/work/web" in second

    def test_name_session_supports_active_session_naming(self, temp_db):
        from infinidev.cli.session_resume import name_session

        register_session("active", "/work")
        store_conversation_turn("active", "user", "automatic title")

        assert name_session("active", "  Release   train\nplanning  ") == (
            "Release train planning"
        )
        assert get_last_session("/work")["title"] == "Release train planning"
        assert name_session("active", "   ") is None

    def test_classic_session_command_names_active_session(self, monkeypatch, capsys):
        from infinidev.cli import commands

        calls: list[tuple[str, str]] = []
        monkeypatch.setattr(
            "infinidev.cli.session_resume.name_session",
            lambda session_id, title: calls.append((session_id, title)) or "Release train",
        )

        assert commands.handle_command(
            "/session Release train", session_id="active-session"
        ) is True

        assert calls == [("active-session", "Release train")]
        assert "Session named: Release train" in capsys.readouterr().out

    def test_classic_resume_carries_durable_display_name(self, monkeypatch):
        from infinidev.cli import main
        from infinidev.cli import session_resume

        chosen = {"session_id": "12345678-rest", "title": "Release train"}
        monkeypatch.setattr(session_resume, "resolve_continue_session", lambda: chosen)
        monkeypatch.setattr(
            session_resume,
            "begin_resumed_session",
            lambda _session_id: [("user", "continue")],
        )

        assert main._resolve_classic_session(True, False) == (
            "12345678-rest",
            [("user", "continue")],
            "Release train",
        )

    def test_classic_resume_uses_short_id_for_untitled_legacy_session(self, monkeypatch):
        from infinidev.cli import main
        from infinidev.cli import session_resume

        chosen = {"session_id": "12345678-rest", "title": None}
        monkeypatch.setattr(session_resume, "resolve_continue_session", lambda: chosen)
        monkeypatch.setattr(session_resume, "begin_resumed_session", lambda _session_id: [])

        assert main._resolve_classic_session(True, False)[2] == "12345678"

    def test_classic_resume_preserves_workspace_and_repaints_turns(
        self, temp_db, monkeypatch
    ):
        from infinidev.cli import main, session_resume

        register_session("classic-cross-workspace", "/original-workspace")
        store_conversation_turn("classic-cross-workspace", "user", "classic history")
        chosen = {
            "session_id": "classic-cross-workspace",
            "title": "Classic session",
        }
        monkeypatch.setattr(session_resume, "resolve_continue_session", lambda: chosen)

        session_id, turns, display_name = main._resolve_classic_session(True, False)

        assert session_id == "classic-cross-workspace"
        assert turns == [("user", "classic history")]
        assert display_name == "Classic session"
        assert get_last_session("/original-workspace")["session_id"] == session_id
        assert get_last_session(os.getcwd()) is None

    def test_tui_resume_preserves_workspace_and_restores_state(self, temp_db, monkeypatch):
        from infinidev.cli import session_resume
        from infinidev.ui.app import _resolve_tui_resume

        register_session("tui-cross-workspace", "/original-workspace")
        store_conversation_turn("tui-cross-workspace", "user", "tui history")
        persist_session_runtime_state(
            "tui-cross-workspace", task_description="Restored task"
        )
        chosen = {"session_id": "tui-cross-workspace", "title": "TUI session"}
        monkeypatch.setattr(session_resume, "resolve_continue_session", lambda: chosen)

        resumed = _resolve_tui_resume(True, False)

        assert resumed is not None
        assert resumed["session_id"] == "tui-cross-workspace"
        assert resumed["turns"] == [("user", "tui history")]
        assert resumed["state"]["task_description"] == "Restored task"
        assert resumed["display_name"] == "TUI session"
        assert get_last_session("/original-workspace")["session_id"] == resumed["session_id"]
        assert get_last_session(os.getcwd()) is None

    def test_storage_bytes_reflect_owned_payloads(self, temp_db):
        register_session("measured", "/work")
        baseline = get_session_storage_bytes("measured")

        store_conversation_turn("measured", "user", "x" * 2048)
        persist_session_note("measured", "y" * 1024)
        store_session_message(
            "measured", {"sender": "Tool", "type": "tool_call", "result": "z" * 4096}
        )

        measured = get_session_storage_bytes("measured")
        assert baseline > 0
        assert measured >= baseline + 2048 + 1024 + 4096
        assert get_session_storage_bytes("missing") == 0
        assert get_session_storage_bytes("") == 0

    def test_storage_bytes_batches_large_session_histories(self, temp_db):
        from infinidev.db.service import get_sessions_storage_bytes

        session_ids = [f"session-{index}" for index in range(1001)]
        for session_id in session_ids:
            register_session(session_id, "/work")

        totals = get_sessions_storage_bytes(session_ids)

        assert list(totals) == session_ids
        assert all(total > 0 for total in totals.values())

    def test_delete_session_removes_owned_rows_and_preserves_other_data(self, temp_db):
        from infinidev.tools.base.db import execute_with_retry

        register_session("remove", "/work")
        register_session("keep", "/work")
        for session_id in ("remove", "keep"):
            store_conversation_turn(session_id, "user", f"turn {session_id}")
            persist_session_note(session_id, f"note {session_id}")
            store_session_message(session_id, {"sender": "You", "text": session_id})
            persist_session_runtime_state(session_id, task_description=session_id)

        def _seed_related(conn):
            conn.execute(
                "INSERT INTO findings (project_id, session_id, topic, content) "
                "VALUES (1, 'remove', 'durable finding', 'project knowledge')"
            )
            conn.execute(
                "INSERT INTO artifacts (project_id, session_id, name, content) "
                "VALUES (1, 'remove', 'durable artifact', 'project output')"
            )
            conn.execute(
                "INSERT INTO objective_verdicts "
                "(project_id, session_id, kind, verdict) VALUES (1, 'remove', 'test', 'PASS')"
            )
            conn.execute(
                "INSERT INTO exploration_trees "
                "(project_id, session_id, problem, tree_json) "
                "VALUES (1, 'remove', 'problem', '{}')"
            )
            conn.execute(
                "INSERT INTO engine_runs (run_id, session_id, engine) "
                "VALUES ('remove-run', 'remove', 'task')"
            )
            conn.execute(
                "INSERT INTO execution_events "
                "(event_id, run_id, session_id, sequence, timestamp, event_type, payload_json) "
                "VALUES ('remove-event', 'remove-run', 'remove', 1, 1.0, 'start', '{}')"
            )
            conn.execute(
                "INSERT INTO graph_states "
                "(run_id, session_id, revision, version, updated_at) "
                "VALUES ('remove-run', 'remove', 1, 2, 1.0)"
            )
            conn.execute(
                "INSERT INTO graph_nodes "
                "(node_id, run_id, session_id, node_type, created_at, updated_at) "
                "VALUES ('node', 'remove-run', 'remove', 'task', 1.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO graph_edges "
                "(edge_id, run_id, source, target, edge_type, created_at) "
                "VALUES ('edge', 'remove-run', 'node', 'node', 'depends', 1.0)"
            )
            conn.execute(
                "INSERT INTO cr_contexts "
                "(task_id, session_id, context_type, content, created_at) "
                "VALUES ('task', 'remove', 'file', 'context', 1.0)"
            )
            context_id = conn.execute(
                "SELECT id FROM cr_contexts WHERE session_id = 'remove'"
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO cr_interactions "
                "(task_id, session_id, context_id, iteration, event_type, target, "
                "target_type, created_at) VALUES "
                "('task', 'remove', ?, 1, 'use', 'src/app.py', 'file', 1.0)",
                (context_id,),
            )
            conn.execute(
                "INSERT INTO cr_session_scores "
                "(task_id, session_id, target, target_type, score, created_at) "
                "VALUES ('task', 'remove', 'src/app.py', 'file', 1.0, 1.0)"
            )
            conn.execute(
                "INSERT INTO image_generation_operations "
                "(operation_id, session_id, request_json, request_fingerprint, provider, "
                "model, profile_version, status) VALUES "
                "('image-op', 'remove', '{}', 'fingerprint', 'provider', 'model', 1, 'done')"
            )
            conn.execute(
                "INSERT INTO image_generation_items (operation_id, item_index, status) "
                "VALUES ('image-op', 0, 'done')"
            )
            conn.commit()

        execute_with_retry(_seed_related)
        assert get_session_storage_bytes("remove") > 0

        assert delete_session("remove") is True
        assert delete_session("remove") is False
        assert get_session_storage_bytes("remove") == 0
        assert get_last_session("/work")["session_id"] == "keep"
        assert get_all_turns("keep") == [("user", "turn keep")]

        def _remaining(conn):
            owned_tables = (
                "sessions", "conversation_turns", "session_notes", "session_messages",
                "session_runtime_state", "objective_verdicts", "exploration_trees",
                "engine_runs", "execution_events", "graph_states",
                "graph_nodes", "graph_edges", "cr_contexts", "cr_interactions",
                "cr_session_scores",
                "image_generation_operations",
            )
            owned = {
                table: conn.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE session_id = ?", ("remove",)
                ).fetchone()[0]
                for table in owned_tables
                if "session_id" in {
                    row["name"] for row in conn.execute(f"PRAGMA table_info({table})")
                }
            }
            return {
                "owned": owned,
                "image_items": conn.execute(
                    "SELECT COUNT(*) FROM image_generation_items "
                    "WHERE operation_id = 'image-op'"
                ).fetchone()[0],
                "graph_edges": conn.execute(
                    "SELECT COUNT(*) FROM graph_edges WHERE run_id = 'remove-run'"
                ).fetchone()[0],
                "finding": conn.execute(
                    "SELECT COUNT(*) FROM findings WHERE session_id = 'remove'"
                ).fetchone()[0],
                "artifact": conn.execute(
                    "SELECT COUNT(*) FROM artifacts WHERE session_id = 'remove'"
                ).fetchone()[0],
            }

        remaining = execute_with_retry(_remaining)
        assert set(remaining["owned"].values()) == {0}
        assert remaining["image_items"] == 0
        assert remaining["graph_edges"] == 0
        assert remaining["finding"] == 1
        assert remaining["artifact"] == 1

    def test_delete_session_cleans_orphaned_legacy_rows(self, temp_db):
        from infinidev.tools.base.db import execute_with_retry

        execute_with_retry(lambda conn: conn.execute(
            "INSERT INTO conversation_turns (session_id, role, content) "
            "VALUES ('orphan', 'user', 'legacy payload')"
        ))

        assert get_session_storage_bytes("orphan") > 0
        assert delete_session("orphan") is True
        assert get_all_turns("orphan") == []

    def test_picker_renames_then_selects_any_session(self, temp_db):
        from infinidev.cli.session_resume import pick_recent_session
        from infinidev.tools.base.db import execute_with_retry

        register_session("older", "/other-workspace")
        store_conversation_turn("older", "user", "old title")
        register_session("newer", "/work")
        store_conversation_turn("newer", "user", "new title")
        execute_with_retry(lambda conn: conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            ("2026-05-30 09:00:00.000", "older"),
        ))
        answers = iter(["rename 2 Durable work", "2"])
        output: list[str] = []

        chosen = pick_recent_session(lambda _message: next(answers), output.append, "/work")

        assert chosen is not None
        assert chosen["session_id"] == "older"
        assert get_last_session("/work")["title"] == "new title"
        rows = {row["session_id"]: row for row in list_recent_sessions(None)}
        assert rows["older"]["title"] == "Durable work"
        assert "Session renamed." in output
        assert any("~" in line and " B" in line for line in output)

    def test_picker_delete_cancellation_is_non_destructive(self, temp_db):
        from infinidev.cli.session_resume import pick_recent_session

        register_session("keep", "/work")
        store_conversation_turn("keep", "user", "keep this work")
        answers = iter(["delete 1", "no", "1"])
        output: list[str] = []

        chosen = pick_recent_session(lambda _message: next(answers), output.append, "/work")

        assert chosen is not None
        assert chosen["session_id"] == "keep"
        assert get_last_session("/work")["session_id"] == "keep"
        assert "Deletion cancelled." in output

    def test_picker_confirmed_delete_refreshes_then_selects(self, temp_db):
        from infinidev.cli.session_resume import pick_recent_session
        from infinidev.tools.base.db import execute_with_retry

        register_session("remove", "/work")
        store_conversation_turn("remove", "user", "large session " + ("x" * 2048))
        register_session("keep", "/work")
        store_conversation_turn("keep", "user", "keep session")
        execute_with_retry(lambda conn: conn.execute(
            "UPDATE sessions SET last_active_at = ? WHERE session_id = ?",
            ("2026-05-30 09:00:00.000", "keep"),
        ))
        answers = iter(["delete 1", "yes", "1"])
        output: list[str] = []

        chosen = pick_recent_session(lambda _message: next(answers), output.append, "/work")

        assert chosen is not None
        assert chosen["session_id"] == "keep"
        assert get_session_storage_bytes("remove") == 0
        assert "Session deleted." in output
        listings = [line for line in output if line.startswith("  ")]
        assert any("KiB" in line for line in listings)
        assert sum("large session" in line for line in listings) == 1

    def test_picker_deleting_last_session_starts_fresh(self, temp_db):
        from infinidev.cli.session_resume import pick_recent_session

        register_session("only", "/work")
        store_conversation_turn("only", "user", "temporary work")
        answers = iter(["delete 1", "y"])
        output: list[str] = []

        chosen = pick_recent_session(lambda _message: next(answers), output.append, "/work")

        assert chosen is None
        assert get_last_session("/work") is None
        assert output[-2:] == [
            "Session deleted.",
            "No recent sessions remain; starting fresh.",
        ]


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

    def test_resume_preserves_original_workspace(self, temp_db):
        from infinidev.cli.session_resume import begin_resumed_session

        register_session("cross-workspace", "/original-workspace")
        store_conversation_turn("cross-workspace", "user", "continue elsewhere")

        begin_resumed_session("cross-workspace", "/launch-workspace")

        assert get_last_session("/original-workspace")["session_id"] == "cross-workspace"
        assert get_last_session("/launch-workspace") is None

    def test_resume_returns_every_turn_without_truncation(self, temp_db):
        from infinidev.cli.session_resume import begin_resumed_session

        register_session("complete-history", "/work")
        for index in range(205):
            store_conversation_turn(
                "complete-history",
                "user" if index % 2 == 0 else "assistant",
                f"turn {index}",
            )
        long_reply = "complete reply " + ("x" * 5000)
        store_conversation_turn("complete-history", "assistant", long_reply)

        turns = begin_resumed_session("complete-history", "/work")

        assert len(turns) == 206
        assert turns[0] == ("user", "turn 0")
        assert turns[204] == ("user", "turn 204")
        assert turns[-1] == ("assistant", long_reply)


class TestStructuredSessionState:
    @pytest.mark.parametrize("field", ["loop_resume", "staged_planning"])
    def test_checkpoint_writer_does_not_overwrite_a_newer_sidebar(
        self, temp_db, monkeypatch, field
    ):
        from infinidev.db import service

        register_session("concurrent-runtime", "/work")
        persist_session_runtime_state(
            "concurrent-runtime", task_description="Old task", ui_state={"text": "old"},
        )
        execute = service.execute_with_retry
        interleaved = False

        def update_sidebar_after_first_operation(fn, *args, **kwargs):
            nonlocal interleaved
            result = execute(fn, *args, **kwargs)
            if not interleaved:
                interleaved = True
                persist_session_runtime_state(
                    "concurrent-runtime",
                    task_description="Latest task",
                    plan_steps=[{"title": "Latest step"}],
                    ui_state={"text": "latest"},
                )
            return result

        monkeypatch.setattr(service, "execute_with_retry", update_sidebar_after_first_operation)
        snapshot = {"state": {"notes": ["Keep progress"]}}
        writer = (
            persist_loop_resume_checkpoint if field == "loop_resume"
            else persist_staged_planning_state
        )
        writer("concurrent-runtime", snapshot)

        runtime = get_session_runtime_state("concurrent-runtime")
        assert runtime["task_description"] == "Latest task"
        assert runtime["plan_steps"] == [{"title": "Latest step"}]
        assert runtime["ui_state"]["text"] == "latest"
        assert runtime["ui_state"][field] == snapshot

    def test_tool_call_is_updated_in_place_without_truncation(self, temp_db):
        register_session("s", "/work")
        message_id = store_session_message(
            "s",
            {
                "sender": "Tool",
                "type": "tool_call",
                "tool_name": "read_file",
                "args": {"path": "src/app.py"},
                "result": "",
                "running": True,
                "_live_output_partial": "not durable",
            },
        )
        store_session_message(
            "s",
            {
                "sender": "Tool",
                "type": "tool_call",
                "tool_name": "read_file",
                "args": {"path": "src/app.py"},
                "result": "x" * 5000,
                "running": False,
            },
            message_id=message_id,
        )

        messages = get_session_messages("s")

        assert len(messages) == 1
        assert messages[0]["result"] == "x" * 5000
        assert messages[0]["args"] == {"path": "src/app.py"}
        assert "_live_output_partial" not in messages[0]
        assert messages[0]["_resume_message_id"] == message_id

    def test_task_plan_and_sidebar_round_trip(self, temp_db):
        register_session("s", "/work")
        steps = [
            {"index": 1, "title": "Inspect", "status": "done"},
            {"index": 2, "title": "Implement", "status": "active"},
        ]
        persist_session_runtime_state(
            "s",
            task_description="Restore the complete session",
            plan_steps=steps,
            ui_state={
                "plan_text": "Step 2: Implement",
                "steps_text": "v Inspect\n> Implement",
                "touched_files": {"src/app.py": 2},
            },
        )

        state = get_session_runtime_state("s")

        assert state["task_description"] == "Restore the complete session"
        assert state["plan_steps"] == steps
        assert state["ui_state"]["touched_files"] == {"src/app.py": 2}

    def test_sidebar_updates_preserve_staged_planning_snapshot(self, temp_db):
        register_session("s", "/work")
        staged = {
            "status": "active",
            "goal": {"title": "Long Goal"},
            "stages": [{"number": 1, "status": "active"}],
        }
        persist_staged_planning_state("s", staged, task_description="Long Goal")

        persist_session_runtime_state(
            "s",
            task_description="Long Goal",
            plan_steps=[{"title": "Current Step"}],
            ui_state={"steps_text": "> Current Step"},
        )

        state = get_session_runtime_state("s")
        assert state["staged_planning"] == staged
        assert state["ui_state"]["staged_planning"] == staged

    def test_sidebar_persistence_does_not_erase_loop_checkpoint(self, temp_db):
        register_session("loop-checkpoint", "/work")
        checkpoint = {
            "version": 1,
            "task_key": "key",
            "terminal_status": "",
            "state": {"notes": ["keep me"]},
        }
        persist_loop_resume_checkpoint("loop-checkpoint", checkpoint)

        persist_session_runtime_state(
            "loop-checkpoint",
            task_description="Current task",
            plan_steps=[{"title": "Active", "status": "active"}],
            ui_state={"actions_text": "Working"},
        )

        runtime = get_session_runtime_state("loop-checkpoint")
        assert runtime["ui_state"]["loop_resume"] == checkpoint
        assert runtime["ui_state"]["actions_text"] == "Working"

    def test_resume_bundle_includes_messages_and_runtime_state(self, temp_db):
        from infinidev.cli.session_resume import resumed_session_state

        register_session("s", "/work")
        store_session_message(
            "s",
            {"sender": "Thinking", "text": "intermediate", "type": "think"},
        )
        persist_session_runtime_state(
            "s",
            task_description="Original task",
            plan_steps=[{"title": "One", "status": "done"}],
        )

        state = resumed_session_state("s")

        assert state["messages"][0]["type"] == "think"
        assert state["task_description"] == "Original task"
        assert state["plan_steps"] == [{"title": "One", "status": "done"}]

    def test_repaint_prefers_structured_messages_over_legacy_turns(self):
        from types import SimpleNamespace

        from infinidev.ui.app import InfinidevApp

        class _History:
            def invalidate_cache(self):
                pass

        app = SimpleNamespace(
            _resume_request={
                "display_name": "Release train",
                "turns": [
                    ("user", "older legacy request"),
                    ("user", "legacy duplicate"),
                ],
                "state": {
                    "messages": [
                        {
                            "sender": "You",
                            "text": "legacy duplicate",
                            "type": "user",
                        },
                        {
                            "sender": "Tool",
                            "text": "read_file",
                            "type": "tool_call",
                            "running": True,
                            "result": "",
                        },
                        {"sender": "Thinking", "text": "why", "type": "think"},
                        {
                            "sender": "Infinidev",
                            "text": "partial answer",
                            "type": "agent",
                            "streaming": True,
                        },
                    ],
                    "ui_state": {
                        "plan_text": "Step 2",
                        "steps_text": "v Inspect\n> Implement",
                        "touched_files": {"src/app.py": 1},
                    },
                },
            },
            session_id="session-123",
            chat_messages=[],
            _restoring_session=True,
            _chat_history_control=_History(),
            _plan_text="",
            _steps_text="",
            _actions_text="",
            _touched_files={},
        )

        def _add_message(sender, text, msg_type):
            app.chat_messages.append(
                {"sender": sender, "text": text, "type": msg_type}
            )

        app.add_message = _add_message
        InfinidevApp._repaint_resumed_history(app)

        assert [message["type"] for message in app.chat_messages] == [
            "user",
            "user",
            "tool_call",
            "think",
            "agent",
            "system",
        ]
        texts = [message["text"] for message in app.chat_messages]
        assert texts.count("legacy duplicate") == 1
        assert texts.index("older legacy request") < texts.index("legacy duplicate")
        assert app.chat_messages[2]["running"] is False
        assert "Interrupted" in app.chat_messages[2]["error"]
        assert app.chat_messages[4]["streaming"] is False
        assert "Interrupted" in app.chat_messages[4]["text"]
        assert "Resumed session Release train" in app.chat_messages[-1]["text"]
        assert app._steps_text == "v Inspect\n> Implement"
        assert app._actions_text == "Idle"
        assert app._restoring_session is False

    def test_repaint_preserves_repeated_legacy_turns_by_multiplicity(self):
        from types import SimpleNamespace

        from infinidev.ui.app import InfinidevApp

        class _History:
            def invalidate_cache(self):
                pass

        app = SimpleNamespace(
            _resume_request={
                "turns": [
                    ("user", "continue"),
                    ("assistant", "intermediate answer"),
                    ("user", "continue"),
                ],
                "state": {
                    "messages": [
                        {"sender": "You", "text": "continue", "type": "user"},
                    ],
                },
            },
            session_id="session-duplicates",
            chat_messages=[],
            _restoring_session=True,
            _chat_history_control=_History(),
            _plan_text="",
            _steps_text="",
            _actions_text="",
            _touched_files={},
        )

        def _add_message(sender, text, msg_type):
            app.chat_messages.append(
                {"sender": sender, "text": text, "type": msg_type}
            )

        app.add_message = _add_message
        InfinidevApp._repaint_resumed_history(app)

        assert [message["text"] for message in app.chat_messages[:-1]] == [
            "continue",
            "intermediate answer",
            "continue",
        ]

    def test_repaint_restores_runtime_state_without_transcript(self):
        from types import SimpleNamespace

        from infinidev.ui.app import InfinidevApp

        class _History:
            def invalidate_cache(self):
                pass

        app = SimpleNamespace(
            _resume_request={
                "display_name": "Runtime only",
                "turns": [],
                "state": {
                    "messages": [],
                    "ui_state": {
                        "plan_text": "Implement durable resume",
                        "steps_text": "> Restore state",
                        "staged_planning": {"status": "active"},
                        "touched_files": {"src/infinidev/ui/app.py": 2},
                    },
                },
            },
            session_id="session-runtime-only",
            chat_messages=[],
            _restoring_session=True,
            _chat_history_control=_History(),
            _plan_text="",
            _steps_text="",
            _actions_text="Running",
            _staged_planning={},
            _touched_files={},
        )

        def _add_message(sender, text, msg_type):
            app.chat_messages.append(
                {"sender": sender, "text": text, "type": msg_type}
            )

        app.add_message = _add_message
        InfinidevApp._repaint_resumed_history(app)

        assert app._plan_text == "Implement durable resume"
        assert app._steps_text == "> Restore state"
        assert app._staged_planning == {"status": "active"}
        assert app._touched_files == {"src/infinidev/ui/app.py": 2}
        assert app._actions_text == "Idle"
        assert app._restoring_session is False
        assert len(app.chat_messages) == 1
        assert "0 prior events restored" in app.chat_messages[0]["text"]

    def test_repaint_keeps_the_entire_chat_scrollable(self):
        from types import SimpleNamespace

        from infinidev.ui.app import InfinidevApp
        from infinidev.ui.controls.chat_history import ChatHistoryControl

        turns = [
            (
                "user" if index % 2 == 0 else "assistant",
                f"complete historical message {index}",
            )
            for index in range(240)
        ]
        chat_messages: list[dict] = []
        history = ChatHistoryControl(chat_messages)
        app = SimpleNamespace(
            _resume_request={"turns": turns, "state": {}},
            session_id="session-long",
            chat_messages=chat_messages,
            _restoring_session=True,
            _chat_history_control=history,
            _plan_text="",
            _steps_text="",
            _actions_text="",
            _touched_files={},
        )

        def _add_message(sender, text, msg_type):
            app.chat_messages.append(
                {"sender": sender, "text": text, "type": msg_type}
            )

        app.add_message = _add_message
        InfinidevApp._repaint_resumed_history(app)
        content = history.create_content(width=80, height=24)
        rendered = "".join(
            text
            for line in (history._line_cache or [])
            for _style, text in line
        )

        assert "complete historical message 0" in rendered
        assert "complete historical message 239" in rendered
        assert content.line_count > 24
        assert content.cursor_position.y == content.line_count - 1
        assert "Resumed session session-" in chat_messages[-1]["text"]
        history.scroll_home()
        top = history.create_content(width=80, height=24)
        assert top.cursor_position.y == 0


class TestResumeContextReplay:
    def test_replay_is_consumed_once(self):
        from infinidev.engine.orchestration import chat_agent as ca
        ca._RESUME_CONTEXT_ONCE.discard("S")  # isolate from other tests
        ca.request_resume_context_once("S")
        assert "S" in ca._RESUME_CONTEXT_ONCE
        # Simulate the build consuming it.
        ca._RESUME_CONTEXT_ONCE.discard("S")
        assert "S" not in ca._RESUME_CONTEXT_ONCE

    def test_request_ignores_empty_session(self):
        from infinidev.engine.orchestration import chat_agent as ca
        before = set(ca._RESUME_CONTEXT_ONCE)
        ca.request_resume_context_once("")
        assert set(ca._RESUME_CONTEXT_ONCE) == before

    def test_first_resumed_prompt_includes_structured_execution_state(self, temp_db):
        from infinidev.engine.orchestration import chat_agent as ca

        register_session("resume-state", "/work")
        store_conversation_turn("resume-state", "user", "original request")
        store_session_message(
            "resume-state",
            {
                "sender": "Tool",
                "type": "tool_call",
                "tool_name": "read_file",
                "args": {"path": "src/app.py"},
                "result": "partial tool result",
                "running": True,
            },
        )
        persist_session_runtime_state(
            "resume-state",
            task_description="Original task description",
            plan_steps=[{"title": "Inspect state", "status": "done"}],
        )

        ca.request_resume_context_once("resume-state")
        first = ca._build_user_message("continue", "resume-state")
        second = ca._build_user_message("another turn", "resume-state")

        assert isinstance(first, str)
        assert "<resumed-session-state>" in first
        assert "Original task description" in first
        assert "Inspect state" in first
        assert "read_file" in first
        assert "partial tool result" in first
        assert '"running":false' in first
        assert "Interrupted before the previous session closed." in first
        assert isinstance(second, str)
        assert "<resumed-session-state>" not in second


    def test_resume_uses_compact_conversation_tail_not_200_turn_replay(
        self, temp_db
    ):
        from infinidev.engine.orchestration import chat_agent as ca

        register_session("compact-tail", "/work")
        for index in range(20):
            store_conversation_turn("compact-tail", "user", f"turn-{index}")

        ca.request_resume_context_once("compact-tail")
        prompt = ca._build_user_message("continue", "compact-tail")

        assert isinstance(prompt, str)
        assert "turn-0" not in prompt
        assert "turn-13" not in prompt
        assert "turn-14" in prompt
        assert "turn-19" in prompt

    def test_oversized_visual_ledger_never_becomes_one_provider_block(
        self, temp_db
    ):
        from infinidev.engine.orchestration import chat_agent as ca

        register_session("large-resume", "/work")
        store_conversation_turn("large-resume", "user", "original task")
        for index in range(80):
            store_session_message(
                "large-resume",
                {
                    "sender": "Tool",
                    "type": "tool_call",
                    "tool_name": "execute_command",
                    "args": {"command": f"command-{index}"},
                    "result": "x" * 200_000 + f"-result-{index}",
                },
            )

        ca.request_resume_context_once("large-resume")
        prompt = ca._build_user_message("continue", "large-resume")

        assert isinstance(prompt, str)
        assert len(prompt.encode("utf-8")) < 1_000_000
        assert "result-79" in prompt
        assert "result-0" not in prompt

    def test_latest_step_marker_bounds_legacy_event_replay(self):
        from infinidev.engine.orchestration.chat_agent import _resume_event_window

        events, omitted = _resume_event_window([
            {"type": "step_checkpoint", "step_title": "Old step"},
            {"type": "tool_call", "result": "old result"},
            {"type": "step_checkpoint", "step_title": "Current step"},
            {"type": "tool_call", "result": "current result"},
        ])

        assert omitted == 0
        assert [event["type"] for event in events] == [
            "step_checkpoint", "tool_call",
        ]
        assert events[0]["step_title"] == "Current step"
        assert events[1]["result"] == "current result"
