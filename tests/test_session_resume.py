"""Tests for the `-c`/`--resume` session-continuation feature.

Covers the three pieces that make "continue yesterday's work" cheap in
infinidev: the sessions registry (find the last session), persisted
session notes (survive process exit), and the one-shot full-history
replay (model sees the whole prior conversation exactly once on resume).
"""

from infinidev.db.service import (
    get_all_turns,
    get_last_session,
    get_session_messages,
    get_session_notes,
    get_session_runtime_state,
    list_recent_sessions,
    persist_session_note,
    persist_session_runtime_state,
    persist_staged_planning_state,
    register_session,
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
            "system",
        ]
        texts = [message["text"] for message in app.chat_messages]
        assert texts.count("legacy duplicate") == 1
        assert texts.index("older legacy request") < texts.index("legacy duplicate")
        assert app.chat_messages[2]["running"] is False
        assert "Interrupted" in app.chat_messages[2]["error"]
        assert app._steps_text == "v Inspect\n> Implement"
        assert app._actions_text == "Idle"
        assert app._restoring_session is False

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
        history.scroll_home()
        top = history.create_content(width=80, height=24)
        assert top.cursor_position.y == 0


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
                "result": "complete tool result",
                "running": False,
            },
        )
        persist_session_runtime_state(
            "resume-state",
            task_description="Original task description",
            plan_steps=[{"title": "Inspect state", "status": "done"}],
        )

        ca.request_full_history_once("resume-state")
        first = ca._build_user_message("continue", "resume-state")
        second = ca._build_user_message("another turn", "resume-state")

        assert isinstance(first, str)
        assert "<resumed-session-state>" in first
        assert "Original task description" in first
        assert "Inspect state" in first
        assert "read_file" in first
        assert "complete tool result" in first
        assert isinstance(second, str)
        assert "<resumed-session-state>" not in second
