"""Tests for the small helpers introduced during the refactor session.

Grab-bag coverage of symbols that were extracted or added in commits
``6964f5c`` through ``c951eec``. Each class below targets one helper
and is kept small so the surface stays grep-able by symbol name.
"""

from __future__ import annotations

import logging

import pytest

# ── best_effort context manager ──────────────────────────────────────────


class TestBestEffort:
    """``engine._best_effort.best_effort`` swallows + logs."""

    def test_no_exception_is_transparent(self):
        """Normal execution inside the block is unaffected."""
        from infinidev.engine._best_effort import best_effort
        ran = []
        with best_effort("should not fire"):
            ran.append(1)
        assert ran == [1]

    def test_swallows_exception(self):
        """An exception inside the block does NOT propagate."""
        from infinidev.engine._best_effort import best_effort
        with best_effort("swallowed"):
            raise RuntimeError("boom")
        # If we reach here, the exception was swallowed.

    def test_logs_at_debug_by_default(self, caplog):
        """The swallowed exception is logged at DEBUG."""
        from infinidev.engine._best_effort import best_effort
        caplog.set_level(logging.DEBUG, logger="infinidev.engine._best_effort")
        with best_effort("op %s failed", "X"):
            raise ValueError("kaboom")
        # At least one DEBUG record referencing the formatted message.
        msgs = [r.getMessage() for r in caplog.records]
        assert any("op X failed" in m for m in msgs)

    def test_custom_log_level(self, caplog):
        """``level`` kwarg lets callers promote to WARNING / ERROR."""
        from infinidev.engine._best_effort import best_effort
        caplog.set_level(logging.WARNING, logger="infinidev.engine._best_effort")
        with best_effort("noisy op", level=logging.WARNING):
            raise RuntimeError("x")
        assert any(
            r.levelno == logging.WARNING for r in caplog.records
        ), "expected a WARNING-level record"


# ── ContextManager — expire_thinking + compact_for_small ────────────────


class TestContextManagerExpireThinking:
    """Assistant messages older than TTL get truncated."""

    def _assistant(self, content: str, age: int = 0) -> dict:
        return {"role": "assistant", "content": content, "_thinking_age": age}

    def test_short_content_never_truncated(self):
        from infinidev.engine.loop.context_manager import ContextManager
        msgs = [self._assistant("short", age=999)]
        ContextManager.expire_thinking(msgs)
        assert msgs[0]["content"] == "short"

    def test_long_content_under_ttl_is_preserved(self):
        from infinidev.engine.loop.context_manager import ContextManager
        long_text = "x" * 200
        msgs = [self._assistant(long_text, age=1)]
        ContextManager.expire_thinking(msgs, ttl=3)
        # Age becomes 2, still under TTL.
        assert msgs[0]["content"] == long_text
        assert msgs[0]["_thinking_age"] == 2

    def test_long_content_over_ttl_is_truncated(self):
        from infinidev.engine.loop.context_manager import ContextManager
        long_text = "first line\n" + ("x" * 400)
        msgs = [self._assistant(long_text, age=3)]
        ContextManager.expire_thinking(msgs, ttl=3)
        # Age becomes 4 > 3 → truncate.
        assert "[thinking truncated]" in msgs[0]["content"]
        assert "first line" in msgs[0]["content"]

    def test_non_assistant_messages_ignored(self):
        from infinidev.engine.loop.context_manager import ContextManager
        msgs = [
            {"role": "user", "content": "x" * 500},
            {"role": "tool", "content": "y" * 500, "tool_call_id": "t1"},
        ]
        ContextManager.expire_thinking(msgs, ttl=1)
        # No changes — neither is assistant.
        assert len(msgs[0]["content"]) == 500
        assert len(msgs[1]["content"]) == 500


class TestContextManagerCompact:
    """``compact_for_small`` shortens old tool results."""

    def test_preserves_system_and_first_user(self):
        from infinidev.engine.loop.context_manager import ContextManager
        msgs = [
            {"role": "system", "content": "S" * 500},
            {"role": "user", "content": "U" * 500},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
            {"role": "assistant", "content": "A3"},
        ]
        ContextManager.compact_for_small(msgs)
        # system + first user untouched
        assert msgs[0]["content"] == "S" * 500
        assert msgs[1]["content"] == "U" * 500

    def test_old_tool_results_truncated(self):
        from infinidev.engine.loop.context_manager import ContextManager
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "tool", "content": "T" * 500, "tool_call_id": "t1"},
            {"role": "assistant", "content": "A1"},
            {"role": "assistant", "content": "A2"},
        ]
        ContextManager.compact_for_small(msgs)
        assert "[truncated for context]" in msgs[2]["content"]
        assert len(msgs[2]["content"]) < 500

    def test_recent_tool_results_preserved(self):
        """Tool results after the cutoff (last 2 assistant rounds) stay intact."""
        from infinidev.engine.loop.context_manager import ContextManager
        msgs = [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": "A1"},
            {"role": "tool", "content": "T_recent" + "!" * 500, "tool_call_id": "t1"},
            {"role": "assistant", "content": "A2"},
        ]
        ContextManager.compact_for_small(msgs)
        # The tool result sits between the last two assistant turns —
        # it's after the cutoff, so it's untouched.
        assert "T_recent" in msgs[3]["content"]
        assert "[truncated" not in msgs[3]["content"]


# ── _is_error_result ─────────────────────────────────────────────────────


class TestIsErrorResult:
    """Guard used by every cache handler in ``tool_executor``."""

    def test_none_is_error(self):
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result(None) is True

    def test_empty_string_is_error(self):
        """Regression: empty strings must be treated as errors (bug #1 fix)."""
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result("") is True

    def test_dict_is_error(self):
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result({"error": "x"}) is True

    def test_error_json_string(self):
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result('{"error": "not found"}') is True

    def test_normal_content_is_not_error(self):
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result("line1\nline2") is False

    def test_success_json(self):
        from infinidev.engine.tool_executor import _is_error_result
        assert _is_error_result('{"status": "ok"}') is False


# ── _resolve_tool cascade ────────────────────────────────────────────────


class TestResolveTool:
    """Alias → case-insensitive → hallucination map fallback."""

    def _dispatch(self):
        """Minimal fake dispatch so we don't have to instantiate tools."""
        return {
            "read_file": "READ_FILE_TOOL",
            "create_file": "CREATE_FILE_TOOL",
            "execute_command": "EXEC_TOOL",
        }

    def test_direct_hit(self):
        from infinidev.engine.loop.tools import _resolve_tool
        tool, name = _resolve_tool(self._dispatch(), "read_file")
        assert tool == "READ_FILE_TOOL"
        assert name == "read_file"

    def test_alias_resolved(self):
        """Deprecated aliases map to their canonical names."""
        from infinidev.engine.loop.tools import _resolve_tool
        # write_file is a known alias for create_file.
        tool, name = _resolve_tool(self._dispatch(), "write_file")
        assert tool == "CREATE_FILE_TOOL"
        assert name == "create_file"

    def test_case_insensitive_match(self):
        from infinidev.engine.loop.tools import _resolve_tool
        tool, name = _resolve_tool(self._dispatch(), "READ_FILE")
        assert tool == "READ_FILE_TOOL"

    def test_hallucination_map_fallback(self):
        """Common small-model hallucinations route to the right tool."""
        from infinidev.engine.loop.tools import _resolve_tool
        tool, name = _resolve_tool(self._dispatch(), "run_command")
        assert tool == "EXEC_TOOL"
        assert name == "execute_command"

    def test_unknown_tool_returns_none(self):
        from infinidev.engine.loop.tools import _resolve_tool
        tool, name = _resolve_tool(self._dispatch(), "totally_made_up_tool")
        assert tool is None
        assert name == "totally_made_up_tool"


# ── context.py render helpers ────────────────────────────────────────────


class _FakeStep:
    def __init__(self, title, status="active", index=1, explanation=""):
        self.title = title
        self.status = status
        self.index = index
        self.explanation = explanation


class _FakePlan:
    def __init__(self, steps):
        self.steps = steps


class _FakeState:
    def __init__(self, *, notes=None, history=None, total_tool_calls=0,
                 tool_calls_since_last_note=0, steps=None):
        self.notes = notes or []
        self.history = history or []
        self.total_tool_calls = total_tool_calls
        self.tool_calls_since_last_note = tool_calls_since_last_note
        self.plan = _FakePlan(steps or [])


class TestRenderNoteNudge:
    """``_render_note_nudge`` guides the model to take notes."""

    def test_empty_when_below_thresholds(self):
        from infinidev.engine.loop.context import _render_note_nudge
        state = _FakeState(total_tool_calls=1)
        assert _render_note_nudge(state, small_model=False) == ""

    def test_reminder_when_many_calls_no_recent_note(self):
        from infinidev.engine.loop.context import _render_note_nudge
        state = _FakeState(
            tool_calls_since_last_note=4,
            total_tool_calls=5,
        )
        out = _render_note_nudge(state, small_model=False)
        assert "note-reminder" in out
        assert "add_note" in out

    def test_warning_when_history_exists_but_zero_notes(self):
        from infinidev.engine.loop.context import _render_note_nudge
        state = _FakeState(
            history=[object()],  # any non-empty sentinel
            total_tool_calls=5,
        )
        out = _render_note_nudge(state, small_model=False)
        assert "note-warning" in out
        assert "ZERO notes" in out

    def test_small_model_uses_terse_form(self):
        from infinidev.engine.loop.context import _render_note_nudge
        state = _FakeState(
            tool_calls_since_last_note=4,
            total_tool_calls=5,
        )
        out = _render_note_nudge(state, small_model=True)
        assert "SAVE NOTES NOW" in out
        # Small-model form should be compact — no XML block wrapping.
        assert "<note-reminder>" not in out


class TestRenderCurrentAction:
    """``_render_current_action`` renders the active step."""

    def test_small_model_terse(self):
        from infinidev.engine.loop.context import _render_current_action
        active = _FakeStep("do the thing", index=2, explanation="details here")
        state = _FakeState(steps=[active])
        out = _render_current_action(active, state, small_model=True)
        assert "DO NOW" in out
        assert "do the thing" in out
        assert "details here" in out

    def test_regular_model_includes_scope_constraint(self):
        from infinidev.engine.loop.context import _render_current_action
        active = _FakeStep("do the thing", index=2)
        pending = _FakeStep("later thing", index=3, status="pending")
        state = _FakeState(steps=[active, pending])
        out = _render_current_action(active, state, small_model=False)
        assert "SCOPE CONSTRAINT" in out
        assert "later thing" in out

    def test_regular_model_no_scope_when_nothing_pending(self):
        from infinidev.engine.loop.context import _render_current_action
        active = _FakeStep("last step", index=3)
        state = _FakeState(steps=[active])  # no pending
        out = _render_current_action(active, state, small_model=False)
        assert "SCOPE CONSTRAINT" not in out


# ── _migrate_add_column validation ───────────────────────────────────────


class TestMigrateAddColumnValidation:
    """SQL injection guard added in the B5 fix."""

    def test_rejects_bad_table_name(self):
        from infinidev.db.service import _migrate_add_column
        with pytest.raises(ValueError, match="invalid table name"):
            _migrate_add_column(None, "bad table; DROP", "col", "TEXT")

    def test_rejects_bad_column_name(self):
        from infinidev.db.service import _migrate_add_column
        with pytest.raises(ValueError, match="invalid column name"):
            _migrate_add_column(None, "findings", "col-with-dash", "TEXT")

    def test_rejects_bad_type(self):
        from infinidev.db.service import _migrate_add_column
        with pytest.raises(ValueError, match="invalid column type"):
            _migrate_add_column(None, "findings", "col", "NOT_A_REAL_TYPE")

    def test_accepts_type_with_default_clause(self):
        """``TEXT DEFAULT 'x'`` passes because the base type is whitelisted."""
        from infinidev.db.service import _migrate_add_column
        # We can't actually execute against a None conn, so we expect
        # the validation to pass and then blow up on conn.execute. The
        # specific error we catch tells us validation let it through.
        with pytest.raises(AttributeError):
            _migrate_add_column(None, "findings", "col", "TEXT DEFAULT 'x'")
