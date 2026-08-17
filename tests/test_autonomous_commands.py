"""Tests for the /auto slash command and its subcommands.

Covers the parent dispatch in src/infinidev/ui/handlers/commands.py:
  * ``/auto``                  — show usage
  * ``/auto <msg>``            — start autonomous chain with instructions
  * ``/auto pause``            — cancel the running engine, keep mode active
  * ``/auto stop``             — cancel the running engine, disable mode

The handlers are exercised against a fake app; the real engine worker is
replaced with a stub so the test does not actually launch the loop.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from infinidev.ui.handlers import commands


# ── Fake app ──────────────────────────────────────────────────────────────


class FakeChatHistoryControl:
    show_thinking = False

    def invalidate_cache(self) -> None:
        pass


class FakeApp:
    """Minimal stand-in for InfinidevApp used by the command handlers."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str, str]] = []
        self.chat_messages: list = []
        self._chat_history_control = FakeChatHistoryControl()
        self._log_entries: list = []
        self._plan_text = ""
        self._thinking_text = ""
        self._steps_text = ""
        self._actions_text = ""
        self._streaming_tool_name = None
        self._streaming_token_count = 0
        self._engine_running = False
        self._autonomous_active = False
        self._cancel_event = threading.Event()
        self._engine_calls: list[tuple] = []

    def add_message(self, speaker: str, msg: str, kind: str) -> None:
        self.messages.append((speaker, msg, kind))

    def invalidate(self) -> None:
        pass

    def _ensure_engine(self) -> None:
        pass


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
def app() -> FakeApp:
    return FakeApp()


@pytest.fixture
def patched_workers(monkeypatch: pytest.MonkeyPatch):
    """Patch run_in_background + run_engine_task so the engine never launches."""

    calls: list[tuple] = []

    def fake_run_in_background(target_app, fn, *args, **kwargs):
        name = getattr(fn, "__name__", None) or getattr(fn, "name", repr(fn))
        calls.append((name, args, kwargs))

    monkeypatch.setattr(
        "infinidev.ui.workers.run_in_background", fake_run_in_background,
    )

    def _worker_stub(*_args, **_kwargs):
        return None

    monkeypatch.setattr("infinidev.ui.workers.run_engine_task", _worker_stub)
    return calls


# ── Dispatch wiring ───────────────────────────────────────────────────────


def test_auto_command_is_registered() -> None:
    """The dispatch table must list /auto as the parent command."""
    assert "/auto" in commands._COMMAND_TABLE


def test_handle_command_routes_auto(app: FakeApp) -> None:
    """``handle_command`` must dispatch /auto to the auto handler."""
    sentinel = MagicMock()
    original = commands._COMMAND_TABLE["/auto"]
    commands._COMMAND_TABLE["/auto"] = sentinel
    try:
        commands.handle_command(app, "/auto refactor X")
    finally:
        commands._COMMAND_TABLE["/auto"] = original
    sentinel.assert_called_once_with(app, ["/auto", "refactor", "X"])


# ── /auto (no args) ───────────────────────────────────────────────────────


def test_auto_without_args_shows_usage(app: FakeApp, patched_workers) -> None:
    commands._cmd_auto(app, ["/auto"])
    assert len(app.messages) == 1
    speaker, msg, kind = app.messages[0]
    assert speaker == "System"
    assert kind == "system"
    assert "Usage:" in msg
    assert "pause" in msg.lower()
    assert "stop" in msg.lower()
    assert not app._engine_running
    assert not app._autonomous_active
    assert patched_workers == []


# ── /auto <msg> ───────────────────────────────────────────────────────────


def test_auto_starts_engine_with_trigger_prefix(
    app: FakeApp, patched_workers,
) -> None:
    """The user's message must be augmented with the autonomous trigger so
    the chat-agent detector trips regardless of the user's wording."""
    commands._cmd_auto(app, ["/auto", "refactor", "X"])

    assert app._engine_running is True
    assert app._autonomous_active is True
    assert len(patched_workers) == 1
    _fn_name, args, kwargs = patched_workers[0]
    prompt_arg = args[1]
    assert "manejate vos" in prompt_arg
    assert "refactor X" in prompt_arg
    assert kwargs.get("exclusive") is True


def test_auto_refuses_when_engine_running(
    app: FakeApp, patched_workers,
) -> None:
    app._engine_running = True
    commands._cmd_auto(app, ["/auto", "do", "stuff"])

    assert patched_workers == []
    assert not app._autonomous_active
    assert any("already running" in m[1] for m in app.messages)


def test_auto_treats_stop_word_as_message_when_followed_by_more(
    app: FakeApp, patched_workers,
) -> None:
    """``/auto stop the build`` must start a chain whose message is the full
    phrase — only an exact ``/auto stop`` invocation stops the chain."""
    commands._cmd_auto(app, ["/auto", "stop", "the", "build"])

    assert len(patched_workers) == 1
    _fn_name, args, _kwargs = patched_workers[0]
    assert "stop the build" in args[1]
    assert not app._cancel_event.is_set()


def test_auto_treats_pause_word_as_message_when_followed_by_more(
    app: FakeApp, patched_workers,
) -> None:
    """Symmetric test: ``/auto pause for breath`` starts a chain, not a pause."""
    commands._cmd_auto(app, ["/auto", "pause", "for", "breath"])

    assert len(patched_workers) == 1
    assert not app._cancel_event.is_set()


# ── /auto pause ───────────────────────────────────────────────────────────


def test_auto_pause_sets_cancel_event_and_keeps_active(app: FakeApp) -> None:
    app._autonomous_active = True
    assert not app._cancel_event.is_set()

    commands._cmd_auto(app, ["/auto", "pause"])

    assert app._cancel_event.is_set()
    # Pause must NOT clear the active flag — the user can still resume.
    assert app._autonomous_active is True
    assert any("paused" in m[1].lower() for m in app.messages)


# ── /auto stop ────────────────────────────────────────────────────────────


def test_auto_stop_sets_cancel_event_and_clears_active(app: FakeApp) -> None:
    app._autonomous_active = True
    assert not app._cancel_event.is_set()

    commands._cmd_auto(app, ["/auto", "stop"])

    assert app._cancel_event.is_set()
    # Stop must clear the flag so the next prompt does not re-chain.
    assert app._autonomous_active is False
    assert any("stopped" in m[1].lower() for m in app.messages)


# ── /auto unlimited / /auto bounded ──────────────────────────────────────


def test_auto_unlimited_sets_flag_and_starts_chain(
    app: FakeApp, patched_workers, monkeypatch
) -> None:
    """``/auto unlimited`` must flip ``AUTONOMOUS_UNLIMITED`` to True and
    start a chain with the rest of the line as the task description.
    """
    from infinidev.config.settings import settings

    previous = settings.AUTONOMOUS_UNLIMITED
    saved: list[bool] = []
    try:
        # Stub the helper so the test never touches the user's real
        # .infinidev/settings.json. The real implementation calls
        # ``settings.save_user_settings`` and ``reload_all``; we replace
        # it with a record-only function.
        def _stub(value: bool) -> None:
            settings.AUTONOMOUS_UNLIMITED = bool(value)
            saved.append(bool(value))

        monkeypatch.setattr(commands, "_set_autonomous_unlimited", _stub)
        commands._cmd_auto(app, ["/auto", "unlimited", "ship", "the", "feature"])
        assert settings.AUTONOMOUS_UNLIMITED is True
        assert saved == [True]
        # The chain was kicked off — engine is marked running and the
        # background worker is invoked with the task description.
        assert app._engine_running is True
        assert app._autonomous_active is True
        assert len(patched_workers) == 1
        worker_args = patched_workers[0][1]
        # The message is forwarded to the engine task — it should
        # contain the user's request and the autonomous trigger phrase.
        assert "ship the feature" in worker_args[1]
        # The on-screen banner marks the mode as UNLIMITED.
        banner_msgs = [m[1] for m in app.messages if "Autonomous mode active" in m[1]]
        assert banner_msgs
        assert "UNLIMITED" in banner_msgs[-1]
    finally:
        settings.AUTONOMOUS_UNLIMITED = previous


def test_auto_bounded_clears_unlimited_flag(
    app: FakeApp, patched_workers, monkeypatch
) -> None:
    """``/auto bounded`` flips ``AUTONOMOUS_UNLIMITED`` back to False and
    starts a chain in the conservative mode. This is the escape hatch
    for users who want to revert to the default after a session in
    unlimited mode.
    """
    from infinidev.config.settings import settings

    previous = settings.AUTONOMOUS_UNLIMITED
    saved: list[bool] = []
    try:
        settings.AUTONOMOUS_UNLIMITED = True

        def _stub(value: bool) -> None:
            settings.AUTONOMOUS_UNLIMITED = bool(value)
            saved.append(bool(value))

        monkeypatch.setattr(commands, "_set_autonomous_unlimited", _stub)
        commands._cmd_auto(app, ["/auto", "bounded", "tidy", "the", "changelog"])
        assert settings.AUTONOMOUS_UNLIMITED is False
        assert saved == [False]
        # Banner shows the bounded mode.
        banner_msgs = [m[1] for m in app.messages if "Autonomous mode active" in m[1]]
        assert banner_msgs
        assert "UNLIMITED" not in banner_msgs[-1]
    finally:
        settings.AUTONOMOUS_UNLIMITED = previous


def test_auto_unlisted_subcommand_falls_through_to_task_description(
    app: FakeApp, patched_workers, monkeypatch
) -> None:
    """``/auto stop the build`` is a task description (not a stop
    command). The unlimited/bounded subcommands share this fallback:
    any other second word is part of the message, not a mode flag.
    """
    from infinidev.config.settings import settings

    previous = settings.AUTONOMOUS_UNLIMITED
    saved: list[bool] = []
    try:
        def _stub(value: bool) -> None:
            settings.AUTONOMOUS_UNLIMITED = bool(value)
            saved.append(bool(value))

        monkeypatch.setattr(commands, "_set_autonomous_unlimited", _stub)
        commands._cmd_auto(app, ["/auto", "refactor", "the", "loader"])
        # Flag is untouched — the second word is not unlimited/bounded.
        assert settings.AUTONOMOUS_UNLIMITED == previous
        assert saved == []
        # The full tail is the task description.
        worker_args = patched_workers[0][1]
        assert "refactor the loader" in worker_args[1]
    finally:
        settings.AUTONOMOUS_UNLIMITED = previous


# ── Robustness ────────────────────────────────────────────────────────────


def test_handlers_do_not_crash_without_cancel_event(app: FakeApp) -> None:
    """If the app has no _cancel_event, pause/stop must not raise."""
    app._cancel_event = None

    commands._cmd_auto(app, ["/auto", "pause"])
    commands._cmd_auto(app, ["/auto", "stop"])


def test_unknown_command_still_warns(app: FakeApp) -> None:
    commands.handle_command(app, "/nope")
    assert any("Unknown command" in m[1] for m in app.messages)
