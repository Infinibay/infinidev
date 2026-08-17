"""Regression tests for the /debug dialog close path.

User-reported bug: opening /debug while the engine is streaming thinking and
pressing Escape to close it left the TUI hung — focus was parked on a now
hidden Float child so subsequent key events went nowhere. These tests pin
the defensive fixes applied to ``DialogManager._init_debug_dialog`` and the
Escape keybinding it installs:

  * the close handler is idempotent (safe under repeated presses and under
    a thinking worker that is firing ``app.invalidate`` from another thread),
  * focus is restored to ``app._chat_input_control`` defensively (no
    exception can leave focus on a hidden child),
  * ``_init_debug_dialog`` is idempotent: a second call does not stack the
    ``get_key_bindings`` wrappers.

The test mocks the ``InfinidevApp`` surface the dialog actually touches; it
does not need a real prompt_toolkit ``Application`` running.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

from infinidev.ui.handlers.dialogs import DialogManager


# ── Helpers ───────────────────────────────────────────────────────────


class _FakeApp:
    """Minimal stand-in for InfinidevApp that supports what DialogManager touches."""

    def __init__(self) -> None:
        self.active_dialog = None
        self.engine = None
        # The close handler reaches into ``app.app.layout.focus(...)`` — give
        # it a chain of mocks so any call resolves without raising.
        self.app = MagicMock()
        self.app.layout = MagicMock()
        self.app.layout.focus = MagicMock()
        self._chat_input_control = MagicMock(name="_chat_input_control")
        self.focus_chat = MagicMock()
        self.invalidate = MagicMock()
        # The thinking worker mutates these on the real app; provide them
        # so the simulated worker in the regression tests can run cleanly.
        self._thinking_full = ""
        self._thinking_text = ""
        # Float container — appending to ``floats`` is fine even when the
        # dialog never gets opened in a real Application.
        self._float_container = MagicMock()
        self._float_container.floats = []


@pytest.fixture
def fake_app(monkeypatch: pytest.MonkeyPatch) -> _FakeApp:
    """Return a fake app with ``run_in_background`` patched to a no-op.

    ``open_debug`` schedules ``_refresh_debug_state`` on a background
    thread; the test only cares about the dialog state and the close
    handler, so swallowing the schedule avoids touching the real executor.
    """
    app = _FakeApp()
    # Stop the background refresh from running — we are not interested in
    # engine/scorer snapshots, only the kb wrap and close handler.
    monkeypatch.setattr(
        "infinidev.ui.workers.run_in_background",
        lambda *a, **kw: None,
    )
    return app


def _trigger_close(manager: DialogManager) -> None:
    """Invoke the Escape handler that ``_init_debug_dialog`` installed.

    Each call to ``ctrl.get_key_bindings()`` returns a *fresh* ``KeyBindings``
    object with a single Escape binding pointing at the shared
    ``_close_debug`` closure. Reaching into the kb to fire the handler is
    the cleanest way to simulate the user pressing Escape.
    """
    kb = manager._debug_sections_window.content.get_key_bindings()
    bindings = list(kb.bindings)
    escape = [b for b in bindings if b.keys == ("escape",)]
    assert escape, "expected an Escape binding on the debug sections control"
    # The handler ignores its event argument — pass a bare Mock.
    escape[0].handler(MagicMock())


# ── Tests ─────────────────────────────────────────────────────────────


def test_open_debug_activates_dialog_while_engine_is_thinking(
    fake_app: _FakeApp,
) -> None:
    """Opening the dialog while the engine streams thinking sets the state.

    We simulate a thinking worker by mutating ``_thinking_full`` from another
    thread in parallel with ``open_debug``. The dialog must still open and
    register its key bindings.
    """
    manager = DialogManager(fake_app)

    thinking_done = threading.Event()
    thinking_started = threading.Event()

    def _thinking_worker() -> None:
        thinking_started.set()
        # Pulse the thinking field a few times to mimic the worker's
        # 10-FPS invalidate cadence. We do NOT call ``fake_app.invalidate``
        # here — ``open_debug`` schedules the dialog's redraw, not us.
        for _ in range(20):
            fake_app._thinking_full = (fake_app._thinking_full or "") + "x"
            if thinking_done.is_set():
                return
        thinking_done.set()

    thread = threading.Thread(target=_thinking_worker, daemon=True)
    thread.start()
    thinking_started.wait(timeout=1.0)

    manager.open_debug()

    assert fake_app.active_dialog == "debug_panel"
    assert manager._debug_initialized is True
    assert manager._debug_state is not None

    thinking_done.set()
    thread.join(timeout=1.0)


def test_escape_close_clears_dialog_and_restores_focus(
    fake_app: _FakeApp,
) -> None:
    """Escape on the open dialog clears ``active_dialog`` and refocuses chat.

    The close handler must:
      * clear ``app.active_dialog`` so the Float's ConditionalContainer hides,
      * move focus to ``app._chat_input_control`` via ``app.app.layout.focus``,
      * schedule a final redraw with ``app.invalidate``.
    """
    manager = DialogManager(fake_app)
    manager.open_debug()
    assert fake_app.active_dialog == "debug_panel"

    _trigger_close(manager)

    assert fake_app.active_dialog is None
    # ``app.app.layout.focus`` must have been called on the chat input
    # control (focus restore is what the original bug left out when an
    # exception was raised by the layout).
    fake_app.app.layout.focus.assert_any_call(fake_app._chat_input_control)
    # And a redraw must have been requested last.
    fake_app.invalidate.assert_called()


def test_escape_close_is_idempotent(fake_app: _FakeApp) -> None:
    """Multiple Escape presses / open-close cycles must not raise or hang.

    Simulates the user mashing Escape, and the engine worker continuing to
    call ``app.invalidate`` between presses. After every press the dialog
    state must be ``None``; on subsequent presses the handler is a no-op.
    """
    manager = DialogManager(fake_app)
    invalidate_call_count = 0

    def _counting_invalidate() -> None:
        nonlocal invalidate_call_count
        invalidate_call_count += 1

    fake_app.invalidate.side_effect = _counting_invalidate

    # Open → close, three times, with a thinking invalidate interleaved.
    for cycle in range(3):
        manager.open_debug()
        assert fake_app.active_dialog == "debug_panel", f"cycle {cycle} did not open"

        # Mimic the engine worker firing invalidates mid-flight.
        for _ in range(5):
            fake_app.invalidate()

        _trigger_close(manager)
        assert fake_app.active_dialog is None, f"cycle {cycle} did not close"

        # Pressing Escape again with no dialog open must be a safe no-op.
        _trigger_close(manager)
        assert fake_app.active_dialog is None


def test_init_debug_dialog_is_idempotent(fake_app: _FakeApp) -> None:
    """A second ``_init_debug_dialog`` call must not stack kb wrappers.

    Captures the ``get_key_bindings`` method identity before and after a
    repeat call; the wrapper that adds the Escape binding must be installed
    once, not twice. Two wraps would mean a second Escape binding firing
    the close logic on every render — the original symptom class.
    """
    manager = DialogManager(fake_app)

    manager._init_debug_dialog()
    first_wrapper = manager._debug_sections_window.content.get_key_bindings

    manager._init_debug_dialog()
    second_wrapper = manager._debug_sections_window.content.get_key_bindings

    assert first_wrapper is second_wrapper, (
        "_init_debug_dialog re-wrapped get_key_bindings — the kb stack grew"
    )

    # And the dialog still works exactly once on Escape.
    manager.open_debug()
    assert fake_app.active_dialog == "debug_panel"
    _trigger_close(manager)
    assert fake_app.active_dialog is None
