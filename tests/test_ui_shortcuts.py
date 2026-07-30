"""The shortcuts the UI advertises must actually exist.

The status line promises `? help`; the sidebar is only reachable by
keyboard. Both were promises with nothing behind them at one point, which
is worse than not offering them at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.ui.keybindings import create_global_keybindings


class _Buffer:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _App:
    """The slice of InfinidevApp the bindings touch."""

    def __init__(self) -> None:
        self._chat_buffer = _Buffer()
        self.sidebar_visible = False
        self.explorer_visible = False
        self.active_tab = "chat"
        self.active_dialog = None
        self.messages: list[tuple[str, str, str]] = []
        self.focused_sidebar = False

    def toggle_sidebar(self) -> None:
        self.sidebar_visible = not self.sidebar_visible

    def toggle_explorer(self) -> None:
        self.explorer_visible = not self.explorer_visible

    def focus_sidebar(self) -> None:
        self.focused_sidebar = True

    def add_message(self, sender: str, text: str, kind: str = "agent") -> None:
        self.messages.append((sender, text, kind))

    def invalidate(self) -> None:
        pass


def _binding(bindings, *keys):
    """Find a handler registered for exactly *keys*.

    Keys arrive either as plain strings or as ``Keys`` enum members whose
    ``value`` is the string form ("f4"), so normalise before comparing.
    """
    for binding in bindings.bindings:
        registered = tuple(getattr(k, "value", k) for k in binding.keys)
        if registered == keys:
            return binding
    raise AssertionError(f"no binding for {keys}")


def _fire(binding, app):
    binding.handler(SimpleNamespace(app=SimpleNamespace(invalidate=lambda: None)))
    return app


@pytest.fixture
def app():
    return _App()


def test_question_mark_opens_help_on_an_empty_prompt(app):
    bindings = create_global_keybindings(app)
    binding = _binding(bindings, "?")
    assert binding.filter() is True
    _fire(binding, app)
    assert app.messages, "? must produce the help text"
    body = app.messages[-1][1]
    assert "PANELS" in body and "COMMANDS" in body
    assert "Alt+." in body, "the sidebar shortcut must be documented"


def test_question_mark_types_normally_once_there_is_text(app):
    app._chat_buffer.text = "what about"
    bindings = create_global_keybindings(app)
    assert _binding(bindings, "?").filter() is False


def test_alt_dot_toggles_the_sidebar(app):
    bindings = create_global_keybindings(app)
    _fire(_binding(bindings, "escape", "."), app)
    assert app.sidebar_visible is True


def test_f4_opens_the_sidebar_before_trying_to_focus_it(app):
    bindings = create_global_keybindings(app)
    binding = _binding(bindings, "f4")
    _fire(binding, app)
    assert app.sidebar_visible is True
    assert app.focused_sidebar is False, "first press opens, it does not focus"
    _fire(binding, app)
    assert app.focused_sidebar is True


def test_slash_sidebar_is_the_terminal_proof_fallback(app):
    from infinidev.ui.handlers.commands import _cmd_sidebar

    _cmd_sidebar(app, ["/sidebar"])
    assert app.sidebar_visible is True


def test_help_lists_only_commands_that_exist():
    """Every `/command` named in the help must be dispatchable."""
    import re

    from infinidev.ui.handlers.commands import _COMMAND_TABLE, _cmd_help

    app = _App()
    _cmd_help(app, ["/help"])
    named = set(re.findall(r"(?:^|\s)(/[a-z]+)", app.messages[-1][1]))
    missing = sorted(name for name in named if name not in _COMMAND_TABLE)
    assert not missing, f"help advertises commands with no handler: {missing}"
