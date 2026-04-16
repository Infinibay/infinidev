"""Tests for multi-line chat input (backslash continuation)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout.controls import BufferControl

from infinidev.ui.controls.chat_input import create_chat_input


# ── Helpers ────────────────────────────────────────────────────────────────


def _make_event(buffer: Buffer) -> KeyPressEvent:
    """Create a minimal KeyPressEvent that satisfies the handlers."""
    # KeyPressEvent expects an app; use a simple namespace with a .current_buffer
    fake_app = SimpleNamespace(current_buffer=buffer)
    return SimpleNamespace(app=fake_app)

def _fire_binding(kb: KeyBindings, keys: tuple[str, ...], buffer: Buffer) -> None:
    """Find a handler registered for *keys* in *kb* and invoke it.

    *keys* uses human-readable names (``"enter"``, ``"up"``) which are
    translated to prompt_toolkit's internal ``Keys`` enum values for matching.
    """
    _KEY_MAP = {
        "enter": "c-m",
        "up": "up",
        "down": "down",
    }
    normalized = tuple(_KEY_MAP.get(k, k) for k in keys)
    for reg in kb.bindings:
        reg_keys = tuple(k.value if hasattr(k, "value") else k for k in reg.keys)
        if reg_keys == normalized:
            reg.handler(_make_event(buffer))
            return
    raise KeyError(f"No binding found for {keys!r}")


def _create_input(
    on_submit: Callable[[str], None] | None = None,
) -> tuple[Buffer, BufferControl, KeyBindings, list[str]]:
    """Create a chat input and return (buf, ctrl, kb, submitted)."""
    submitted: list[str] = []
    buf, ctrl, kb = create_chat_input(
        on_submit=lambda txt: submitted.append(txt),
    )
    return buf, ctrl, kb, submitted


# ── Enter submits ──────────────────────────────────────────────────────────


class TestEnterSubmits:
    """Plain Enter on a single line submits the message."""

    def test_submits_non_empty_text(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "hello world"
        _fire_binding(kb, ("enter",), buf)
        assert submitted == ["hello world"]

    def test_clears_buffer_after_submit(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "hello"
        _fire_binding(kb, ("enter",), buf)
        assert buf.text == ""

    def test_empty_text_does_not_submit(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "   "
        _fire_binding(kb, ("enter",), buf)
        assert submitted == []


# ── Backslash continuation ─────────────────────────────────────────────────


class TestBackslashContinuation:
    r"""Enter when line ends with \ removes the backslash and inserts a newline."""

    def test_backslash_creates_newline(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "hello\\"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)
        assert "\n" in buf.text
        assert "\\" not in buf.text
        assert submitted == []  # did not submit

    def test_backslash_in_middle_of_multiline(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "line1\nline2\\"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)
        assert buf.text == "line1\nline2\n"
        assert submitted == []

    def test_backslash_only_line(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "\\"
        buf.cursor_position = 1
        _fire_binding(kb, ("enter",), buf)
        assert buf.text == "\n"
        assert submitted == []

    def test_no_backslash_does_not_continuate(self):
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "hello"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)
        assert submitted == ["hello"]
        assert buf.text == ""

    def test_backslash_not_at_end_does_not_continuate(self):
        """A backslash in the middle of the line should still submit."""
        buf, _ctrl, kb, submitted = _create_input()
        buf.text = "hel\\lo"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)
        assert submitted == ["hel\\lo"]



# ── Submit after multi-line editing ────────────────────────────────────────


class TestSubmitAfterMultiline:
    r"""After building a multi-line message via \ continuation,
    a final Enter on a non-backslash line submits the full text."""

    def test_backslash_then_submit(self):
        buf, _ctrl, kb, submitted = _create_input()
        # Build: "hello\" -> "hello\n", type "world", submit
        buf.text = "hello\\"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)  # continuation
        buf.text = "hello\nworld"
        buf.cursor_position = len(buf.text)
        _fire_binding(kb, ("enter",), buf)  # submit
        assert submitted == ["hello\nworld"]
