"""Tests for ChatHistoryControl scroll-anchor behaviour.

The user must keep full control of scroll position: new messages should
NOT yank the viewport to the bottom while the user is reading older
content. Only `end` / `pagedown` (or scrolling all the way down) should
re-engage tail-following.
"""

from __future__ import annotations

import pytest

from infinidev.ui.controls.chat_history import ChatHistoryControl


@pytest.fixture()
def chat():
    msgs: list[dict] = []
    c = ChatHistoryControl(msgs)
    return c, msgs


def _force_render(chat: ChatHistoryControl, width: int = 80) -> None:
    """Force a rebuild — bypasses the rebuild throttle."""
    chat._line_cache = None
    chat._last_rebuild = 0.0
    chat.create_content(width=width, height=24)


def test_invalidate_cache_does_not_reset_scroll_position(chat):
    c, msgs = chat
    # Seed some messages and render once to establish line_count.
    for i in range(10):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})
    _force_render(c)

    # User scrolls up.
    c._follow_tail = False
    c._scroll_offset = 5

    # New message arrives; cache invalidates.
    msgs.append({"type": "agent", "sender": "A", "text": "new msg"})
    c.invalidate_cache()

    # Critical: tail-follow stays disabled and offset is preserved
    # (or grown, but never reset to 0).
    assert c._follow_tail is False
    assert c._scroll_offset >= 5


def test_new_messages_keep_cursor_y_stable_when_scrolled_up(chat):
    c, msgs = chat
    for i in range(20):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})
    _force_render(c)

    initial_line_count = c._line_count
    assert initial_line_count > 0

    # Pretend user scrolled up by 8 lines.
    c._follow_tail = False
    c._scroll_offset = 8
    initial_cursor_y = initial_line_count - 1 - 8

    # Add 3 new messages.
    for i in range(3):
        msgs.append({"type": "agent", "sender": "A", "text": f"new {i}"})
    c.invalidate_cache()
    _force_render(c)

    new_line_count = c._line_count
    new_cursor_y = new_line_count - 1 - c._scroll_offset

    # The user's anchor (cursor_y) should be unchanged → they keep
    # seeing the same content, while the new messages pile up below.
    assert new_cursor_y == initial_cursor_y


def test_tail_follow_keeps_cursor_at_bottom_with_new_messages(chat):
    c, msgs = chat
    for i in range(5):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})
    _force_render(c)

    # Default: at bottom.
    assert c._follow_tail is True
    assert c._scroll_offset == 0

    msgs.append({"type": "agent", "sender": "A", "text": "another"})
    c.invalidate_cache()
    _force_render(c)

    # Cursor should be at the new last line (line_count - 1).
    cursor_y = c._line_count - 1 - c._scroll_offset
    assert cursor_y == c._line_count - 1


def test_end_key_reengages_tail_follow(chat):
    c, msgs = chat
    for i in range(10):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})
    _force_render(c)

    c._follow_tail = False
    c._scroll_offset = 4

    # Simulate `end` key.
    kb = c.get_key_bindings()
    end_handlers = [b.handler for b in kb.bindings if "end" in str(b.keys)]
    assert end_handlers, "end key binding not found"
    end_handlers[0](type("Ev", (), {})())

    assert c._follow_tail is True
    assert c._scroll_offset == 0
