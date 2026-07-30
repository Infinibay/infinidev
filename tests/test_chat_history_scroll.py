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


def _flat_text(c: ChatHistoryControl) -> str:
    """Concatenate all rendered (cached) line text for content assertions."""
    return "".join(t for line in (c._line_cache or []) for _, t in line)


# ── C1: consecutive agent/system replies must stay visible by default ──

def test_consecutive_agent_messages_both_visible_by_default(chat):
    c, msgs = chat
    msgs.append({"type": "agent", "sender": "A", "text": "first reply"})
    msgs.append({"type": "agent", "sender": "A", "text": "second reply"})
    _force_render(c)

    # No manual toggle yet → default must be EXPANDED, so BOTH replies
    # render. Previously the group defaulted to collapsed and only the
    # last one showed under a "Responses (2)" header.
    assert c._group_states == {}
    flat = _flat_text(c)
    assert "first reply" in flat
    assert "second reply" in flat


def test_group_header_toggle_collapses_then_expands(chat):
    """Grouping still works — for machine output, which is what it is for.

    Conversation turns (user/agent) are exempt now: folding replies under
    a "▼ Responses (2)" header hid the thing the user came to read. System
    notices are the case grouping was built for.
    """
    c, msgs = chat
    msgs.append({"type": "system", "sender": "System", "text": "first reply"})
    msgs.append({"type": "system", "sender": "System", "text": "second reply"})
    _force_render(c)

    # The group header is the first clickable line (offset 0).
    toggle = c._clickable_lines[0]

    # First click: collapse → earlier reply hidden, last one kept.
    toggle()
    assert c._group_states[0] is True
    _force_render(c)
    flat = _flat_text(c)
    assert "first reply" not in flat
    assert "second reply" in flat

    # Second click: expand again → both visible.
    c._clickable_lines[0]()
    assert c._group_states[0] is False
    _force_render(c)
    flat = _flat_text(c)
    assert "first reply" in flat
    assert "second reply" in flat


# ── C2: anchor stays put while "thinking" indicator is active ──────────

def test_anchor_stable_when_scrolled_up_with_thinking_active(chat):
    c, msgs = chat
    for i in range(20):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})
    c.show_thinking = True
    _force_render(c)

    initial_line_count = c._line_count  # includes the +2 thinking lines
    c._follow_tail = False
    c._scroll_offset = 8
    initial_cursor_y = initial_line_count - 1 - 8

    # New messages arrive while thinking is still active.
    for i in range(3):
        msgs.append({"type": "agent", "sender": "A", "text": f"new {i}"})
    c.invalidate_cache()
    _force_render(c)

    new_cursor_y = c._line_count - 1 - c._scroll_offset
    # The anchor must not drift by the 2 thinking-indicator lines.
    assert new_cursor_y == initial_cursor_y


# ── C3: rebuild throttle must engage during streaming bursts ───────────

def test_streaming_throttle_reuses_lines_when_cache_nulled(chat):
    c, msgs = chat
    for i in range(5):
        msgs.append({"type": "agent", "sender": "A", "text": f"msg {i}"})

    # First real render establishes _last_lines and a recent _last_rebuild.
    c._line_cache = None
    c._last_rebuild = 0.0
    c.create_content(width=80, height=24)

    # Count full rebuilds during a simulated streaming burst.
    calls = {"n": 0}
    orig = c._do_rebuild

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    c._do_rebuild = counting

    # Streaming grows the last message's text and nulls the cache every
    # frame (exactly what append_to_last_message does), all inside the
    # throttle window. The throttle must NOT depend on _line_cache being
    # non-None — otherwise it never engages here.
    for _ in range(10):
        msgs[-1]["text"] += "x"
        c.invalidate_cache()
        c.create_content(width=80, height=24)

    # Throttle engaged → far fewer than 10 rebuilds.
    assert calls["n"] <= 1
