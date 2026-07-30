"""Tests for message_widgets rendering math and badge behaviour.

Covers the chat-Window width invariants (the Window uses
wrap_lines=False, so any over-long line is hard-clipped — width math
matters), the empty-message suppression, the diff title-bar fill, the
copy-badge revert scheduling, and cell-width-aware truncation.
"""

from __future__ import annotations

import infinidev.ui.controls.message_widgets as mw
from infinidev.ui.controls.message_widgets import (
    BorderedWidget,
    DiffWidget,
    _cell_len,
    _render_bordered_body,
    _truncate,
    _wrap_fragments,
)


def _flat(lines) -> list[str]:
    return ["".join(t for _, t in line) for line in lines]


# ── B1: long markdown lines keep styling, drop markers ───────────────────


def test_wrap_fragments_preserves_style_and_width():
    frags = [("bold", "x" * 50), ("italic", "y" * 50)]
    rows = _wrap_fragments(frags, 40)
    # Every wrapped row fits the column budget.
    for row in rows:
        assert _cell_len("".join(t for _, t in row)) <= 40
    # Both styles carry through to the continuation rows.
    styles = {s for row in rows for s, _ in row}
    assert "bold" in styles and "italic" in styles
    # No content is lost.
    joined = "".join(t for row in rows for _, t in row)
    assert joined == "x" * 50 + "y" * 50


def test_wrap_fragments_handles_wide_chars():
    rows = _wrap_fragments([("s", "中" * 10)], 6)
    for row in rows:
        assert _cell_len("".join(t for _, t in row)) <= 6
    assert "".join(t for row in rows for _, t in row) == "中" * 10


def test_long_bold_markdown_line_keeps_style_and_strips_markers():
    text = "**" + " ".join(["word"] * 40) + "**"
    lines = _render_bordered_body(
        text, width=40, border_char="|", border_style="b",
        body_style="base", fill_style="", use_markdown=True,
    )
    flat = "\n".join(_flat(lines))
    # Markers are stripped even though the line wrapped...
    assert "**" not in flat
    assert "word" in flat
    # ...and bold styling survives on the wrapped continuation lines.
    assert any("bold" in s for line in lines for s, _ in line)


# ── B5: empty agent message renders nothing ──────────────────────────────


def test_empty_agent_message_renders_nothing():
    w = BorderedWidget("agent", "Responses")
    assert w.render({"sender": "Infinidev", "text": ""}, 80).lines == []
    assert w.render({"sender": "Infinidev", "text": "  \n  "}, 80).lines == []


def test_agent_reply_renders_as_plain_indented_prose():
    """No name tag, no mark, no background — the reply is the content."""
    w = BorderedWidget("agent", "Responses")
    rr = w.render({"sender": "Infinidev", "text": "hello"}, 80)
    flat = "".join(t for line in rr.lines for _, t in line)
    assert rr.lines
    assert "hello" in flat
    assert "Infinidev:" not in flat
    assert rr.lines[0][0][1].strip() == "", "no gutter glyph on assistant text"
    # Copy stays reachable even without a header line to hang it on.
    assert 0 in rr.clickable_offsets
    assert "⧉" in flat


def test_user_message_is_marked_once_not_on_every_line():
    w = BorderedWidget("user", "Messages")
    rr = w.render({"sender": "You", "text": "line one\nline two"}, 80)
    marks = [line[0][1] for line in rr.lines if line and line[0][1].strip()]
    assert marks == ["> "], "one prompt mark, on the first line only"


def test_copy_glyph_is_not_drawn_on_every_message_type():
    """It used to render " [⧉] " on every message; that was the loudest
    thing on screen after the text itself."""
    user = BorderedWidget("user", "Messages").render(
        {"sender": "You", "text": "hello"}, 80
    )
    flat = "".join(t for line in user.lines for _, t in line)
    assert "⧉" not in flat
    assert 0 in user.clickable_offsets, "the click target must survive"


def test_named_speaker_keeps_its_header():
    """A sender whose identity matters (Reviewer, critic verdicts) is named."""
    w = BorderedWidget("agent", "Responses")
    rr = w.render({"sender": "Reviewer", "text": "needs a test"}, 80)
    flat = "".join(t for line in rr.lines for _, t in line)
    assert "Reviewer:" in flat


# ── B4: diff title bar fills the full width ──────────────────────────────


def test_diff_title_bar_fills_full_width():
    rr = DiffWidget().render(
        {"text": "src/foo.py", "diff_text": "", "collapsed": True}, 40,
    )
    title = "".join(t for _, t in rr.lines[0])
    # Was width-1 before the fix (a 1-col unpainted notch on the right).
    assert len(title) == 40


# ── B3: copy badge schedules its own revert ──────────────────────────────


def test_schedule_badge_revert_noop_without_app():
    # Must not raise when there is no running prompt_toolkit app (tests).
    mw._schedule_badge_revert()


def test_copy_handler_schedules_revert_and_records_highlight(monkeypatch):
    import infinidev.ui.clipboard as cb
    monkeypatch.setattr(cb, "copy_to_clipboard", lambda text: True)

    calls = {"n": 0}
    monkeypatch.setattr(
        mw, "_schedule_badge_revert",
        lambda: calls.__setitem__("n", calls["n"] + 1),
    )

    w = BorderedWidget("agent", "Responses")
    msg = {"sender": "Infinidev", "text": "hi"}
    rr = w.render(msg, 80)
    rr.clickable_offsets[0]()  # simulate a click on the copy button

    assert calls["n"] == 1
    # The very next render reflects the recorded highlight.
    flat = "".join(t for line in w.render(msg, 80).lines for _, t in line)
    assert "copied" in flat


# ── B7: cell-width-aware truncation ──────────────────────────────────────


def test_truncate_counts_columns_not_code_points():
    # Two wide chars (4 cols) + ellipsis (1 col) = 5 cols.
    assert _truncate("中文字符", 5) == "中文…"
    # ASCII behaviour is unchanged.
    assert _truncate("hello world", 5) == "hell…"
    assert _truncate("short", 10) == "short"
