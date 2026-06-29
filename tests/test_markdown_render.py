"""Tests for the lightweight markdown → fragment renderer.

Focus: the inline parser must emit every character (no silent drops),
must not false-match italic across whitespace-only spans, and headings
must strip hashes up to H6.
"""

from __future__ import annotations

from infinidev.ui.controls.markdown_render import render_markdown_line, _parse_inline


def _text(frags) -> str:
    return "".join(t for _, t in frags)


# ── B2: every character is emitted; arithmetic preserved ─────────────────


def test_lone_asterisk_arithmetic_preserved():
    # "5 * 4 = 20" used to lose the lone '*' (became "5  4 = 20").
    frags = _parse_inline("5 * 4 = 20", "", "")
    assert _text(frags) == "5 * 4 = 20"


def test_double_stray_asterisks_not_italic():
    # "a * b * c" must NOT render " b " as italic.
    frags = _parse_inline("a * b * c", "", "")
    assert _text(frags) == "a * b * c"
    assert all("italic" not in style for style, _ in frags)


def test_lone_backtick_preserved():
    frags = _parse_inline("a ` b", "", "")
    assert _text(frags) == "a ` b"


# ── B2: real bold/italic/code still parse ────────────────────────────────


def test_real_bold_parses_and_strips_markers():
    frags = _parse_inline("**hi**", "base", "")
    assert _text(frags) == "hi"
    assert any("bold" in style for style, _ in frags)


def test_real_italic_parses_and_strips_markers():
    frags = _parse_inline("*hi*", "base", "")
    assert _text(frags) == "hi"
    assert any("italic" in style for style, _ in frags)


def test_real_inline_code_parses_and_strips_markers():
    frags = _parse_inline("`code`", "base", "")
    assert _text(frags) == "code"


def test_inline_code_may_contain_spaces():
    frags = _parse_inline("`foo bar`", "base", "")
    assert _text(frags) == "foo bar"


def test_bold_inside_a_sentence():
    frags = _parse_inline("a **b** c", "base", "")
    assert _text(frags) == "a b c"
    assert any("bold" in style for style, _ in frags)


# ── B6: headings up to H6 strip their hashes ─────────────────────────────


def test_headings_h1_through_h6_strip_hashes():
    for level in range(1, 7):
        line = "#" * level + " Title"
        frags = render_markdown_line(line, "base", "")
        rendered = _text(frags)
        assert rendered == "Title", f"H{level} did not strip hashes"
        assert "#" not in rendered
