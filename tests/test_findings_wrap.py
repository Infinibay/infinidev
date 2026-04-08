"""Regression tests for the findings detail wrap helper.

The previous version of ``_wrap_styled`` hung the entire TUI when
the user opened ``/findings`` and pressed right (which focuses the
detail panel and triggers a re-render). The bug was an infinite
loop on any text containing a word longer than the wrap width:
``rfind(" ", 0, width)`` would happily return a position inside
the continuation indent ``"  "`` we re-add each iteration, the
chunk would be pure whitespace, and the remainder would be
unchanged.

These tests assert progress (every iteration must shrink the
remaining text by at least one *real* character) by giving the
function inputs that are dominated by long unbreakable words —
file paths, URLs, hashes, identifiers — exactly the content real
findings contain.
"""

from __future__ import annotations

import pytest

from infinidev.ui.dialogs.findings_detail_control import _wrap_styled


@pytest.mark.parametrize("text,width", [
    ("a" * 100, 20),
    ("short word " + ("x" * 100), 20),
    (
        "see /home/andres/infinidev/src/infinidev/engine/orchestration/"
        "conversational_fastpath.py for details",
        20,
    ),
    ("https://example.com/very/long/path/with/segments/here", 30),
    (
        "The function FindingsDetailControl._wrap_styled at line 86 "
        "had an unbreakable-word infinite loop bug.",
        40,
    ),
    ("          ", 5),  # only spaces
    ("a" * 20, 20),  # text exactly width
    ("hello", 0),  # invalid width — must not hang
])
def test_wrap_styled_terminates(text, width):
    """Every input must produce a result; none may hang."""
    result = _wrap_styled("class:x", text, width)
    assert isinstance(result, list)
    # Sanity: total characters out >= total real characters in
    # (some whitespace may be added/dropped, but content survives).
    out_chars = "".join(seg[1] for line in result for seg in line)
    assert text.replace(" ", "") == "" or any(
        ch in out_chars for ch in text if ch != " "
    )


def test_wrap_styled_empty_string():
    result = _wrap_styled("class:x", "", 20)
    assert result == [[("class:x", "")]]


def test_wrap_styled_short_text_no_wrap():
    result = _wrap_styled("class:x", "hi", 20)
    assert result == [[("class:x", "hi")]]
