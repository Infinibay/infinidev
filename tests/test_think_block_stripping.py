"""Reasoning blocks must never reach the transcript — including broken ones.

Two paths strip them and they did not agree. ``_THINK_BLOCK_RE`` has always
matched all three spellings a model might use; ``ThinkStreamFilter``, which
is what the TUI actually sees while a reply streams, knew only ``<think>``.

The case that surfaced it is not a model quirk but a provider one. Asked to
split reasoning out (MiniMax's ``reasoning_split``, and others like it), the
server moves the block into ``reasoning_content`` and leaves its terminator
behind in ``content`` — so the user gets a lone ``</thinking>`` and no
thinking. A stripper that only removes balanced pairs cannot see that.
"""

from __future__ import annotations

import pytest

from infinidev.engine.loop.llm_caller import ThinkStreamFilter, strip_think_blocks


def _streamed(raw: str, chunk: int = 3) -> str:
    """Feed *raw* through the filter in small deltas, as a stream arrives."""
    f = ThinkStreamFilter()
    out = "".join(f.feed(raw[i:i + chunk]) for i in range(0, len(raw), chunk))
    return out + f.flush()


BALANCED = [
    pytest.param("<think>reasoning</think>The answer", id="short"),
    pytest.param("<thinking>reasoning</thinking>The answer", id="long"),
    pytest.param("<thoughts>reasoning</thoughts>The answer", id="thoughts"),
    pytest.param("<THINKING>reasoning</THINKING>The answer", id="uppercase"),
]


@pytest.mark.parametrize("raw", BALANCED)
def test_balanced_blocks_are_removed_on_both_paths(raw):
    assert strip_think_blocks(raw) == "The answer"
    assert _streamed(raw) == "The answer"


@pytest.mark.parametrize("tag", ["think", "thinking", "thoughts"])
def test_an_orphan_terminator_does_not_reach_the_user(tag):
    """The reported symptom, exactly: the closing tag and nothing else."""
    raw = f"</{tag}>The answer"
    assert strip_think_blocks(raw) == "The answer"
    assert _streamed(raw) == "The answer"


def test_an_orphan_opening_tag_truncates_rather_than_leaking():
    """A stream cut mid-reasoning must not spill what follows the tag."""
    assert strip_think_blocks("The answer<thinking>cut off") == "The answer"
    assert _streamed("The answer<thinking>cut off") == "The answer"


def test_a_tag_split_across_deltas_is_still_caught():
    """The long spellings are longer than the short one, so the hold-back
    window has to grow with them or a boundary split leaks the tag."""
    for size in (1, 2, 4, 7, 11):
        assert _streamed("<thinking>r</thinking>The answer", chunk=size) == "The answer"


def test_text_without_tags_survives_untouched():
    assert strip_think_blocks("The answer") == "The answer"
    assert _streamed("The answer") == "The answer"


def test_the_long_spelling_is_not_read_as_the_short_one():
    """``<thinking>`` contains no ``<think>``, but a naive longest-match bug
    would leave a stray ``ing>`` in the transcript."""
    assert "ing>" not in _streamed("<thinking>r</thinking>ok")
    assert "ing>" not in strip_think_blocks("<thinking>r</thinking>ok")
