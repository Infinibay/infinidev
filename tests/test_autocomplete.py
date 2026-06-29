"""Tests for the /command autocomplete overlay rendering.

Covers:
- D1: the 8-item render window follows selected_index so the highlighted
  entry stays visible when navigating past the window.
- D2: newlines separate entries only — no trailing blank line.
"""

from __future__ import annotations

from infinidev.ui.controls.autocomplete import AutocompleteState


def _texts(fragments) -> list[str]:
    """Pull the text out of each fragment (handles 2- and 3-tuples)."""
    return [frag[1] for frag in fragments]


# ── D1: sliding window follows the selection ────────────────────────────────


def test_selected_entry_is_visible_past_the_window():
    state = AutocompleteState()
    state.update("/")  # matches every command — more than the 8-item window
    assert len(state.matches) > 8

    # Navigate well past the first window.
    for _ in range(10):
        state.select_next()
    assert state.selected_index == 10

    fragments = state.get_fragments()
    selected_cmd = state.matches[state.selected_index][0]
    # The highlighted command fragment carries the "#ffffff bold" style.
    highlighted = [frag[1] for frag in fragments if "#ffffff bold" in frag[0]]
    assert highlighted == [f" {selected_cmd} "]


def test_window_never_exceeds_eight_entries():
    state = AutocompleteState()
    state.update("/")
    for _ in range(15):
        state.select_next()
    fragments = state.get_fragments()
    # Each entry contributes a command fragment styled bold/ACCENT; count the
    # newline separators instead: an N-entry window has N-1 newlines.
    newlines = _texts(fragments).count("\n")
    assert newlines == 8 - 1


# ── D2: newline is a separator, never trailing ──────────────────────────────


def test_single_entry_has_no_newline():
    state = AutocompleteState()
    state.update("/think")  # exactly one command starts with "/think"
    assert len(state.matches) == 1
    texts = _texts(state.get_fragments())
    assert "\n" not in texts


def test_newlines_separate_entries_only_no_trailing():
    state = AutocompleteState()
    state.update("/models")  # /models, /models list, /models set, /models manage
    n = len(state.matches[:8])
    assert n >= 2
    texts = _texts(state.get_fragments())
    assert texts.count("\n") == n - 1
    assert texts[-1] != "\n"  # no trailing blank line
