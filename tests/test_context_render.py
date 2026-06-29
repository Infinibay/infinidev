"""Tests for the sidebar context/usage bar rendering helpers.

Covers:
- ``_format_count`` compact formatting and the 1000k → 1.0M roll-over
- ``build_usage_bar_fragments`` clamping for out-of-range percentages
"""

from __future__ import annotations

from infinidev.ui.managers.context_render import (
    _format_count,
    build_usage_bar_fragments,
)
from infinidev.ui.theme import BAR_WIDTH, BAR_FILLED


# ── _format_count ──────────────────────────────────────────────────────────


def test_format_count_basic():
    assert _format_count(None) == "?"
    assert _format_count(0) == "0"
    assert _format_count(999) == "999"
    assert _format_count(12_345) == "12k"
    assert _format_count(2_500_000) == "2.5M"


def test_format_count_rolls_over_to_megabytes_at_rounding_boundary():
    # 999_500..999_999 round up to "1000k" under the k branch — show "1.0M".
    assert _format_count(999_499) == "999k"
    assert _format_count(999_500) == "1.0M"
    assert _format_count(999_999) == "1.0M"
    assert _format_count(1_000_000) == "1.0M"


# ── build_usage_bar_fragments clamping ──────────────────────────────────────


def _bar_cells(fragments) -> str:
    """Concatenate just the filled-bar text from the rendered fragments."""
    return "".join(t for _s, t in fragments if BAR_FILLED in t)


def test_usage_bar_lower_clamps_negative_pct():
    # A negative pct must not produce any filled cells.
    frags = build_usage_bar_fragments("Chat", 0, 100, -0.5)
    assert _bar_cells(frags) == ""


def test_usage_bar_upper_clamps_pct_above_one():
    # An over-100% pct must not exceed BAR_WIDTH filled cells.
    frags = build_usage_bar_fragments("Chat", 100, 100, 1.5)
    assert len(_bar_cells(frags)) == BAR_WIDTH
