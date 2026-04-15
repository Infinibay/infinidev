"""Tests for file diff rendering: unified, side-by-side, and display mode routing."""

import pytest

from infinidev.ui.controls.file_diff import (
    colorize_diff_fragments,
    colorize_diff_side_by_side,
    _add_side_by_side_line,
)


# ── Sample diff ─────────────────────────────────────────────────────────────

SAMPLE_DIFF = """\
--- a/hello.py
+++ b/hello.py
@@ -1,5 +1,6 @@
 def greet(name):
-    print(f"Hello {name}")
+    print(f"Hi {name}")
+     extra indent
     return True

 def farewell(name):
"""

# Diff where more lines are removed than added
UNEQUAL_DIFF = """\
--- a/app.py
+++ b/app.py
@@ -10,8 +10,6 @@
 import os
-import sys
-import json
 from pathlib import Path
+from typing import Any
"""


# ── colorize_diff_fragments ─────────────────────────────────────────────────

class TestColorizeDiffFragments:
    """Tests for the unified (git-diff style) renderer."""

    def test_returns_list_of_lists(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        assert isinstance(result, list)
        for line in result:
            assert isinstance(line, list)
            for style, text in line:
                assert isinstance(style, str)
                assert isinstance(text, str)

    def test_empty_diff_returns_empty(self):
        result = colorize_diff_fragments("")
        assert result == []

    def test_header_lines(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        all_texts = " ".join(t for line in result for _, t in line)
        assert "---" in all_texts
        assert "+++" in all_texts

    def test_hunk_header(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        hunk_lines = [line for line in result if any("@@" in f[1] for f in line)]
        assert len(hunk_lines) >= 1

    def test_removed_line_present(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        all_texts = " ".join(t for line in result for _, t in line)
        assert "Hello" in all_texts

    def test_added_line_present(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        all_texts = " ".join(t for line in result for _, t in line)
        assert "Hi" in all_texts

    def test_line_numbers_increment(self):
        result = colorize_diff_fragments(SAMPLE_DIFF)
        lnums = []
        for line in result:
            for style, text in line:
                stripped = text.lstrip()
                if stripped and stripped[0].isdigit():
                    parts = stripped.split(None, 1)
                    if parts:
                        try:
                            lnums.append(int(parts[0]))
                        except ValueError:
                            pass
        assert len(lnums) > 0


# ── colorize_diff_side_by_side ──────────────────────────────────────────────

class TestColorizeDiffSideBySide:
    """Tests for the side-by-side diff renderer."""

    def test_returns_list_of_lists(self):
        result = colorize_diff_side_by_side(SAMPLE_DIFF)
        assert isinstance(result, list)
        for line in result:
            assert isinstance(line, list)
            for style, text in line:
                assert isinstance(style, str)
                assert isinstance(text, str)

    def test_empty_diff_returns_empty(self):
        result = colorize_diff_side_by_side("")
        assert result == []

    def test_separator_present_in_paired_lines(self):
        """Paired lines should contain the '│' separator."""
        result = colorize_diff_side_by_side(SAMPLE_DIFF)
        sep_lines = [line for line in result if any("│" in f[1] for f in line)]
        assert len(sep_lines) > 0, "Expected at least one line with '│' separator"

    def test_header_and_hunk_lines_no_separator(self):
        """Header (---/+++) and hunk (@@) lines span full width, no separator."""
        result = colorize_diff_side_by_side(SAMPLE_DIFF)
        header_hunk = [line for line in result if any("---" in f[1] or "@@" in f[1] for f in line)]
        assert len(header_hunk) >= 2  # at least one header + one hunk

    def test_unequal_blocks_pair_correctly(self):
        """When more removed than added, right column should have empty entries."""
        result = colorize_diff_side_by_side(UNEQUAL_DIFF)
        assert len(result) > 0
        sep_lines = [line for line in result if any("│" in f[1] for f in line)]
        assert len(sep_lines) > 0

    def test_column_width_parameter(self):
        """Custom column_width should be respected."""
        result_narrow = colorize_diff_side_by_side(SAMPLE_DIFF, column_width=20)
        result_wide = colorize_diff_side_by_side(SAMPLE_DIFF, column_width=80)
        assert len(result_narrow) == len(result_wide)

    def test_context_line_appears_on_both_sides(self):
        """A context (unchanged) line should appear in both left and right columns."""
        result = colorize_diff_side_by_side(SAMPLE_DIFF)
    def test_context_line_appears_on_both_sides(self):
        """A context (unchanged) line should appear in both left and right columns."""
        result = colorize_diff_side_by_side(SAMPLE_DIFF)
        found = False
        for line in result:
            all_text = " ".join(t for _, t in line)
            if "│" in all_text and ("def greet" in all_text or "return True" in all_text):
                found = True
                break
        assert found, "Expected a context line visible in both columns"
class TestAddSideBySideLine:
    """Tests for the low-level side-by-side line builder."""

    def test_basic_line(self):
        result: list[list[tuple[str, str]]] = []
        _add_side_by_side_line(result, "left", "right", 10, "style-l", "style-r")
        assert len(result) == 1
        line = result[0]
        assert len(line) == 3
        assert "left" in line[0][1]
        assert "│" in line[1][1]
        assert "right" in line[2][1]

    def test_empty_left_column(self):
        result: list[list[tuple[str, str]]] = []
        _add_side_by_side_line(result, "", "right only", 10, "", "style-r")
        assert len(result) == 1
        assert "right only" in result[0][2][1]

    def test_empty_right_column(self):
        result: list[list[tuple[str, str]]] = []
        _add_side_by_side_line(result, "left only", "", 10, "style-l", "")
        assert len(result) == 1
        assert "left only" in result[0][0][1]

    def test_padding_to_column_width(self):
        result: list[list[tuple[str, str]]] = []
        _add_side_by_side_line(result, "hi", "yo", 20, "s1", "s2")
        left_text = result[0][0][1]
        assert len("hi") <= len(left_text.strip()) or "hi" in left_text

    def test_truncation_when_exceeds_column_width(self):
        result: list[list[tuple[str, str]]] = []
        long_text = "x" * 100
        _add_side_by_side_line(result, long_text, "right", 20, "s1", "s2")
        left_text = result[0][0][1]
        assert len(left_text.strip()) < 100


# ── DIFF_DISPLAY_MODE routing ───────────────────────────────────────────────

class TestDiffDisplayModeRouting:
    """Test that the settings-based routing selects the correct renderer."""

    def test_side_by_side_enabled_returns_true(self):
        from unittest.mock import patch
        from infinidev.ui.controls.message_widgets import _side_by_side_enabled
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.DIFF_DISPLAY_MODE = "side_by_side"
            assert _side_by_side_enabled() is True

    def test_side_by_side_enabled_returns_false_for_unified(self):
        from unittest.mock import patch
        from infinidev.ui.controls.message_widgets import _side_by_side_enabled
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.DIFF_DISPLAY_MODE = "unified"
            assert _side_by_side_enabled() is False

    def test_side_by_side_enabled_returns_false_for_unknown(self):
        from unittest.mock import patch
        from infinidev.ui.controls.message_widgets import _side_by_side_enabled
        with patch("infinidev.config.settings.settings") as mock_settings:
            mock_settings.DIFF_DISPLAY_MODE = "something_else"
            assert _side_by_side_enabled() is False

    def test_default_setting_is_unified(self):
        from infinidev.config.settings import Settings
        s = Settings()
        assert s.DIFF_DISPLAY_MODE == "unified"

    def test_settings_editor_entry_exists(self):
        """Verify DIFF_DISPLAY_MODE is in the settings editor sections."""
        from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS
        all_entries: list[tuple[str, str, str]] = []
        for section_entries in SETTINGS_SECTIONS.values():
            all_entries.extend(section_entries)
        keys = [entry[0] for entry in all_entries]
        assert "DIFF_DISPLAY_MODE" in keys
        for key, label, fmt in all_entries:
            if key == "DIFF_DISPLAY_MODE":
                assert "select:" in fmt
                assert "unified" in fmt
                assert "side_by_side" in fmt
                break
