"""Collapsible file diff display as FormattedText.

Replaces the Textual FileChangeDiffWidget with a pure FormattedText renderer.
Each diff is stored as a dict in the chat messages list with type="diff".
"""

from __future__ import annotations

import os
import re

from infinidev.ui.theme import (
    STYLE_DIFF_REMOVED, STYLE_DIFF_ADDED, STYLE_DIFF_HUNK,
    STYLE_DIFF_HEADER, STYLE_DIFF_TITLE,
    TEXT, TEXT_MUTED, DIFF_REMOVED, DIFF_ADDED, DIFF_HUNK,
    DIFF_HEADER, DIFF_TITLE_FG, DIFF_TITLE_BG,
)


def colorize_diff_fragments(diff_text: str) -> list[list[tuple[str, str]]]:
    """Convert a unified diff to prompt_toolkit style fragment lines.

    Returns a list of lines, where each line is a list of (style, text) tuples.
    """
    lines: list[list[tuple[str, str]]] = []
    old_num = 0
    new_num = 0
    in_hunk = False

    for line in diff_text.splitlines():
        if not in_hunk and (line.startswith("---") or line.startswith("+++")):
            lines.append([(f"{DIFF_HEADER} bold", f"     {line}")])
        elif line.startswith("@@"):
            in_hunk = True
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            if match:
                old_num = int(match.group(1)) - 1
                new_num = int(match.group(2)) - 1
            lines.append([(f"{DIFF_HUNK} bold", f"     {line}")])
        elif line.startswith("-"):
            old_num += 1
            lnum = f"{old_num:>4} "
            lines.append([(f"{DIFF_REMOVED}", f"{lnum}{line}")])
        elif line.startswith("+"):
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append([(f"{DIFF_ADDED}", f"{lnum}{line}")])
        else:
            old_num += 1
            new_num += 1
            lnum = f"{new_num:>4} "
            lines.append([
                (f"{TEXT_MUTED}", lnum),
                (f"{TEXT}", line),
            ])

    return lines


def colorize_diff_side_by_side(diff_text: str, column_width: int = 40) -> list[list[tuple[str, str]]]:
    """Convert a unified diff to side-by-side prompt_toolkit style fragment lines.

    Each line shows old (left) and new (right) columns separated by '│'.
    ``column_width`` controls the minimum width of each column.
    """
    # Parse unified diff into paired lines
    old_lines: list[tuple[str, str]] = []   # (style_prefix, text)
    new_lines: list[tuple[str, str]] = []
    result_lines: list[list[tuple[str, str]]] = []

    old_num = 0
    new_num = 0
    in_hunk = False
    half_w = column_width

    # We need to accumulate context/removed/added lines and pair them
    # Strategy: walk through diff, collect removed and added blocks,
    # then pair them side-by-side.

    entries: list[tuple[str, int | None, int | None, str, str]] = []
    # (kind, old_line_num, new_line_num, style, text)
    # kind: "context", "old", "new", "header", "hunk"

    for line in diff_text.splitlines():
        if not in_hunk and (line.startswith("---") or line.startswith("+++")):
            entries.append(("header", None, None, f"{DIFF_HEADER} bold", line))
        elif line.startswith("@@"):
            in_hunk = True
            match = re.match(r"@@ -(\d+)(?:,\d+)? \+(\d+)", line)
            if match:
                old_num = int(match.group(1)) - 1
                new_num = int(match.group(2)) - 1
            entries.append(("hunk", None, None, f"{DIFF_HUNK} bold", line))
        elif line.startswith("-"):
            old_num += 1
            entries.append(("old", old_num, None, f"{DIFF_REMOVED}", line))
        elif line.startswith("+"):
            new_num += 1
            entries.append(("new", None, new_num, f"{DIFF_ADDED}", line))
        else:
            old_num += 1
            new_num += 1
            entries.append(("context", old_num, new_num, f"{TEXT}", line))

    # Now build side-by-side pairs
    i = 0
    while i < len(entries):
        kind = entries[i][0]

        if kind in ("header", "hunk"):
            _, _, _, style, text = entries[i]
            result_lines.append([(style, f"     {text}")])
            i += 1
            continue

        if kind == "context":
            _, old_n, new_n, style, text = entries[i]
            left_text = f"{old_n:>4} {text}" if old_n else f"     {text}"
            right_text = f"{new_n:>4} {text}" if new_n else f"     {text}"
            _add_side_by_side_line(result_lines, left_text, right_text, half_w,
                                   style, style)
            i += 1
            continue

        # Collect consecutive old and new lines
        old_block: list[tuple[int, str, str]] = []  # (line_num, style, text)
        new_block: list[tuple[int, str, str]] = []

        while i < len(entries) and entries[i][0] == "old":
            _, old_n, _, style, text = entries[i]
            old_block.append((old_n, style, text))
            i += 1
        while i < len(entries) and entries[i][0] == "new":
            _, _, new_n, style, text = entries[i]
            new_block.append((new_n, style, text))
            i += 1

        # Pair them up
        max_len = max(len(old_block), len(new_block))
        for j in range(max_len):
            if j < len(old_block):
                old_n, style, text = old_block[j]
                left_text = f"{old_n:>4} {text}"
                left_style = style
            else:
                left_text = ""
                left_style = ""
            if j < len(new_block):
                new_n, style, text = new_block[j]
                right_text = f"{new_n:>4} {text}"
                right_style = style
            else:
                right_text = ""
                right_style = ""
            _add_side_by_side_line(result_lines, left_text, right_text, half_w,
                                   left_style, right_style)

    return result_lines


def _add_side_by_side_line(
    result: list[list[tuple[str, str]]],
    left_text: str,
    right_text: str,
    half_w: int,
    left_style: str,
    right_style: str,
) -> None:
    """Append a single side-by-side line to *result*."""
    sep = " │ "
    # Truncate/pad left column
    if left_style:
        # Strip ANSI/content to measure visible width
        vis_left = _visible_len(left_text)
        if vis_left > half_w:
            left_text = left_text[:half_w]
            pad_l = ""
        else:
            pad_l = " " * (half_w - vis_left)
    else:
        pad_l = " " * max(0, half_w - len(left_text))

    fragments: list[tuple[str, str]] = []
    if left_style:
        fragments.append((left_style, f" {left_text}{pad_l}"))
    else:
        fragments.append(("", f" {left_text}{pad_l}"))
    fragments.append((f"{TEXT_MUTED}", sep))
    if right_style:
        fragments.append((right_style, f" {right_text}"))
    else:
        fragments.append(("", f" {right_text}"))
    result.append(fragments)


def _visible_len(text: str) -> int:
    """Return the visible length of text (strips leading line-number prefix)."""
    # Text like "   12 -removed content" — we just count characters
    return len(text)



def diff_header_fragments(file_path: str, action: str = "modified",
                          change_count: int = 1) -> list[tuple[str, str]]:
    """Render a diff header line (collapsible title)."""
    icon = "+" if action == "created" else "~"
    basename = os.path.basename(file_path)
    count_str = f" ({change_count} edits)" if change_count > 1 else ""
    return [
        (f"{DIFF_TITLE_FG} bg:{DIFF_TITLE_BG}", f" {icon} {basename}{count_str} "),
    ]


class DiffState:
    """Tracks diff data for a file, including collapsed/expanded state."""

    def __init__(self, file_path: str, diff_text: str,
                 action: str = "modified", change_count: int = 1) -> None:
        self.file_path = file_path
        self.diff_text = diff_text
        self.action = action
        self.change_count = change_count
        self.collapsed = True
        self._diff_lines: list[list[tuple[str, str]]] | None = None

    def update(self, diff_text: str, action: str = "modified",
               change_count: int = 1) -> None:
        self.diff_text = diff_text
        self.action = action
        self.change_count = change_count
        self._diff_lines = None  # invalidate cache

    def get_lines(self) -> list[list[tuple[str, str]]]:
        """Get all display lines (header + diff if expanded)."""
        header = diff_header_fragments(self.file_path, self.action, self.change_count)
        lines = [header]

        if not self.collapsed:
            if self._diff_lines is None:
                self._diff_lines = colorize_diff_fragments(self.diff_text)
            lines.extend(self._diff_lines)

        return lines
