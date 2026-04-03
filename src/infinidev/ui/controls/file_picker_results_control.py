"""Quick file picker (Ctrl+P) — VS Code / JetBrains style.

A floating dialog with a search input and fuzzy-filtered file list.
Files are discovered by walking the project directory (respecting
.gitignore via pathspec). Typing filters the list in real-time.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any, Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import BufferControl, UIControl, UIContent

from infinidev.ui.theme import TEXT, TEXT_MUTED, PRIMARY, ACCENT, SURFACE_LIGHT


# Directories to always skip (fast reject before pathspec).
_SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", "egg-info", ".eggs", ".infinidev",
}

# Max files to index (safety valve for huge repos).
_MAX_FILES = 10_000

# Max results to display at once.
_MAX_RESULTS = 20


def _discover_files(root: str) -> list[str]:
    """Walk the project tree and return relative file paths.

    Skips common non-source directories and respects .gitignore
    if pathspec is available.
    """
    root_path = Path(root)
    ignore_spec = None

    # Try to load .gitignore patterns
    gitignore = root_path / ".gitignore"
    if gitignore.is_file():
        try:
            import pathspec
            with open(gitignore, "r") as f:
                ignore_spec = pathspec.PathSpec.from_lines("gitwildmatch", f)
        except ImportError:
            pass  # pathspec not installed — skip .gitignore filtering

    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in _SKIP_DIRS and not d.startswith(".")
        ]

        rel_dir = os.path.relpath(dirpath, root)
        if rel_dir == ".":
            rel_dir = ""

        for fname in filenames:
            if fname.startswith("."):
                continue
            rel_path = os.path.join(rel_dir, fname) if rel_dir else fname
            if ignore_spec and ignore_spec.match_file(rel_path):
                continue
            files.append(rel_path)
            if len(files) >= _MAX_FILES:
                return files

    return files


def _fuzzy_match(query: str, path: str) -> int | None:
    """Simple fuzzy match: all query chars must appear in order in path.

    Returns a score (lower is better) or None if no match.
    Prefers: exact basename match > path contains query > fuzzy.
    """
    path_lower = path.lower()
    query_lower = query.lower()

    # Exact substring match in filename (best)
    basename = os.path.basename(path_lower)
    if query_lower in basename:
        return len(basename) - len(query_lower)

    # Exact substring match in full path
    if query_lower in path_lower:
        return len(path_lower) - len(query_lower) + 100

    # Fuzzy: all chars in order
    idx = 0
    gaps = 0
    for char in query_lower:
        found = path_lower.find(char, idx)
        if found == -1:
            return None
        gaps += found - idx
        idx = found + 1

    return gaps + 200


class FilePickerResultsControl(UIControl):
    """Renders the filtered file list with highlighted selection."""

    def __init__(self) -> None:
        self.results: list[str] = []
        self.cursor: int = 0

    def create_content(self, width: int, height: int) -> UIContent:
        visible = self.results[:_MAX_RESULTS]
        if not visible:
            lines = [[(f"{TEXT_MUTED}", "  No files found")]]
            return UIContent(get_line=lambda i: lines[i], line_count=1)

        lines = []
        for i, path in enumerate(visible):
            if i == self.cursor:
                style = f"bg:{ACCENT} #000000 bold"
                prefix = " > "
            else:
                style = f"{TEXT}"
                prefix = "   "

            # Highlight basename vs directory
            dirname = os.path.dirname(path)
            basename = os.path.basename(path)
            if dirname:
                lines.append([
                    (style, prefix),
                    (style + " bold", basename),
                    (f"{TEXT_MUTED}", f"  {dirname}/"),
                ])
            else:
                lines.append([
                    (style, prefix),
                    (style + " bold", basename),
                ])
        return UIContent(get_line=lambda i: lines[i], line_count=len(lines))

    def move_cursor(self, delta: int) -> None:
        max_idx = min(len(self.results), _MAX_RESULTS) - 1
        if max_idx < 0:
            return
        self.cursor = max(0, min(self.cursor + delta, max_idx))

    def get_selected(self) -> str | None:
        visible = self.results[:_MAX_RESULTS]
        if 0 <= self.cursor < len(visible):
            return visible[self.cursor]
        return None

    def is_focusable(self) -> bool:
        return False


