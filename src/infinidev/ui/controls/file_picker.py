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


from infinidev.ui.controls.file_picker_results_control import FilePickerResultsControl
from infinidev.ui.controls.file_picker_state import FilePickerState
