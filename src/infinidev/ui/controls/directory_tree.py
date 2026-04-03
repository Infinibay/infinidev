"""Directory tree control for the file explorer.

Custom UIControl that renders a navigable file tree with:
- File-type icons (100+ extensions + 30 special filenames)
- Expand/collapse directories with Enter
- Dirty file indicators
- Lazy directory loading
- Hidden file dimming
"""

from __future__ import annotations

import os
import pathlib
from typing import Any, Callable

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from infinidev.ui.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, PRIMARY, ACCENT, WARNING,
    EXPLORER_HIDDEN,
)


# ── Icons ────────────────────────────────────────────────────────────────

ICON_FOLDER_CLOSED = "\U0001f4c1 "    # 📁
ICON_FOLDER_OPEN = "\U0001f4c2 "      # 📂
ICON_FILE = "\U0001f4c4 "             # 📄

# Extension → icon
FILE_ICONS: dict[str, str] = {
    ".py": "\U0001f40d ", ".pyw": "\U0001f40d ", ".pyx": "\U0001f40d ", ".pyi": "\U0001f40d ",
    ".js": "\u26a1 ", ".mjs": "\u26a1 ", ".cjs": "\u26a1 ",
    ".ts": "\U0001f4d8 ", ".tsx": "\U0001f4d8 ", ".jsx": "\u26a1 ",
    ".html": "\U0001f310 ", ".htm": "\U0001f310 ",
    ".css": "\U0001f3a8 ", ".scss": "\U0001f3a8 ", ".sass": "\U0001f3a8 ", ".less": "\U0001f3a8 ",
    ".svg": "\U0001f3a8 ",
    ".json": "\U0001f4cb ", ".toml": "\u2699\ufe0f ", ".yaml": "\u2699\ufe0f ",
    ".yml": "\u2699\ufe0f ", ".xml": "\U0001f4c3 ", ".ini": "\u2699\ufe0f ",
    ".cfg": "\u2699\ufe0f ", ".conf": "\u2699\ufe0f ", ".env": "\U0001f512 ",
    ".properties": "\u2699\ufe0f ",
    ".md": "\U0001f4d6 ", ".mdx": "\U0001f4d6 ", ".rst": "\U0001f4d6 ",
    ".txt": "\U0001f4c4 ", ".log": "\U0001f4dc ", ".csv": "\U0001f4ca ",
    ".sh": "\U0001f4bb ", ".bash": "\U0001f4bb ", ".zsh": "\U0001f4bb ",
    ".fish": "\U0001f4bb ", ".bat": "\U0001f4bb ", ".cmd": "\U0001f4bb ", ".ps1": "\U0001f4bb ",
    ".rs": "\U0001f980 ", ".go": "\U0001f439 ", ".java": "\u2615 ",
    ".kt": "\U0001f4a0 ", ".scala": "\U0001f534 ",
    ".c": "\U0001f527 ", ".h": "\U0001f527 ", ".cpp": "\U0001f527 ", ".hpp": "\U0001f527 ",
    ".cs": "\U0001f7e3 ", ".swift": "\U0001f3af ", ".rb": "\U0001f48e ",
    ".php": "\U0001f418 ", ".lua": "\U0001f319 ",
    ".r": "\U0001f4c8 ", ".R": "\U0001f4c8 ", ".jl": "\U0001f7e2 ",
    ".ex": "\U0001f7e3 ", ".exs": "\U0001f7e3 ",
    ".erl": "\U0001f7e0 ", ".hs": "\U0001f7e3 ", ".ml": "\U0001f7e0 ",
    ".sql": "\U0001f5c3 ", ".db": "\U0001f5c3 ", ".sqlite": "\U0001f5c3 ",
    ".png": "\U0001f5bc\ufe0f ", ".jpg": "\U0001f5bc\ufe0f ", ".jpeg": "\U0001f5bc\ufe0f ",
    ".gif": "\U0001f5bc\ufe0f ", ".ico": "\U0001f5bc\ufe0f ", ".webp": "\U0001f5bc\ufe0f ",
    ".mp3": "\U0001f3b5 ", ".wav": "\U0001f3b5 ", ".ogg": "\U0001f3b5 ",
    ".mp4": "\U0001f3ac ", ".avi": "\U0001f3ac ", ".mkv": "\U0001f3ac ",
    ".zip": "\U0001f4e6 ", ".tar": "\U0001f4e6 ", ".gz": "\U0001f4e6 ",
    ".7z": "\U0001f4e6 ", ".rar": "\U0001f4e6 ",
    ".pdf": "\U0001f4d5 ", ".doc": "\U0001f4d5 ", ".docx": "\U0001f4d5 ",
    ".xls": "\U0001f4ca ", ".xlsx": "\U0001f4ca ",
    ".woff": "\U0001f520 ", ".woff2": "\U0001f520 ", ".ttf": "\U0001f520 ",
    ".pem": "\U0001f511 ", ".key": "\U0001f511 ", ".crt": "\U0001f511 ",
    ".lock": "\U0001f512 ",
}

# Special filenames → icon
SPECIAL_FILE_ICONS: dict[str, str] = {
    "Dockerfile": "\U0001f433 ", "docker-compose.yml": "\U0001f433 ",
    "docker-compose.yaml": "\U0001f433 ",
    "Makefile": "\U0001f3d7\ufe0f ", "CMakeLists.txt": "\U0001f3d7\ufe0f ",
    "Rakefile": "\U0001f48e ", "Gemfile": "\U0001f48e ", "Gemfile.lock": "\U0001f48e ",
    "Cargo.toml": "\U0001f980 ", "Cargo.lock": "\U0001f980 ",
    "go.mod": "\U0001f439 ", "go.sum": "\U0001f439 ",
    "package.json": "\U0001f4e6 ", "package-lock.json": "\U0001f4e6 ",
    "yarn.lock": "\U0001f4e6 ", "pnpm-lock.yaml": "\U0001f4e6 ",
    "tsconfig.json": "\U0001f4d8 ",
    "requirements.txt": "\U0001f40d ", "setup.py": "\U0001f40d ",
    "setup.cfg": "\U0001f40d ", "pyproject.toml": "\U0001f40d ",
    "Pipfile": "\U0001f40d ", "Pipfile.lock": "\U0001f40d ",
    "LICENSE": "\U0001f4dc ", "LICENSE.md": "\U0001f4dc ", "LICENSE.txt": "\U0001f4dc ",
    ".gitignore": "\U0001f500 ", ".gitmodules": "\U0001f500 ",
    ".gitattributes": "\U0001f500 ", ".dockerignore": "\U0001f433 ",
    ".eslintrc.js": "\U0001f9f9 ", ".eslintrc.json": "\U0001f9f9 ",
    ".prettierrc": "\U0001f9f9 ", ".editorconfig": "\u2699\ufe0f ",
    "CLAUDE.md": "\U0001f916 ",
}

# Directories to hide from tree by default
_HIDDEN_DIRS = {
    "__pycache__", ".git", "node_modules", ".venv", "venv", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "dist", "build",
    ".eggs",
}

_HIDDEN_SUFFIXES = (".egg-info",)


def _get_file_icon(name: str) -> str:
    """Get the appropriate icon for a filename."""
    if name in SPECIAL_FILE_ICONS:
        return SPECIAL_FILE_ICONS[name]
    ext = pathlib.Path(name).suffix.lower()
    return FILE_ICONS.get(ext, ICON_FILE)


class _TreeNode:
    """Internal tree node for directory display."""
    __slots__ = ("name", "path", "is_dir", "expanded", "children", "loaded")

    def __init__(self, name: str, path: str, is_dir: bool) -> None:
        self.name = name
        self.path = path
        self.is_dir = is_dir
        self.expanded = False
        self.children: list[_TreeNode] = []
        self.loaded = False

    def load_children(self) -> None:
        """Lazy-load directory contents."""
        if self.loaded or not self.is_dir:
            return
        self.loaded = True
        try:
            entries = sorted(os.scandir(self.path), key=lambda e: (not e.is_dir(), e.name.lower()))
            for entry in entries:
                # Skip hidden dirs
                if entry.is_dir() and (entry.name in _HIDDEN_DIRS
                        or entry.name.endswith(_HIDDEN_SUFFIXES)):
                    continue
                self.children.append(_TreeNode(entry.name, entry.path, entry.is_dir()))
        except PermissionError:
            pass

    def toggle(self) -> None:
        """Toggle expand/collapse for directories."""
        if self.is_dir:
            self.expanded = not self.expanded
            if self.expanded and not self.loaded:
                self.load_children()


from infinidev.ui.controls.directory_tree_control import DirectoryTreeControl
