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
from infinidev.ui.controls.directory_tree import _TreeNode


class DirectoryTreeControl(UIControl):
    """Custom UIControl for the file explorer tree.

    Renders a navigable tree with icons, dirty markers, and keyboard navigation.
    """

    def __init__(self, root_path: str,
                 on_file_selected: Callable[[str], None] | None = None) -> None:
        self._root = _TreeNode(os.path.basename(root_path), root_path, True)
        self._root.expanded = True
        self._root.load_children()
        self._on_file_selected = on_file_selected
        self._dirty_paths: set[str] = set()
        self._cursor: int = 0
        self._flat_nodes: list[tuple[int, _TreeNode]] = []  # (depth, node)
        self._rebuild_flat()
        self._kb = self._create_keybindings()

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event: MouseEvent) -> None:
        """Handle mouse: click = select, double-click = open, scroll = navigate."""
        import time

        if mouse_event.event_type == MouseEventType.SCROLL_UP:
            self.move_cursor(-3)
            return None

        if mouse_event.event_type == MouseEventType.SCROLL_DOWN:
            self.move_cursor(3)
            return None

        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            row = mouse_event.position.y
            if 0 <= row < len(self._flat_nodes):
                now = time.monotonic()
                # Double-click detection: same row within 0.4s
                if (row == self._cursor
                        and hasattr(self, '_last_click_time')
                        and now - self._last_click_time < 0.4):
                    self.activate_cursor()
                    self._last_click_time = 0  # reset to prevent triple
                else:
                    # Single click: just select
                    self._cursor = row
                self._last_click_time = now

    def get_key_bindings(self) -> KeyBindings | None:
        """Return navigation key bindings for this control."""
        return self._kb

    def _create_keybindings(self) -> KeyBindings:
        """Build the key bindings for tree navigation."""
        kb = KeyBindings()
        tree = self

        @kb.add("up")
        def _up(event):
            tree.move_cursor(-1)

        @kb.add("down")
        def _down(event):
            tree.move_cursor(1)

        @kb.add("enter")
        def _enter(event):
            tree.activate_cursor()

        @kb.add("right")
        def _expand(event):
            if not tree._flat_nodes:
                return
            _, node = tree._flat_nodes[tree._cursor]
            if node.is_dir and not node.expanded:
                node.toggle()
                tree._rebuild_flat()

        @kb.add("left")
        def _collapse(event):
            if not tree._flat_nodes:
                return
            _, node = tree._flat_nodes[tree._cursor]
            if node.is_dir and node.expanded:
                node.toggle()
                tree._rebuild_flat()

        @kb.add("r")
        def _refresh(event):
            tree.refresh()

        return kb

    def set_dirty_paths(self, paths: set[str]) -> None:
        self._dirty_paths = paths

    def _rebuild_flat(self) -> None:
        """Flatten the tree into a visible-nodes list."""
        flat: list[tuple[int, _TreeNode]] = []

        def _walk(node: _TreeNode, depth: int) -> None:
            flat.append((depth, node))
            if node.is_dir and node.expanded:
                for child in node.children:
                    _walk(child, depth + 1)

        for child in self._root.children:
            _walk(child, 0)
        self._flat_nodes = flat
        if self._cursor >= len(self._flat_nodes):
            self._cursor = max(0, len(self._flat_nodes) - 1)

    def move_cursor(self, delta: int) -> None:
        self._cursor = max(0, min(len(self._flat_nodes) - 1, self._cursor + delta))

    def activate_cursor(self) -> None:
        """Enter key: toggle dir or select file."""
        if not self._flat_nodes:
            return
        _, node = self._flat_nodes[self._cursor]
        if node.is_dir:
            node.toggle()
            self._rebuild_flat()
        elif self._on_file_selected:
            self._on_file_selected(node.path)

    def refresh(self) -> None:
        """Reload the tree from disk."""
        self._root.loaded = False
        self._root.children.clear()
        self._root.load_children()
        self._rebuild_flat()

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, (depth, node) in enumerate(self._flat_nodes):
            line = self._render_node(depth, node, selected=(i == self._cursor))
            lines.append(line)

        if not lines:
            lines = [[(f"{TEXT_MUTED}", " (empty)")]]

        def get_line(i: int) -> list[tuple[str, str]]:
            if 0 <= i < len(lines):
                return lines[i]
            return []

        from prompt_toolkit.data_structures import Point
        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=self._cursor),
            show_cursor=False,
        )

    def _render_node(self, depth: int, node: _TreeNode,
                     selected: bool) -> list[tuple[str, str]]:
        """Render a single tree node as styled fragments."""
        indent = "  " * depth
        is_dirty = node.path in self._dirty_paths
        is_hidden = node.name.startswith(".")

        fragments: list[tuple[str, str]] = []

        # Selection highlight
        bg = f" bg:{PRIMARY}" if selected else ""

        # Indent
        if indent:
            fragments.append((f"{TEXT_DIM}{bg}", indent))

        # Icon
        if node.is_dir:
            icon = ICON_FOLDER_OPEN if node.expanded else ICON_FOLDER_CLOSED
            fragments.append((f"{TEXT} bold{bg}", icon))
        else:
            icon = _get_file_icon(node.name)
            fragments.append((f"{TEXT}{bg}", icon))

        # Dirty marker
        if is_dirty:
            fragments.append((f"{WARNING} bold{bg}", "\u25cf "))

        # Name
        if is_hidden:
            name_style = f"{EXPLORER_HIDDEN} italic{bg}"
        elif node.is_dir:
            name_style = f"{TEXT} bold{bg}"
        else:
            name_style = f"{TEXT}{bg}"

        name = node.name
        ext = pathlib.Path(name).suffix
        if ext and not node.is_dir:
            base = name[: -len(ext)]
            fragments.append((name_style, base))
            fragments.append((f"{TEXT_MUTED}{bg}", ext))
        else:
            fragments.append((name_style, name))

        return fragments

