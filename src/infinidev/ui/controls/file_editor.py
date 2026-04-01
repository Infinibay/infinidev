"""File editor control — Buffer per open file with syntax highlighting.

Uses prompt_toolkit's Buffer + BufferControl with PygmentsLexer for syntax
highlighting. Each open file gets its own Buffer instance.
"""

from __future__ import annotations

import os
import pathlib
from typing import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.key_binding import KeyBindings

try:
    from prompt_toolkit.lexers import PygmentsLexer
    from pygments.lexers import get_lexer_for_filename, TextLexer
    HAS_PYGMENTS = True
except ImportError:
    HAS_PYGMENTS = False

class FileEditor:
    """Manages a single open file's editing state."""

    def __init__(self, file_path: str, on_dirty_change: Callable[[str, bool], None] | None = None) -> None:
        self.file_path = file_path
        self.tab_id = file_path  # Use path as tab identifier
        self.name = os.path.basename(file_path)
        self._on_dirty_change = on_dirty_change

        # Read file content
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
        except Exception:
            content = ""

        self._original_content = content
        self._dirty = False

        # Create buffer
        self.buffer = Buffer(
            document=Document(content, cursor_position=0),
            multiline=True,
            name=f"editor-{file_path}",
            on_text_changed=self._on_text_changed,
        )

        # Create control with syntax highlighting
        lexer = None
        if HAS_PYGMENTS:
            try:
                pygments_lexer = get_lexer_for_filename(file_path)
                lexer = PygmentsLexer(type(pygments_lexer))
            except Exception:
                pass

        self.control = BufferControl(
            buffer=self.buffer,
            lexer=lexer,
            focusable=True,
            focus_on_click=True,
        )

    def _on_text_changed(self, buf: Buffer) -> None:
        """Track dirty state on text changes."""
        new_dirty = buf.text != self._original_content
        if new_dirty != self._dirty:
            self._dirty = new_dirty
            if self._on_dirty_change:
                self._on_dirty_change(self.tab_id, self._dirty)

    @property
    def is_dirty(self) -> bool:
        return self._dirty

    def save(self) -> bool:
        """Save the buffer content to disk. Returns True on success."""
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                f.write(self.buffer.text)
            self._original_content = self.buffer.text
            self._dirty = False
            if self._on_dirty_change:
                self._on_dirty_change(self.tab_id, False)
            return True
        except Exception:
            return False

    def goto_line(self, line: int) -> None:
        """Move cursor to a specific line number (1-based)."""
        lines = self.buffer.text.split("\n")
        target = max(0, min(line - 1, len(lines) - 1))
        pos = sum(len(lines[i]) + 1 for i in range(target))
        self.buffer.cursor_position = min(pos, len(self.buffer.text))
