"""In-file search bar for Ctrl+F functionality.

Provides a search input buffer, match navigation (next/prev),
and match count display. Works with a FileEditor's buffer to
highlight and jump to matches.
"""

from __future__ import annotations

import re
from typing import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from infinidev.ui.theme import TEXT, TEXT_MUTED, PRIMARY, ACCENT


class SearchBarState:
    """Manages search state for in-file search."""

    def __init__(self) -> None:
        self.visible: bool = False
        self.query: str = ""
        self.matches: list[int] = []  # list of match start positions
        self.current_match: int = -1
        self._target_buffer: Buffer | None = None

        self.search_buffer = Buffer(
            name="search-bar",
            on_text_changed=self._on_query_changed,
        )
        self.search_control = BufferControl(
            buffer=self.search_buffer,
            focusable=True,
        )

    def set_target(self, buf: Buffer | None) -> None:
        """Set the buffer to search in."""
        self._target_buffer = buf
        self.matches.clear()
        self.current_match = -1

    def toggle(self) -> None:
        self.visible = not self.visible
        if not self.visible:
            self.matches.clear()
            self.current_match = -1

    def _on_query_changed(self, buf: Buffer) -> None:
        """Re-run search when query changes."""
        self.query = buf.text
        self._do_search()

    def _do_search(self) -> None:
        """Find all matches in the target buffer."""
        self.matches.clear()
        self.current_match = -1

        if not self.query or not self._target_buffer:
            return

        text = self._target_buffer.text
        try:
            pattern = re.compile(re.escape(self.query), re.IGNORECASE)
            self.matches = [m.start() for m in pattern.finditer(text)]
            if self.matches:
                self.current_match = 0
                self._goto_current()
        except re.error:
            pass

    def next_match(self) -> None:
        """Jump to the next match."""
        if not self.matches:
            return
        self.current_match = (self.current_match + 1) % len(self.matches)
        self._goto_current()

    def prev_match(self) -> None:
        """Jump to the previous match."""
        if not self.matches:
            return
        self.current_match = (self.current_match - 1) % len(self.matches)
        self._goto_current()

    def _goto_current(self) -> None:
        """Move the target buffer cursor to the current match."""
        if self._target_buffer and 0 <= self.current_match < len(self.matches):
            pos = self.matches[self.current_match]
            self._target_buffer.cursor_position = pos

    def get_status_fragments(self) -> FormattedText:
        """Render the match count status."""
        if not self.query:
            return FormattedText([(f"{TEXT_MUTED}", " Search...")])

        if not self.matches:
            return FormattedText([(f"{TEXT_MUTED}", " No matches")])

        return FormattedText([
            (f"{ACCENT}", f" {self.current_match + 1}/{len(self.matches)} "),
        ])
