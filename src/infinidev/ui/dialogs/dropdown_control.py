"""Settings editor dialog — two-panel interactive settings browser.

Left panel: sections. Right panel: settings with inline editing.
Supports bool (checkbox-style toggle), select, numeric, and string fields.
"""

from __future__ import annotations
from typing import Any, Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.containers import HSplit, VSplit, Window, ConditionalContainer
from prompt_toolkit.layout.controls import UIControl, UIContent, BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.filters import Condition
from prompt_toolkit.mouse_events import MouseEventType

from infinidev.ui.theme import (
    PRIMARY, TEXT, TEXT_MUTED, ACCENT, SUCCESS, ERROR,
    SURFACE, SURFACE_LIGHT, SURFACE_DARK,
)
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "settings_editor"

# ── Settings metadata: (key, description, type) ─────────────────────────
# type: "bool", "int", "float", "str", "select:opt1,opt2,opt3"

# The settings metadata lives in `settings_editor_state.py` and only there.
# This module used to carry its own copy of SETTINGS_SECTIONS. Nothing in
# `src/` ever read it — only a test did, which is what made a dead copy look
# maintained. Three copies of one list is how the provider list and the
# prompt-style list each drifted; this one was removed before it could.
from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState


class DropdownControl(UIControl):
    """Floating dropdown picker for select fields."""

    def __init__(self, state: SettingsEditorState) -> None:
        self._state = state

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event) -> None:
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            row = mouse_event.position.y
            if 0 <= row < len(self._state.dropdown_options):
                self._state.dropdown_cursor = row
                self._state.dropdown_confirm()

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        s = self._state

        @kb.add("up")
        def _up(event):
            s.dropdown_move(-1)

        @kb.add("down")
        def _down(event):
            s.dropdown_move(1)

        @kb.add("enter")
        def _confirm(event):
            s.dropdown_confirm()

        @kb.add("escape")
        def _cancel(event):
            s.dropdown_close()

        @kb.add("backspace")
        def _backspace(event):
            s.dropdown_backspace()

        # Typing any printable character adds to the search filter
        @kb.add("<any>")
        def _type(event):
            char = event.data
            if char and len(char) == 1 and char.isprintable():
                s.dropdown_type(char)

        return kb

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        options = self._state.dropdown_filtered
        current_val = str(self._state._get_value(self._state._dropdown_key))
        filter_text = self._state.dropdown_filter
        lines = []

        # Search bar (always visible when dropdown is open)
        if filter_text:
            search_display = f" Search: {filter_text}_ ({len(options)} matches)"
        else:
            search_display = " Type to search..."
        pad = " " * max(0, width - len(search_display))
        lines.append([(f"bg:{SURFACE_LIGHT} {ACCENT} italic", f"{search_display}{pad}")])

        for i, opt in enumerate(options):
            is_current = opt == current_val
            marker = ">" if is_current else " "
            if i == self._state.dropdown_cursor:
                style = f"bg:{PRIMARY} #ffffff bold"
            elif is_current:
                style = f"{ACCENT}"
            else:
                style = f"{TEXT}"
            pad = " " * max(0, width - len(opt) - 4)
            lines.append([(style, f" {marker} {opt}{pad}")])

        if len(lines) == 1:  # only search bar, no results
            lines.append([(f"{TEXT_MUTED}", " No matches")])

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))
