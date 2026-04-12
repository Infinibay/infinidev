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
from infinidev.ui.controls.clickable_scrollbar import scrollable_window

DIALOG_NAME = "settings_editor"

# Settings metadata is defined in settings_editor_state.py (single source of truth).
# Imported here for backward compatibility with code that references it from this module.
from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS


from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState
from infinidev.ui.dialogs.sections_control import SectionsControl
from infinidev.ui.dialogs.settings_control import SettingsControl
from infinidev.ui.dialogs.dropdown_control import DropdownControl

def create_settings_editor(on_save: Callable[[str, str], None] | None = None,
                           on_focus_change: Callable[[str], None] | None = None,
                           on_edit_start: Callable[[], None] | None = None):
    """Create the two-panel settings editor dialog.

    Returns (frame, state, sections_control, settings_control).
    """
    state = SettingsEditorState(on_save=on_save, on_focus_change=on_focus_change,
                                on_edit_start=on_edit_start)
    sections_ctrl = SectionsControl(state)
    settings_ctrl = SettingsControl(state)

    _, settings_container = scrollable_window(
        settings_ctrl, display_arrows=True,
    )

    body = VSplit([
        Window(content=sections_ctrl, width=D(preferred=20)),
        Window(width=1, char="│", style=f"{PRIMARY}"),
        HSplit([
            settings_container,
            # Edit buffer (visible only when editing)
            ConditionalContainer(
                content=Window(
                    content=BufferControl(buffer=state.edit_buffer, focusable=True),
                    height=1,
                    style=f"bg:{SURFACE_LIGHT} #ffffff",
                ),
                filter=Condition(lambda: state.editing),
            ),
        ], width=D(weight=1)),
    ])

    frame = dialog_frame("Settings", body, width=90, height=30, border_color=PRIMARY)
    return frame, state, sections_ctrl, settings_ctrl
