"""Settings control — right panel of the settings editor dialog."""

from __future__ import annotations

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import UIControl, UIContent
from prompt_toolkit.mouse_events import MouseEventType

from infinidev.ui.theme import (
    PRIMARY, TEXT, TEXT_MUTED, ACCENT, SUCCESS,
    SURFACE_LIGHT,
)
from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState

# Preset token values — kept in sync with config/thinking_budget.py
_THINKING_PRESET_TOKENS: dict[str, str] = {
    "low": "1,024",
    "medium": "4,096",
    "high": "16,384",
    "ultra": "unlimited",
}


class SettingsControl(UIControl):
    """Right panel: settings list with inline editing."""

    def __init__(self, state: SettingsEditorState) -> None:
        self._state = state

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event) -> None:
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            # Map click row to setting index (each setting takes 3 lines)
            row = mouse_event.position.y
            idx = row // 3
            settings = self._state.current_settings
            if 0 <= idx < len(settings):
                self._state.setting_cursor = idx
                self._state.focus_panel = "settings"
                self._state.activate()

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        s = self._state

        @kb.add("up")
        def _up(event):
            if not s.editing:
                s.move_setting(-1)

        @kb.add("down")
        def _down(event):
            if not s.editing:
                s.move_setting(1)

        @kb.add("enter")
        @kb.add(" ")
        def _activate(event):
            if s.editing:
                s.confirm_edit()
            else:
                s.activate()

        @kb.add("escape")
        def _escape(event):
            if s.editing:
                s.cancel_edit()
            else:
                s.focus_panel = "sections"

        @kb.add("left")
        @kb.add("s-tab")
        def _back(event):
            if not s.editing:
                s.focus_panel = "sections"

        return kb

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        settings = self._state.current_settings
        lines = []
        active_panel = self._state.focus_panel == "settings"

        for i, (key, desc, stype) in enumerate(settings):
            value = self._state._get_value(key)
            selected = i == self._state.setting_cursor and active_panel

            # Line 1: key name
            if selected:
                key_style = f"bg:{PRIMARY} #ffffff bold"
            else:
                key_style = f"{ACCENT} bold"
            pad = " " * max(0, width - len(key) - 2)
            lines.append([(key_style, f" {key}{pad} ")])

            # Line 2: value with type indicator
            #
            # Special case: THINKING_BUDGET_TOKENS shows the preset value
            # (read-only) unless THINKING_BUDGET is "custom".
            is_budget_tokens = key == "THINKING_BUDGET_TOKENS"
            budget_preset = str(self._state._get_value("THINKING_BUDGET")).lower() if is_budget_tokens else ""
            is_budget_readonly = is_budget_tokens and budget_preset != "custom"

            if is_budget_readonly:
                # Show the preset's token count as read-only
                preset_label = _THINKING_PRESET_TOKENS.get(budget_preset, str(value))
                lines.append([
                    (f"{TEXT}", f"   = "),
                    (f"{TEXT_MUTED}", f"{preset_label}"),
                    (f"{TEXT_MUTED}", f"  (set by {budget_preset} preset)"),
                ])
            elif selected and self._state.editing:
                # Show edit buffer content
                edit_text = self._state.edit_buffer.text
                lines.append([
                    (f"bg:{SURFACE_LIGHT} {TEXT}", f"   > "),
                    (f"bg:{SURFACE_LIGHT} #ffffff bold", f"{edit_text}"),
                    (f"bg:{SURFACE_LIGHT} {TEXT_MUTED}", " (Enter=save, Esc=cancel)"),
                ])
            elif stype == "bool":
                bool_val = bool(value)
                indicator = f"[{'x' if bool_val else ' '}]"
                color = SUCCESS if bool_val else TEXT_MUTED
                lines.append([
                    (f"   {color} bold", f"   {indicator} "),
                    (f"{TEXT}", f"{'Enabled' if bool_val else 'Disabled'}"),
                ])
            elif stype.startswith("select:") or stype.startswith("select_dynamic:"):
                current_val = str(value)
                # Show as plain text (no ▾) for free-text model fields
                if key == "LLM_MODEL" and self._state._is_free_text_model():
                    lines.append([
                        (f"{TEXT}", f"   = "),
                        (f"{ACCENT} bold", f"{current_val}"),
                        (f"{TEXT_MUTED}", "  (free text)"),
                    ])
                else:
                    lines.append([
                        (f"{TEXT}", f"   = "),
                        (f"{ACCENT} bold", f"{current_val}"),
                        (f"{TEXT_MUTED}", "  ▾"),
                    ])
            else:
                val_str = str(value)
                lines.append([(f"{TEXT}", f"   = {val_str}")])

            # Line 3: description
            lines.append([(f"{TEXT_MUTED}", f"   {desc}")])

        if not lines:
            lines = [[(f"{TEXT_MUTED}", " No settings in this section")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []
        return UIContent(get_line=get_line, line_count=len(lines))


