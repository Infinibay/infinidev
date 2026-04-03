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

SETTINGS_SECTIONS: dict[str, list[tuple[str, str, str]]] = {
    "LLM": [
        ("LLM_PROVIDER", "LLM provider", "select:ollama,llama_cpp,vllm,openai,anthropic,gemini,zai,kimi,minimax,openrouter,openai_compatible"),
        ("LLM_MODEL", "LLM model", "select_dynamic:provider_models"),
        ("LLM_BASE_URL", "API base URL", "str"),
        ("LLM_API_KEY", "API key for the LLM provider", "str"),
        ("LLM_TIMEOUT", "LLM request timeout in seconds", "int"),
    ],
    "Embedding": [
        ("EMBEDDING_PROVIDER", "Embedding provider", "select:ollama,openai,huggingface"),
        ("EMBEDDING_MODEL", "Embedding model name", "str"),
        ("EMBEDDING_BASE_URL", "Embedding API base URL", "str"),
    ],
    "Loop Engine": [
        ("LOOP_MAX_ITERATIONS", "Max plan-execute cycles per task", "int"),
        ("LOOP_MAX_TOOL_CALLS_PER_ACTION", "Tool calls per step (0=unlimited)", "int"),
        ("LOOP_MAX_TOTAL_TOOL_CALLS", "Total tool calls per task", "int"),
        ("LOOP_HISTORY_WINDOW", "Summaries to retain (0=all)", "int"),
        ("LOOP_STEP_NUDGE_THRESHOLD", "Nudge after N tool calls without step_complete", "int"),
        ("LOOP_SUMMARIZER_ENABLED", "Use dedicated LLM call for summaries", "bool"),
    ],
    "Phases": [
        ("ANALYSIS_ENABLED", "Enable analysis phase before execution", "bool"),
        ("REVIEW_ENABLED", "Enable code review after execution", "bool"),
        ("GATHER_ENABLED", "Enable gather phase (deep context)", "bool"),
    ],
    "Permissions": [
        ("EXECUTE_COMMANDS_PERMISSION", "Shell command permission mode", "select:auto_approve,ask,allowed_list"),
        ("FILE_OPERATIONS_PERMISSION", "File operations permission mode", "select:auto_approve,ask,allowed_paths"),
    ],
    "Sandbox": [
        ("SANDBOX_ENABLED", "Enable sandboxed execution", "bool"),
    ],
    "Timeouts": [
        ("COMMAND_TIMEOUT", "Shell command timeout (seconds)", "int"),
        ("WEB_TIMEOUT", "Web request timeout (seconds)", "int"),
        ("CODE_INTERPRETER_TIMEOUT", "Code interpreter timeout (seconds)", "int"),
    ],
    "File Limits": [
        ("MAX_FILE_SIZE_BYTES", "Max file size for reads (bytes)", "int"),
        ("MAX_DIR_LISTING", "Max entries in directory listing", "int"),
    ],
    "Code Intel": [
        ("CODE_INTEL_ENABLED", "Enable tree-sitter indexing", "bool"),
        ("CODE_INTEL_AUTO_INDEX", "Auto-index on file changes", "bool"),
        ("CODE_INTEL_MAX_FILE_SIZE", "Max file size for indexing (bytes)", "int"),
    ],
    "Tree Engine": [
        ("TREE_MAX_NODES", "Max nodes in exploration tree", "int"),
        ("TREE_MAX_DEPTH", "Max tree depth", "int"),
        ("TREE_MAX_LLM_CALLS", "Max LLM calls per exploration", "int"),
        ("TREE_MAX_TOOL_CALLS", "Max tool calls per exploration", "int"),
    ],
}
from infinidev.ui.dialogs.settings_editor_state import SettingsEditorState


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
            if selected and self._state.editing:
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


