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
        ("LLM_PROVIDER", "LLM provider", "select:ollama,llama_cpp,vllm,openai,anthropic,gemini,zai,kimi,minimax,openrouter,qwen,openai_compatible"),
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


