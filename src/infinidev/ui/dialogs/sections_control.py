"""Settings editor dialog — two-panel interactive settings browser.

Left panel: sections. Right panel: settings with inline editing.
Supports bool (checkbox-style toggle), select, numeric, and string fields.
"""

from __future__ import annotations
from typing import Any, Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.data_structures import Point
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


class SectionsControl(UIControl):
    """Left panel: clickable section list."""

    def __init__(self, state: SettingsEditorState) -> None:
        self._state = state

    def is_focusable(self) -> bool:
        return True

    def mouse_handler(self, mouse_event) -> None:
        if mouse_event.event_type == MouseEventType.MOUSE_UP:
            row = mouse_event.position.y
            if 0 <= row < len(self._state.sections):
                self._state.section_cursor = row
                self._state.setting_cursor = 0
                self._state.focus_panel = "sections"

    def get_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()
        s = self._state

        @kb.add("up")
        def _up(event):
            s.move_section(-1)

        @kb.add("down")
        def _down(event):
            s.move_section(1)

        @kb.add("enter")
        @kb.add("right")
        @kb.add("tab")
        def _enter(event):
            s.focus_panel = "settings"

        return kb

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, section in enumerate(self._state.sections):
            active = self._state.focus_panel == "sections"
            if i == self._state.section_cursor:
                if active:
                    style = f"bg:{PRIMARY} #ffffff bold"
                else:
                    style = f"bg:{SURFACE_LIGHT} {TEXT} bold"
            else:
                style = f"{TEXT}"
            pad = " " * max(0, width - len(section) - 2)
            lines.append([(style, f" {section}{pad} ")])

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []

        # Report a cursor position so the enclosing Window auto-scrolls
        # to keep the selected section visible when the list overflows.
        cursor_row = 0
        if self._state.sections:
            cursor_row = max(
                0, min(self._state.section_cursor, len(self._state.sections) - 1)
            )
        return UIContent(
            get_line=get_line,
            line_count=len(lines),
            cursor_position=Point(x=0, y=cursor_row),
            show_cursor=False,
        )


