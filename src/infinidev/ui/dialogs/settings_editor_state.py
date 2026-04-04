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
    "Thinking": [
        ("THINKING_ENABLED", "Enable reasoning (best-effort for local models)", "bool"),
        ("THINKING_BUDGET", "Thinking budget (local models may ignore)", "select:low,medium,high,ultra,custom"),
        ("THINKING_BUDGET_TOKENS", "Custom token budget (when preset=custom)", "int"),
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
    "UI": [
        ("MARKDOWN_MESSAGES", "Render LLM responses with markdown styling", "bool"),
    ],
}


class SettingsEditorState:
    """Full state for the interactive settings editor."""

    def __init__(self, on_save: Callable[[str, str], None] | None = None,
                 on_focus_change: Callable[[str], None] | None = None,
                 on_edit_start: Callable[[], None] | None = None) -> None:
        self.sections = list(SETTINGS_SECTIONS.keys())
        self.section_cursor: int = 0
        self.setting_cursor: int = 0
        self.editing: bool = False  # True when editing a value
        self._focus_panel: str = "sections"  # "sections" or "settings"
        self._on_save = on_save
        self._on_focus_change = on_focus_change
        self._on_edit_start = on_edit_start
        self._ollama_models: list[str] | None = None  # cached model list
        self._pending_changes: dict[str, str] = {}  # unsaved changes for cross-field deps

        # Dropdown picker state
        self.dropdown_open: bool = False
        self.dropdown_options: list[str] = []      # all options (unfiltered)
        self.dropdown_cursor: int = 0
        self._dropdown_key: str = ""               # which setting key the dropdown is for
        self.dropdown_filter: str = ""             # search text for filtering

        # Edit buffer for string/int/float editing
        self.edit_buffer = Buffer(
            name="setting-edit",
            multiline=False,
            accept_handler=lambda buff: self.confirm_edit(),
        )

    @property
    def focus_panel(self) -> str:
        return self._focus_panel

    @focus_panel.setter
    def focus_panel(self, value: str) -> None:
        if value != self._focus_panel:
            self._focus_panel = value
            if self._on_focus_change:
                self._on_focus_change(value)

    @property
    def current_section(self) -> str:
        return self.sections[self.section_cursor] if self.sections else ""

    @property
    def current_settings(self) -> list[tuple[str, str, str]]:
        return SETTINGS_SECTIONS.get(self.current_section, [])

    @property
    def current_setting(self) -> tuple[str, str, str] | None:
        settings = self.current_settings
        if 0 <= self.setting_cursor < len(settings):
            return settings[self.setting_cursor]
        return None

    def _get_value(self, key: str) -> Any:
        try:
            from infinidev.config.settings import settings
            return getattr(settings, key, "?")
        except Exception:
            return "?"

    def move_section(self, delta: int) -> None:
        self.section_cursor = max(0, min(len(self.sections) - 1, self.section_cursor + delta))
        self.setting_cursor = 0
        self.editing = False

    def move_setting(self, delta: int) -> None:
        settings = self.current_settings
        if settings:
            self.setting_cursor = max(0, min(len(settings) - 1, self.setting_cursor + delta))
            self.editing = False

    def _get_select_options(self, stype: str) -> list[str]:
        """Get options for a select field, including dynamic ones."""
        if stype == "select_dynamic:ollama_models":
            return self._fetch_provider_models()
        if stype == "select_dynamic:provider_models":
            return self._fetch_provider_models()
        if stype.startswith("select:"):
            return stype[7:].split(",")
        return []

    def _is_free_text_model(self) -> bool:
        """Return True if the current provider uses free-text model input."""
        from infinidev.config.settings import settings
        from infinidev.config.providers import get_provider
        provider_id = self._pending_changes.get("LLM_PROVIDER", settings.LLM_PROVIDER)
        provider = get_provider(provider_id)
        return provider.model_list_format == "free_text"

    def _fetch_provider_models(self) -> list[str]:
        """Fetch available models for the current provider (cached)."""
        if self._ollama_models is not None:
            return self._ollama_models
        try:
            from infinidev.config.settings import settings
            from infinidev.config.providers import fetch_models
            provider_id = self._pending_changes.get("LLM_PROVIDER", settings.LLM_PROVIDER)
            api_key = self._pending_changes.get("LLM_API_KEY", settings.LLM_API_KEY)
            base_url = self._pending_changes.get("LLM_BASE_URL", settings.LLM_BASE_URL)
            self._ollama_models = fetch_models(provider_id, api_key, base_url)
        except Exception:
            self._ollama_models = []
        return self._ollama_models

    def activate(self) -> None:
        """Enter/Space on current setting: toggle bool, start editing, or cycle select."""
        setting = self.current_setting
        if not setting:
            return

        key, desc, stype = setting
        value = self._get_value(key)

        if stype == "bool":
            # Toggle immediately
            new_val = not bool(value)
            self._save(key, str(new_val))

        elif stype.startswith("select:") or stype.startswith("select_dynamic:"):
            # For free-text providers (e.g. openai_compatible), use inline text editor
            if key == "LLM_MODEL" and self._is_free_text_model():
                self.editing = True
                self.edit_buffer.set_document(
                    Document(str(value)), bypass_readonly=True
                )
                if self._on_edit_start:
                    self._on_edit_start()
                return

            # Open dropdown picker
            options = self._get_select_options(stype)
            if not options:
                # No models found (server down?) — fall back to text input
                if key == "LLM_MODEL":
                    self.editing = True
                    self.edit_buffer.set_document(
                        Document(str(value)), bypass_readonly=True
                    )
                    if self._on_edit_start:
                        self._on_edit_start()
                return
            self.dropdown_open = True
            self.dropdown_options = options
            self._dropdown_key = key
            current = str(value)
            try:
                self.dropdown_cursor = options.index(current)
            except ValueError:
                self.dropdown_cursor = 0
            if self._on_focus_change:
                self._on_focus_change("dropdown")

        else:
            # Block editing on THINKING_BUDGET_TOKENS when preset is not "custom"
            if key == "THINKING_BUDGET_TOKENS":
                budget = str(self._get_value("THINKING_BUDGET")).lower()
                if budget != "custom":
                    return  # Read-only — value is controlled by the preset

            # Start inline editing for str/int/float
            self.editing = True
            self.edit_buffer.set_document(
                Document(str(value)), bypass_readonly=True
            )
            if self._on_edit_start:
                self._on_edit_start()

    def confirm_edit(self) -> None:
        """Confirm the current edit."""
        if not self.editing:
            return
        setting = self.current_setting
        if not setting:
            return
        key, desc, stype = setting
        self._save(key, self.edit_buffer.text)
        self.editing = False

    def cancel_edit(self) -> None:
        self.editing = False

    @property
    def dropdown_filtered(self) -> list[str]:
        """Return dropdown options filtered by search text (case-insensitive)."""
        if not self.dropdown_filter:
            return self.dropdown_options
        query = self.dropdown_filter.lower()
        return [o for o in self.dropdown_options if query in o.lower()]

    def dropdown_move(self, delta: int) -> None:
        filtered = self.dropdown_filtered
        if filtered:
            self.dropdown_cursor = max(0, min(len(filtered) - 1,
                                              self.dropdown_cursor + delta))

    def dropdown_type(self, char: str) -> None:
        """Append a character to the search filter."""
        self.dropdown_filter += char
        self.dropdown_cursor = 0

    def dropdown_backspace(self) -> None:
        """Delete last character from search filter."""
        if self.dropdown_filter:
            self.dropdown_filter = self.dropdown_filter[:-1]
            self.dropdown_cursor = 0

    def dropdown_confirm(self) -> None:
        filtered = self.dropdown_filtered
        if self.dropdown_open and 0 <= self.dropdown_cursor < len(filtered):
            selected = filtered[self.dropdown_cursor]
            self._save(self._dropdown_key, selected)
        self.dropdown_close()

    def dropdown_close(self) -> None:
        self.dropdown_open = False
        self.dropdown_options = []
        self.dropdown_filter = ""
        self._dropdown_key = ""
        if self._on_focus_change:
            self._on_focus_change("settings")

    def _save(self, key: str, value: str) -> None:
        """Save a setting value."""
        try:
            from infinidev.config.settings import settings, reload_all
            # Convert type
            setting = next((s for s in self.current_settings if s[0] == key), None)
            if setting:
                stype = setting[2]
                if stype == "bool":
                    converted = value.lower() in ("true", "1", "yes")
                    settings.save_user_settings({key: converted})
                elif stype == "int":
                    settings.save_user_settings({key: int(value)})
                elif stype == "float":
                    settings.save_user_settings({key: float(value)})
                else:
                    settings.save_user_settings({key: value})
                reload_all()

                # When provider changes, auto-fill base_url and clear model cache
                if key == "LLM_PROVIDER":
                    self._on_provider_change(value)

        except Exception:
            pass
        if self._on_save:
            self._on_save(key, value)

    def _on_provider_change(self, provider_id: str) -> None:
        """Update related settings when provider changes."""
        from infinidev.config.settings import settings, reload_all
        from infinidev.config.providers import get_provider
        provider = get_provider(provider_id)

        updates: dict = {}
        # Auto-fill base_url from provider defaults
        if provider.default_base_url:
            updates["LLM_BASE_URL"] = provider.default_base_url
        # Clear model — old model won't be valid for new provider
        updates["LLM_MODEL"] = ""
        # Set API key: "ollama" placeholder for local, empty for cloud providers
        if not provider.api_key_required:
            updates["LLM_API_KEY"] = "ollama"
        else:
            # Clear the key so user must enter their own — don't send "ollama" to cloud APIs
            updates["LLM_API_KEY"] = ""

        if updates:
            settings.save_user_settings(updates)
            reload_all()
            # Track pending changes for model fetch
            self._pending_changes.update(updates)

        # Clear cached model list so it re-fetches for new provider
        self._ollama_models = None


