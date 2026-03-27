"""TUI implementation for Infinidev using Textual."""

import logging
import os
import pathlib
import re
import uuid
from typing import Any

logger = logging.getLogger(__name__)
from rich.text import Text
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from typing import Optional, Callable, Set
from textual.widgets import (
    Header, Footer, Static, TextArea, Label, OptionList, Markdown,
    DirectoryTree, TabbedContent, TabPane, LoadingIndicator,
    ProgressBar, Button, Select, Input, Checkbox,
)
from textual.widgets.option_list import Option
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from textual import on, events, work

# Infinidev imports
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop_engine import LoopEngine
from infinidev.flows.event_listeners import event_bus
from infinidev.engine.analysis_engine import AnalysisEngine
from infinidev.engine.review_engine import ReviewEngine
from infinidev.db.service import (
    init_db, store_conversation_turn, get_recent_summaries,
)
from infinidev.config.settings import reload_all, Settings
from infinidev.cli.file_watcher import FileWatcher
from infinidev.cli.index_queue import IndexQueue
import infinidev.prompts.flows  # noqa: F401 — registers flows
from infinidev.ui.widgets.context_widgets import QueuedMessageWidget, QueuedMessageStatus
from infinidev.ui.widgets.file_diff_widget import FileChangeDiffWidget, colorize_diff
from infinidev.ui.widgets.image_viewer import ImageViewer, is_image_file

# ── Extension → language map for TextArea syntax highlighting ────────────
_EXT_LANG = {
    ".py": "python", ".pyw": "python",
    ".js": "javascript", ".mjs": "javascript", ".cjs": "javascript",
    ".ts": "javascript", ".tsx": "javascript", ".jsx": "javascript",
    ".json": "json",
    ".html": "html", ".htm": "html",
    ".css": "css",
    ".sql": "sql",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".toml": "toml",
    ".yaml": "yaml", ".yml": "yaml",
    ".xml": "xml", ".svg": "xml",
    ".md": "markdown", ".mdx": "markdown",
    ".regex": "regex",
}

COMMANDS = [
    ("/models", "Show current model configuration"),
    ("/models list", "List available Ollama models"),
    ("/models set", "Change Ollama model (e.g., /models set llama3)"),
    ("/models manage", "Pick a model interactively"),
    ("/settings", "Show or edit settings configuration"),
    ("/settings browse", "Open settings editor modal"),
    ("/think", "Gather context deeply before next task (enables gather phase)"),
    ("/explore", "Decompose and explore a complex problem"),
    ("/brainstorm", "Brainstorm ideas and solutions for a problem"),
    ("/init", "Explore and document the current project"),
    ("/findings", "Browse all findings"),
    ("/knowledge", "Browse project knowledge"),
    ("/documentation", "Browse cached library documentation"),
    ("/docs", "Browse cached library documentation (alias)"),
    ("/clear", "Clear chat history"),
    ("/help", "Show this help"),
    ("/exit", "Exit the CLI"),
    ("/quit", "Exit the CLI"),
]


# ── Reusable widgets ────────────────────────────────────────────────────


class SidebarPanel(Vertical):
    """A panel for the sidebar to show current progress."""
    def __init__(self, title: str, id: str = None):
        super().__init__(id=id)
        self.title_label = Label(f"[bold]{title}[/bold]", classes="sidebar-title")
        self.content_static = Static("", classes="sidebar-content")

    def compose(self) -> ComposeResult:
        yield self.title_label
        yield self.content_static

    def update_content(self, text: str):
        import re
        try:
            from rich.text import Text
            Text.from_markup(text)
            self.content_static.update(text)
        except Exception:
            plain = re.sub(r"\[/?[^\]]*\]", "", text)
            self.content_static.update(plain)


class ContextPanel(Vertical):
    """Widget showing per-window context usage: used, available, and %."""

    def __init__(self, id: str = None):
        super().__init__(id=id)
        self.styles.height = 5
        self.styles.max_height = 5
        self._status: dict[str, Any] = {}
        self._flow: str = ""

    def set_flow(self, flow: str) -> None:
        """Update the current flow indicator."""
        self._flow = flow
        self._refresh()

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="ctx-model")
        yield Static("", id="ctx-chat", classes="ctx-line")
        yield Static("", id="ctx-task", classes="ctx-line")

    def _refresh(self) -> None:
        """Refresh all lines from current status."""
        model = self._status.get("model", "unknown")
        max_ctx = self._status.get("max_context", 4096)
        flow_part = f"  [bold yellow]{self._flow}[/bold yellow]" if self._flow else ""
        self.query_one("#ctx-model", Static).update(
            f"[bold]{model}[/bold] [dim]({max_ctx} ctx)[/dim]{flow_part}"
        )

        chat = self._status.get("chat", {})
        self._update_line("#ctx-chat", "Chat",
                          chat.get("current_tokens", 0),
                          chat.get("remaining_tokens", 0),
                          chat.get("usage_percentage", 0.0))

        tasks = self._status.get("tasks", {})
        self._update_line("#ctx-task", "Task",
                          tasks.get("current_tokens", 0),
                          tasks.get("remaining_tokens", 0),
                          tasks.get("usage_percentage", 0.0))

    def update_status(self, status: dict[str, Any]) -> None:
        """Update the panel with new status data."""
        self._status = status
        self._refresh()

    def _update_line(self, line_id: str, label: str,
                     used: int, available: int, pct: float) -> None:
        """Render one compact context line: label used/avail bar pct."""
        pct_val = min(pct, 1.0)
        if pct_val > 0.8:
            color = "red"
        elif pct_val > 0.5:
            color = "yellow"
        else:
            color = "green"

        bar_width = 8
        filled = int(bar_width * pct_val)
        empty = bar_width - filled
        bar = "█" * filled + "░" * empty
        self.query_one(line_id, Static).update(
            f"[bold]{label}[/bold] {used}/{available} [{color}]{bar}[/{color}] {pct_val*100:.0f}%"
        )


class ChatInput(TextArea):
    """Custom TextArea for chat input with Enter to send and Shift+Enter for new line."""

    class Submitted(events.Message):
        """Message sent when the user presses Enter."""
        def __init__(self, value: str):
            super().__init__()
            self.value = value

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._msg_history: list[str] = []
        self._history_index: int = 0
        self._draft: str = ""

    async def _on_key(self, event: events.Key) -> None:
        """Intercept specific keys before TextArea processes them."""
        if event.key == "enter":
            event.prevent_default()
            event.stop()
            if self.text.strip():
                self._msg_history.append(self.text)
                self._history_index = len(self._msg_history)
                self._draft = ""
                self.post_message(self.Submitted(self.text))
                self.text = ""
            return
        if event.key == "up" and self.cursor_location[0] == 0 and self._msg_history:
            event.prevent_default()
            event.stop()
            if self._history_index > 0:
                if self._history_index == len(self._msg_history):
                    self._draft = self.text
                self._history_index -= 1
                self.text = self._msg_history[self._history_index]
                self.move_cursor(self.document.end)
            return
        if event.key == "down" and self.cursor_location[0] == self.document.line_count - 1 and self._msg_history:
            event.prevent_default()
            event.stop()
            if self._history_index < len(self._msg_history) - 1:
                self._history_index += 1
                self.text = self._msg_history[self._history_index]
                self.move_cursor(self.document.end)
            elif self._history_index == len(self._msg_history) - 1:
                self._history_index = len(self._msg_history)
                self.text = self._draft
                self.move_cursor(self.document.end)
            return
        if event.key == "tab":
            menu = self.app.query_one("#autocomplete-menu", OptionList)
            if menu.has_class("-visible"):
                event.prevent_default()
                event.stop()
                menu.focus()
                return
        if event.key == "escape":
            menu = self.app.query_one("#autocomplete-menu", OptionList)
            if menu.has_class("-visible"):
                event.prevent_default()
                event.stop()
                menu.remove_class("-visible")
                return
        # All other keys — fall through so that Textual's MRO dispatch
        # calls TextArea._on_key automatically.


# ── Modal screens ───────────────────────────────────────────────────────


class ModelPickerScreen(ModalScreen[str | None]):
    """Modal that lets the user pick an Ollama model with arrow keys."""

    BINDINGS = [Binding("escape", "cancel", "Cancel", show=True)]

    def __init__(self, models: list[dict], current_tag: str):
        super().__init__()
        self._models = models
        self._current_tag = current_tag

    def compose(self) -> ComposeResult:
        with Vertical(id="model-picker-box"):
            yield Label("Select a model", id="model-picker-title")
            yield OptionList(id="model-picker-list")
            yield Label("Enter = select · Esc = cancel", id="model-picker-hint")

    def on_mount(self) -> None:
        ol = self.query_one("#model-picker-list", OptionList)
        highlight_idx = 0
        for i, m in enumerate(self._models):
            name = m.get("name", "")
            size_gb = m.get("size", 0) / (1024 ** 3)
            marker = "  ← current" if name == self._current_tag else ""
            ol.add_option(Option(f"{name}  ({size_gb:.1f} GB){marker}", id=name))
            if name == self._current_tag:
                highlight_idx = i
        ol.highlighted = highlight_idx
        ol.focus()

    @on(OptionList.OptionSelected, "#model-picker-list")
    def on_select(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(event.option.id)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SettingsEditorScreen(ModalScreen[None]):
    """Modal to view and edit all Infinidev settings with two-panel layout."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+s", "save", "Save", show=True),
    ]

    # Section icons and ordering for the left panel
    SECTION_ORDER = [
        ("LLM", "Model"),
        ("Embedding", "Embeddings"),
        ("Loop Engine", "Loop Engine"),
        ("Phases", "Phases"),
        ("Permissions", "Permissions"),
        ("Sandbox", "Sandbox"),
        ("Timeouts", "Timeouts"),
        ("File Limits", "File Limits"),
        ("Database", "Database"),
        ("Web Tools", "Web Tools"),
        ("Knowledge", "Knowledge"),
        ("Workspace", "Workspace"),
        ("Code Interpreter", "Code Interpreter"),
    ]

    # Human-readable labels for settings
    SETTINGS_INFO = {
        "LLM_MODEL": {"section": "LLM", "label": "Model", "default": "ollama_chat/qwen2.5-coder:7b", "type": "string",
                       "desc": "LiteLLM model identifier. Format: provider/model (e.g. ollama_chat/qwen2.5-coder:7b)."},
        "LLM_BASE_URL": {"section": "LLM", "label": "Base URL", "default": "http://localhost:11434", "type": "string",
                         "desc": "Base URL for the LLM provider (Ollama, OpenAI-compatible, etc)."},
        "LLM_API_KEY": {"section": "LLM", "label": "API Key", "default": "ollama", "type": "string",
                        "desc": "API key for the LLM provider. Use 'ollama' for local Ollama."},
        "EMBEDDING_PROVIDER": {"section": "Embedding", "label": "Provider", "default": "ollama", "type": "string",
                               "desc": "Provider for text embeddings used in semantic search."},
        "EMBEDDING_MODEL": {"section": "Embedding", "label": "Model", "default": "nomic-embed-text", "type": "string",
                            "desc": "Embedding model name (must be available in the provider)."},
        "EMBEDDING_BASE_URL": {"section": "Embedding", "label": "Base URL", "default": "http://localhost:11434", "type": "string",
                               "desc": "Base URL for the embedding provider."},
        "LOOP_MAX_ITERATIONS": {"section": "Loop Engine", "label": "Max Iterations", "default": 50, "type": "int",
                                "desc": "Maximum plan-execute-summarize iterations per task."},
        "LOOP_MAX_TOOL_CALLS_PER_ACTION": {"section": "Loop Engine", "label": "Tool Calls / Step", "default": 0, "type": "int",
                                           "desc": "Max tool calls per step. 0 = unlimited (only global limit applies)."},
        "LOOP_MAX_TOTAL_TOOL_CALLS": {"section": "Loop Engine", "label": "Total Tool Calls", "default": 1000, "type": "int",
                                      "desc": "Hard limit on total tool calls per task across all steps."},
        "LOOP_HISTORY_WINDOW": {"section": "Loop Engine", "label": "History Window", "default": 0, "type": "int",
                                "desc": "Number of past action summaries to keep in context. 0 = keep all."},
        "ANALYSIS_ENABLED": {"section": "Phases", "label": "Analysis Phase", "default": True, "type": "bool",
                             "desc": "Enable the pre-development analyst phase that explores code and produces a spec."},
        "REVIEW_ENABLED": {"section": "Phases", "label": "Review Phase", "default": True, "type": "bool",
                           "desc": "Enable the post-development code review phase."},
        "EXECUTE_COMMANDS_PERMISSION": {"section": "Permissions", "label": "Shell Commands", "default": "auto_approve", "type": "select",
                                        "choices": ["auto_approve", "ask", "allowed_list"],
                                        "desc": "How to handle shell command execution. auto_approve=allow all, ask=prompt user, allowed_list=only commands in ALLOWED_COMMANDS_LIST."},
        "ALLOWED_COMMANDS_LIST": {"section": "Permissions", "label": "Allowed Commands", "default": [], "type": "list",
                                  "desc": "Commands allowed when shell permission is 'allowed_list'. Comma-separated."},
        "FILE_OPERATIONS_PERMISSION": {"section": "Permissions", "label": "File Operations", "default": "ask", "type": "select",
                                       "choices": ["ask", "auto_approve", "allowed_paths"],
                                       "desc": "How to handle file write/edit. ask=prompt user, auto_approve=allow all, allowed_paths=only within listed paths."},
        "ALLOWED_FILE_PATHS": {"section": "Permissions", "label": "Allowed File Paths", "default": [], "type": "list",
                               "desc": "Paths allowed when file permission is 'allowed_paths'. Comma-separated."},
        "SANDBOX_ENABLED": {"section": "Sandbox", "label": "Enabled", "default": False, "type": "bool",
                            "desc": "Enable sandbox mode. Restricts file access to allowed base directories."},
        "ALLOWED_BASE_DIRS": {"section": "Sandbox", "label": "Allowed Directories", "default": ["/"], "type": "list",
                              "desc": "Directories the agent can access when sandbox is enabled. Comma-separated."},
        "ALLOWED_COMMANDS": {"section": "Sandbox", "label": "Allowed Commands", "default": [], "type": "list",
                             "desc": "Legacy: shell commands allowed in sandbox mode. Comma-separated."},
        "COMMAND_TIMEOUT": {"section": "Timeouts", "label": "Shell Command", "default": 120, "type": "int",
                            "desc": "Max seconds for a shell command before it is killed."},
        "WEB_TIMEOUT": {"section": "Timeouts", "label": "Web Requests", "default": 30, "type": "int",
                        "desc": "Max seconds for web fetch/search requests."},
        "GIT_PUSH_TIMEOUT": {"section": "Timeouts", "label": "Git Push", "default": 120, "type": "int",
                             "desc": "Max seconds for git push operations."},
        "MAX_FILE_SIZE_BYTES": {"section": "File Limits", "label": "Max File Size (bytes)", "default": 5242880, "type": "int",
                                "desc": "Max file size (bytes) the agent can read. Default 5 MB."},
        "MAX_DIR_LISTING": {"section": "File Limits", "label": "Max Directory Entries", "default": 1000, "type": "int",
                            "desc": "Max entries returned by list_directory."},
        "DB_PATH": {"section": "Database", "label": "Database Path", "default": "~/.infinidev/infinidev.db", "type": "string",
                    "desc": "Path to the SQLite database for projects, tasks, and findings."},
        "MAX_RETRIES": {"section": "Database", "label": "Max Retries", "default": 5, "type": "int",
                        "desc": "Max retries for database operations on WAL contention."},
        "RETRY_BASE_DELAY": {"section": "Database", "label": "Retry Base Delay (s)", "default": 0.1, "type": "float",
                             "desc": "Base delay (seconds) for exponential backoff on DB retries."},
        "WEB_CACHE_TTL_SECONDS": {"section": "Web Tools", "label": "Cache TTL (s)", "default": 3600, "type": "int",
                                  "desc": "Cache duration (seconds) for web search/fetch results."},
        "WEB_RPM_LIMIT": {"section": "Web Tools", "label": "Rate Limit (rpm)", "default": 20, "type": "int",
                          "desc": "Max web requests per minute (rate limiting)."},
        "WEB_ROBOTS_CACHE_TTL": {"section": "Web Tools", "label": "Robots.txt Cache (s)", "default": 3600, "type": "int",
                                 "desc": "Cache duration (seconds) for robots.txt lookups."},
        "DEDUP_SIMILARITY_THRESHOLD": {"section": "Knowledge", "label": "Dedup Threshold", "default": 0.82, "type": "float",
                                       "desc": "Cosine similarity threshold for deduplicating findings (0-1)."},
        "WORKSPACE_BASE_DIR": {"section": "Workspace", "label": "Base Directory", "default": ".", "type": "string",
                               "desc": "Base directory for the agent's workspace."},
        "CODE_INTERPRETER_TIMEOUT": {"section": "Code Interpreter", "label": "Timeout (s)", "default": 120, "type": "int",
                                     "desc": "Max seconds for code interpreter execution."},
        "CODE_INTERPRETER_MAX_OUTPUT": {"section": "Code Interpreter", "label": "Max Output (chars)", "default": 50000, "type": "int",
                                        "desc": "Max characters of output captured from code interpreter."},
    }

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._edited_values: dict[str, str] = {}
        self._current_section: str = self.SECTION_ORDER[0][0]
        # Build section -> keys mapping preserving definition order
        self._sections: dict[str, list[str]] = {}
        for key, info in self.SETTINGS_INFO.items():
            sec = info["section"]
            if sec not in self._sections:
                self._sections[sec] = []
            self._sections[sec].append(key)

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-box"):
            yield Label("Settings", id="settings-title")
            with Horizontal(id="settings-body"):
                yield OptionList(id="settings-sections")
                with Vertical(id="settings-right"):
                    yield OptionList(id="settings-list")
                    yield Label("", id="settings-desc")
            with Horizontal(id="settings-footer"):
                yield Label("[dim]Ctrl+S save   Esc cancel[/dim]", id="settings-hint")
                with Horizontal(id="settings-buttons"):
                    yield Button("Save", variant="success", id="btn-save")
                    yield Button("Cancel", variant="default", id="btn-cancel")

    def on_mount(self) -> None:
        # Populate sections list
        sections_ol = self.query_one("#settings-sections", OptionList)
        for section_key, section_label in self.SECTION_ORDER:
            if section_key in self._sections:
                sections_ol.add_option(Option(section_label, id=section_key))
        # Highlight first section
        if sections_ol.option_count > 0:
            sections_ol.highlighted = 0
        self._populate_settings(self._current_section)
        sections_ol.focus()

    def _populate_settings(self, section: str) -> None:
        """Fill the right panel with settings for the given section."""
        self._current_section = section
        ol = self.query_one("#settings-list", OptionList)
        ol.clear_options()

        keys = self._sections.get(section, [])
        for key in keys:
            info = self.SETTINGS_INFO[key]
            label = info.get("label", key)
            # Show edited value if available, otherwise current
            if key in self._edited_values:
                val = self._edited_values[key]
                display_val = str(val) if not isinstance(val, list) else ", ".join(val)
                line = f"[bold]{label}[/bold]  [bold green]{display_val}[/bold green] [dim italic]*[/dim italic]"
            else:
                current_value = getattr(self._settings, key, info.get("default", ""))
                display = str(current_value) if current_value not in (None, "", []) else "[dim]not set[/dim]"
                line = f"[bold]{label}[/bold]  [dim]{display}[/dim]"
            ol.add_option(Option(line, id=key))

        # Update description for first item
        if keys:
            desc = self.SETTINGS_INFO[keys[0]].get("desc", "")
            self.query_one("#settings-desc", Label).update(f"[dim]{desc}[/dim]")
        else:
            self.query_one("#settings-desc", Label).update("")

    @on(OptionList.OptionHighlighted, "#settings-sections")
    def on_section_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option and event.option.id:
            self._populate_settings(event.option.id)

    @on(OptionList.OptionSelected, "#settings-sections")
    def on_section_select(self, event: OptionList.OptionSelected) -> None:
        """When a section is selected (Enter), move focus to the settings list."""
        self.query_one("#settings-list", OptionList).focus()

    @on(OptionList.OptionHighlighted, "#settings-list")
    def on_setting_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option and event.option.id:
            info = self.SETTINGS_INFO.get(event.option.id, {})
            desc = info.get("desc", "")
            key_name = event.option.id
            self.query_one("#settings-desc", Label).update(
                f"[dim italic]{key_name}[/dim italic]\n[dim]{desc}[/dim]"
            )

    @on(OptionList.OptionSelected, "#settings-list")
    def on_setting_select(self, event: OptionList.OptionSelected) -> None:
        if event.option and event.option.id:
            self._show_setting_editor(event.option.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def _show_setting_editor(self, setting_key: str) -> None:
        info = self.SETTINGS_INFO.get(setting_key, {})
        current_value = self._edited_values.get(setting_key, getattr(self._settings, setting_key, None))
        setting_type = info.get("type", "string")
        description = info.get("desc", "")
        choices = info.get("choices", None)

        editor = SettingValueEditor(
            setting_key, current_value, setting_type,
            description=description, choices=choices,
            label=info.get("label", setting_key),
        )
        self.app.push_screen(editor, lambda value: self._on_value_updated(setting_key, value))

    def _on_value_updated(self, key: str, value: str | None) -> None:
        if value is not None:
            setting_type = self.SETTINGS_INFO.get(key, {}).get("type", "string")
            if setting_type == "list" and isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            self._edited_values[key] = value
            # Refresh the current section to show the edit
            self._populate_settings(self._current_section)
        self.query_one("#settings-list", OptionList).focus()

    def action_save(self) -> None:
        """Save all edited settings."""
        if not self._edited_values:
            self.dismiss(None)
            return

        from infinidev.config.settings import reload_all
        try:
            self._settings.save_user_settings(self._edited_values)
            reload_all()
            self.notify(f"Saved {len(self._edited_values)} setting(s).", timeout=3)
            self.dismiss(None)
        except Exception as e:
            self.notify(f"Failed to save: {e}", severity="error", timeout=5)
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Cancel and discard changes."""
        self.dismiss(None)


class PermissionDetailScreen(ModalScreen[None]):
    """Modal to view full permission details with syntax highlighting."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    def __init__(self, content: str):
        super().__init__()
        self._content = content

    def compose(self) -> ComposeResult:
        with Vertical(id="perm-detail-box"):
            yield Label("Permission Detail — Full Content", id="perm-detail-title")
            yield TextArea(self._content, read_only=True, language="python", id="perm-detail-code")
            yield Label("[dim]Esc to close[/dim]", id="perm-detail-hint")

    def on_mount(self) -> None:
        self.query_one("#perm-detail-code", TextArea).focus()

    def action_close(self) -> None:
        self.dismiss(None)


class SettingValueEditor(ModalScreen[str | None]):
    """Modal to edit a single setting value.

    Renders a Select widget for bool/select types, TextArea for everything else.
    Shows a description of the setting above the input.
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, key: str, current_value, value_type: str,
                 description: str = "", choices: list[str] | None = None,
                 label: str = ""):
        super().__init__()
        self._key = key
        self._label = label or key
        self._current_value = current_value
        self._value_type = value_type
        self._description = description
        self._choices = choices
        self._new_value = str(current_value)

    def compose(self) -> ComposeResult:
        from textual.widgets import Select
        with Vertical(id="setting-editor-box"):
            yield Label(self._label, id="setting-editor-title")
            if self._description:
                yield Label(f"[dim]{self._description}[/dim]", id="setting-editor-desc")

            if self._value_type == "bool":
                options = [("true", "true"), ("false", "false")]
                current = "true" if str(self._current_value).lower() in ("true", "1", "yes") else "false"
                yield Select(options, value=current, id="setting-editor-select")
            elif self._value_type == "select" and self._choices:
                options = [(c, c) for c in self._choices]
                current = str(self._current_value) if str(self._current_value) in self._choices else self._choices[0]
                yield Select(options, value=current, id="setting-editor-select")
            else:
                yield TextArea(self._new_value, id="setting-editor-input")

            with Vertical(id="setting-editor-footer"):
                with Horizontal():
                    yield Button("Save", variant="success", id="btn-editor-save")
                    yield Button("Cancel", variant="error", id="btn-editor-cancel")

    def on_mount(self) -> None:
        try:
            ta = self.query_one("#setting-editor-input", TextArea)
            ta.focus()
            ta.select_all()
        except Exception:
            # Select widget — just focus it
            try:
                from textual.widgets import Select
                self.query_one("#setting-editor-select", Select).focus()
            except Exception:
                pass

    @on(TextArea.Changed, "#setting-editor-input")
    def on_change(self, event: TextArea.Changed) -> None:
        self._new_value = event.control.text

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-editor-save":
            self.action_save()
        elif event.button.id == "btn-editor-cancel":
            self.action_cancel()

    def action_save(self) -> None:
        from textual.widgets import Select
        # Read from Select if present
        try:
            sel = self.query_one("#setting-editor-select", Select)
            value = str(sel.value) if sel.value is not None else str(self._current_value)
            self.dismiss(value)
            return
        except Exception:
            pass
        self.dismiss(self._new_value if self._new_value else str(self._current_value))

    def action_cancel(self) -> None:
        self.dismiss(None)


class FindingsBrowserScreen(ModalScreen[None]):
    """Modal to browse and read findings / knowledge base."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    def __init__(self, findings: list[dict], title: str = "Findings"):
        super().__init__()
        self._findings = findings
        self._title = title

    def compose(self) -> ComposeResult:
        with Vertical(id="findings-box"):
            yield Label(self._title, id="findings-title")
            with Horizontal(id="findings-body"):
                yield OptionList(id="findings-list")
                yield VerticalScroll(Static("Select a finding to view details.", id="findings-detail-text"), id="findings-detail")
            yield Label("↑↓ = navigate · Enter = view · Esc = close", id="findings-hint")

    def on_mount(self) -> None:
        ol = self.query_one("#findings-list", OptionList)
        if not self._findings:
            ol.add_option(Option("(no findings yet)", id="__empty__"))
        else:
            for f in self._findings:
                type_icon = {"project_context": "📌", "observation": "👁", "conclusion": "✓",
                             "hypothesis": "?", "experiment": "⚗", "proof": "✔"}.get(f["finding_type"], "·")
                status_mark = "" if f["status"] in ("active", "provisional") else f" [{f['status']}]"
                ol.add_option(Option(f"{type_icon} {f['topic'][:50]}{status_mark}", id=str(f["id"])))
        ol.focus()

    @on(OptionList.OptionHighlighted, "#findings-list")
    def on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id == "__empty__":
            return
        finding = next((f for f in self._findings if str(f["id"]) == event.option.id), None)
        if not finding:
            return
        detail = self.query_one("#findings-detail-text", Static)
        lines = [
            f"[bold]{finding['topic']}[/bold]",
            f"Type: {finding['finding_type']}  |  Confidence: {finding['confidence']:.0%}  |  Status: {finding['status']}",
            f"Created: {finding['created_at']}",
            "─" * 40,
            finding["content"] or "(no content)",
        ]
        detail.update("\n".join(lines))

    def action_close(self) -> None:
        self.dismiss(None)


class DocsBrowserScreen(ModalScreen[None]):
    """Modal to browse locally cached library documentation."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    def __init__(self, libraries: list[dict], sections: dict[str, list[dict]]):
        super().__init__()
        self._libraries = libraries  # [{"library_name", "language", "version", "section_count"}]
        self._sections = sections    # {key: [{"section_title", "content", "section_order"}]}
        self._current_lib_key: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical(id="docs-box"):
            yield Label("Library Documentation", id="docs-title")
            with Horizontal(id="docs-body"):
                yield OptionList(id="docs-lib-list")
                yield OptionList(id="docs-section-list")
                yield VerticalScroll(Static("Select a library to browse its documentation.", id="docs-content-text"), id="docs-content")
            yield Label("↑↓ = navigate · Tab = switch panel · Esc = close", id="docs-hint")

    def on_mount(self) -> None:
        ol = self.query_one("#docs-lib-list", OptionList)
        if not self._libraries:
            ol.add_option(Option("(no documentation yet)", id="__empty__"))
        else:
            for lib in self._libraries:
                label = f"{lib['library_name']} ({lib['language']}) v{lib['version']}  [{lib['section_count']} sections]"
                ol.add_option(Option(label, id=lib["key"]))
        ol.focus()

    @on(OptionList.OptionHighlighted, "#docs-lib-list")
    def on_lib_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option.id == "__empty__":
            return
        key = event.option.id
        if key == self._current_lib_key:
            return
        self._current_lib_key = key
        # Populate sections list
        sl = self.query_one("#docs-section-list", OptionList)
        sl.clear_options()
        secs = self._sections.get(key, [])
        for sec in secs:
            sl.add_option(Option(sec["section_title"], id=f"{key}::{sec['section_title']}"))
        # Show first section content
        detail = self.query_one("#docs-content-text", Static)
        if secs:
            detail.update(f"[bold]{secs[0]['section_title']}[/bold]\n{'─' * 40}\n{secs[0]['content']}")
        else:
            detail.update("(no sections)")

    @on(OptionList.OptionHighlighted, "#docs-section-list")
    def on_section_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if not event.option.id or not self._current_lib_key:
            return
        section_title = event.option.id.split("::", 1)[1] if "::" in event.option.id else event.option.id
        secs = self._sections.get(self._current_lib_key, [])
        sec = next((s for s in secs if s["section_title"] == section_title), None)
        if not sec:
            return
        detail = self.query_one("#docs-content-text", Static)
        detail.update(f"[bold]{sec['section_title']}[/bold]\n{'─' * 40}\n{sec['content']}")

    def action_close(self) -> None:
        self.dismiss(None)


# ── Unsaved Changes Modal ────────────────────────────────────────────────


class UnsavedChangesScreen(ModalScreen[str]):
    """Modal asking to save/discard/cancel when closing a modified file."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, filename: str):
        super().__init__()
        self._filename = filename

    def compose(self) -> ComposeResult:
        with Vertical(id="unsaved-box"):
            yield Label(f"Unsaved changes in [bold]{self._filename}[/bold]", id="unsaved-title")
            yield Label("Do you want to save before closing?", id="unsaved-desc")
            with Horizontal(id="unsaved-buttons"):
                yield Button("Save", id="btn-unsaved-save", variant="success")
                yield Button("Discard", id="btn-unsaved-discard", variant="error")
                yield Button("Cancel", id="btn-unsaved-cancel", variant="default")

    def on_mount(self) -> None:
        self.query_one("#btn-unsaved-save").focus()

    @on(Button.Pressed, "#btn-unsaved-save")
    def _save(self, event: Button.Pressed) -> None:
        self.dismiss("save")

    @on(Button.Pressed, "#btn-unsaved-discard")
    def _discard(self, event: Button.Pressed) -> None:
        self.dismiss("discard")

    @on(Button.Pressed, "#btn-unsaved-cancel")
    def _cancel_btn(self, event: Button.Pressed) -> None:
        self.dismiss("cancel")

    def action_cancel(self) -> None:
        self.dismiss("cancel")


# ── Search Bar Widget ────────────────────────────────────────────────────


class SearchBar(Horizontal):
    """Inline search bar for finding text in the active file editor."""

    class Dismissed(events.Message):
        """Fired when the search bar is closed."""

    def __init__(self, editor: TextArea, **kwargs):
        super().__init__(id="search-bar", **kwargs)
        self._editor = editor
        self._matches: list[tuple[int, int, int]] = []  # (row, col_start, col_end)
        self._current_idx = 0

    def compose(self) -> ComposeResult:
        yield Input(placeholder="Search...", id="search-input")
        yield Label("0/0", id="search-match-count")
        yield Button("Prev", id="btn-search-prev", variant="default")
        yield Button("Next", id="btn-search-next", variant="default")
        yield Button("X", id="btn-search-close", variant="default")

    def on_mount(self) -> None:
        self.query_one("#search-input", Input).focus()

    @on(Input.Changed, "#search-input")
    def _on_search_changed(self, event: Input.Changed) -> None:
        self._do_search(event.value)

    def _do_search(self, query: str) -> None:
        self._matches.clear()
        self._current_idx = 0
        if not query:
            self.query_one("#search-match-count", Label).update("0/0")
            return
        text = self._editor.text
        lines = text.split("\n")
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            return
        for row_idx, line in enumerate(lines):
            for m in pattern.finditer(line):
                self._matches.append((row_idx, m.start(), m.end()))
        total = len(self._matches)
        if total > 0:
            self._current_idx = 0
            self._goto_match()
        self.query_one("#search-match-count", Label).update(
            f"{self._current_idx + 1 if total else 0}/{total}"
        )

    def _goto_match(self) -> None:
        if not self._matches:
            return
        row, col_start, col_end = self._matches[self._current_idx]
        self._editor.select_line(row)
        from textual.widgets.text_area import Selection, Location
        self._editor.selection = Selection(
            start=(row, col_start), end=(row, col_end)
        )
        self._editor.scroll_cursor_visible()

    @on(Button.Pressed, "#btn-search-next")
    def _next_match(self, event: Button.Pressed) -> None:
        if self._matches:
            self._current_idx = (self._current_idx + 1) % len(self._matches)
            self._goto_match()
            self._update_counter()

    @on(Button.Pressed, "#btn-search-prev")
    def _prev_match(self, event: Button.Pressed) -> None:
        if self._matches:
            self._current_idx = (self._current_idx - 1) % len(self._matches)
            self._goto_match()
            self._update_counter()

    def _update_counter(self) -> None:
        total = len(self._matches)
        self.query_one("#search-match-count", Label).update(
            f"{self._current_idx + 1 if total else 0}/{total}"
        )

    @on(Button.Pressed, "#btn-search-close")
    def _close(self, event: Button.Pressed) -> None:
        self.post_message(self.Dismissed())
        self.remove()

    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.prevent_default()
            event.stop()
            self.post_message(self.Dismissed())
            self.remove()
        elif event.key == "enter":
            event.prevent_default()
            event.stop()
            if self._matches:
                self._current_idx = (self._current_idx + 1) % len(self._matches)
                self._goto_match()
                self._update_counter()


# ── Project Search Modal ─────────────────────────────────────────────────

# Directories and patterns to skip during project search
_IGNORED_DIRS = {
    "node_modules", "__pycache__", ".git", ".hg", ".svn",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "dist", "build", ".eggs",
    ".venv", "venv", "env",
    ".next", ".nuxt", ".cache", "coverage",
    ".DS_Store", "Thumbs.db",
    ".idea", ".vscode",
}

_IGNORED_EXTENSIONS = {
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe",
    ".min.js", ".min.css", ".map",
    ".lock", ".egg", ".whl",
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".pdf", ".doc", ".docx",
}


class ProjectSearchScreen(ModalScreen[None]):
    """Modal for searching text across all project files."""

    BINDINGS = [Binding("escape", "close", "Close", show=True)]

    def compose(self) -> ComposeResult:
        with Vertical(id="project-search-box"):
            yield Label("Search in Project", id="project-search-title")
            with Horizontal(id="project-search-input-row"):
                yield Input(placeholder="Search query...", id="project-search-input")
                yield Checkbox("Skip junk", value=True, id="project-search-skip-junk")
            yield Label("", id="project-search-status")
            with Horizontal(id="project-search-body"):
                yield OptionList(id="project-search-results")
                yield TextArea("", read_only=True, id="project-search-preview",
                               theme="monokai", show_line_numbers=True)
            yield Label("Enter = Open file  |  Esc = Close", id="project-search-hint")

    def on_mount(self) -> None:
        self.query_one("#project-search-input", Input).focus()
        self._result_entries: list[dict] = []  # {path, line_no, text, context}
        self._search_id = 0

    @on(Input.Changed, "#project-search-input")
    def _on_query_change(self, event: Input.Changed) -> None:
        query = event.value.strip()
        if len(query) < 2:
            self.query_one("#project-search-results", OptionList).clear_options()
            self.query_one("#project-search-status", Label).update("")
            self._result_entries.clear()
            return
        self._search_id += 1
        skip_junk = True
        try:
            skip_junk = self.query_one("#project-search-skip-junk", Checkbox).value
        except Exception:
            pass
        self._run_search(query, self._search_id, skip_junk)

    @work(thread=True)
    def _run_search(self, query: str, search_id: int, skip_junk: bool = True) -> None:
        results: list[dict] = []
        try:
            pattern = re.compile(re.escape(query), re.IGNORECASE)
        except re.error:
            return
        root = pathlib.Path(os.getcwd())
        file_count = 0
        match_count = 0
        for fpath in root.rglob("*"):
            if self._search_id != search_id:
                return  # cancelled
            if not fpath.is_file():
                continue
            # Check skip patterns
            if skip_junk:
                parts = fpath.relative_to(root).parts
                if any(p in _IGNORED_DIRS for p in parts):
                    continue
                if fpath.suffix.lower() in _IGNORED_EXTENSIONS:
                    continue
                if any(p.endswith(".egg-info") for p in parts):
                    continue
            # Skip large files
            try:
                if fpath.stat().st_size > 1_000_000:
                    continue
            except OSError:
                continue
            # Read and search
            try:
                text = fpath.read_text(errors="replace")
            except Exception:
                continue
            lines = text.split("\n")
            for line_no, line in enumerate(lines, 1):
                if pattern.search(line):
                    # Gather context (2 lines before/after)
                    ctx_start = max(0, line_no - 3)
                    ctx_end = min(len(lines), line_no + 2)
                    context = "\n".join(lines[ctx_start:ctx_end])
                    rel = str(fpath.relative_to(root))
                    results.append({
                        "path": str(fpath),
                        "rel": rel,
                        "line_no": line_no,
                        "text": line.strip()[:120],
                        "context": context,
                    })
                    match_count += 1
                    if match_count >= 500:
                        break
            file_count += 1
            if match_count >= 500:
                break
        if self._search_id != search_id:
            return
        self.app.call_from_thread(self._show_results, results, file_count, match_count, query)

    def _show_results(self, results: list[dict], file_count: int, match_count: int, query: str) -> None:
        self._result_entries = results
        ol = self.query_one("#project-search-results", OptionList)
        ol.clear_options()
        highlight_re = re.compile(re.escape(query), re.IGNORECASE)
        for r in results:
            prefix = f"{r['rel']}:{r['line_no']}  "
            line_text = r['text']
            full = prefix + line_text
            if len(full) > 100:
                full = full[:97] + "..."
                line_text = full[len(prefix):]
            label = Text(full)
            # Highlight the file:line prefix in dim
            label.stylize("dim", 0, len(prefix))
            # Highlight all query matches in bold yellow
            for m in highlight_re.finditer(full):
                label.stylize("bold yellow", m.start(), m.end())
            ol.add_option(Option(label))
        status = f"{match_count} matches in {file_count} files"
        if match_count >= 500:
            status += " (limited)"
        self.query_one("#project-search-status", Label).update(status)

    @on(OptionList.OptionHighlighted, "#project-search-results")
    def _on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if event.option_index is not None and event.option_index < len(self._result_entries):
            entry = self._result_entries[event.option_index]
            preview = self.query_one("#project-search-preview", TextArea)
            preview.text = entry.get("context", "")

    @on(OptionList.OptionSelected, "#project-search-results")
    def _on_select(self, event: OptionList.OptionSelected) -> None:
        if event.option_index is not None and event.option_index < len(self._result_entries):
            entry = self._result_entries[event.option_index]
            self.dismiss(None)
            self.app.open_file_at_line(entry["path"], entry["line_no"])

    def action_close(self) -> None:
        self.dismiss(None)


# ── Custom DirectoryTree with dirty indicators ──────────────────────────


# File extension → icon mapping for the explorer
_FILE_ICONS: dict[str, str] = {
    # Python
    ".py": "\U0001f40d ",     # snake
    ".pyw": "\U0001f40d ",
    ".pyx": "\U0001f40d ",
    ".pyi": "\U0001f40d ",
    # JavaScript / TypeScript
    ".js": "\u26a1 ",         # lightning
    ".mjs": "\u26a1 ",
    ".cjs": "\u26a1 ",
    ".ts": "\U0001f4d8 ",     # blue book
    ".tsx": "\U0001f4d8 ",
    ".jsx": "\u26a1 ",
    # Web
    ".html": "\U0001f310 ",   # globe
    ".htm": "\U0001f310 ",
    ".css": "\U0001f3a8 ",    # palette
    ".scss": "\U0001f3a8 ",
    ".sass": "\U0001f3a8 ",
    ".less": "\U0001f3a8 ",
    ".svg": "\U0001f3a8 ",
    # Data / Config
    ".json": "\U0001f4cb ",   # clipboard
    ".toml": "\u2699\ufe0f ",  # gear
    ".yaml": "\u2699\ufe0f ",
    ".yml": "\u2699\ufe0f ",
    ".xml": "\U0001f4c3 ",    # page with curl
    ".ini": "\u2699\ufe0f ",
    ".cfg": "\u2699\ufe0f ",
    ".conf": "\u2699\ufe0f ",
    ".env": "\U0001f512 ",    # lock
    ".properties": "\u2699\ufe0f ",
    # Documentation
    ".md": "\U0001f4d6 ",     # open book
    ".mdx": "\U0001f4d6 ",
    ".rst": "\U0001f4d6 ",
    ".txt": "\U0001f4c4 ",    # page
    ".log": "\U0001f4dc ",    # scroll
    ".csv": "\U0001f4ca ",    # bar chart
    # Shell / Scripts
    ".sh": "\U0001f4bb ",     # laptop
    ".bash": "\U0001f4bb ",
    ".zsh": "\U0001f4bb ",
    ".fish": "\U0001f4bb ",
    ".bat": "\U0001f4bb ",
    ".cmd": "\U0001f4bb ",
    ".ps1": "\U0001f4bb ",
    # Compiled / Systems
    ".rs": "\U0001f980 ",     # crab (Rust)
    ".go": "\U0001f439 ",     # hamster (Go gopher-ish)
    ".java": "\u2615 ",       # coffee
    ".kt": "\U0001f4a0 ",     # diamond shape (Kotlin)
    ".scala": "\U0001f534 ",  # red circle
    ".c": "\U0001f527 ",      # wrench
    ".h": "\U0001f527 ",
    ".cpp": "\U0001f527 ",
    ".hpp": "\U0001f527 ",
    ".cs": "\U0001f7e3 ",     # purple circle (C#)
    ".swift": "\U0001f3af ",  # dart (Swift)
    ".rb": "\U0001f48e ",     # gem (Ruby)
    ".php": "\U0001f418 ",    # elephant (PHP)
    ".lua": "\U0001f319 ",    # crescent moon
    ".r": "\U0001f4c8 ",      # chart up
    ".R": "\U0001f4c8 ",
    ".jl": "\U0001f7e2 ",     # green circle (Julia)
    ".ex": "\U0001f7e3 ",     # purple (Elixir)
    ".exs": "\U0001f7e3 ",
    ".erl": "\U0001f7e0 ",    # orange (Erlang)
    ".hs": "\U0001f7e3 ",     # Haskell
    ".ml": "\U0001f7e0 ",     # OCaml
    # Database
    ".sql": "\U0001f5c3 ",    # card file box
    ".db": "\U0001f5c3 ",
    ".sqlite": "\U0001f5c3 ",
    # Docker / Infra
    "Dockerfile": "\U0001f433 ",   # whale
    ".dockerfile": "\U0001f433 ",
    "docker-compose.yml": "\U0001f433 ",
    "docker-compose.yaml": "\U0001f433 ",
    # Images
    ".png": "\U0001f5bc\ufe0f ",   # frame
    ".jpg": "\U0001f5bc\ufe0f ",
    ".jpeg": "\U0001f5bc\ufe0f ",
    ".gif": "\U0001f5bc\ufe0f ",
    ".ico": "\U0001f5bc\ufe0f ",
    ".webp": "\U0001f5bc\ufe0f ",
    ".bmp": "\U0001f5bc\ufe0f ",
    # Audio / Video
    ".mp3": "\U0001f3b5 ",    # musical note
    ".wav": "\U0001f3b5 ",
    ".ogg": "\U0001f3b5 ",
    ".flac": "\U0001f3b5 ",
    ".mp4": "\U0001f3ac ",    # clapper board
    ".avi": "\U0001f3ac ",
    ".mkv": "\U0001f3ac ",
    ".mov": "\U0001f3ac ",
    ".webm": "\U0001f3ac ",
    # Archives
    ".zip": "\U0001f4e6 ",    # package
    ".tar": "\U0001f4e6 ",
    ".gz": "\U0001f4e6 ",
    ".bz2": "\U0001f4e6 ",
    ".xz": "\U0001f4e6 ",
    ".7z": "\U0001f4e6 ",
    ".rar": "\U0001f4e6 ",
    # Documents
    ".pdf": "\U0001f4d5 ",    # closed book
    ".doc": "\U0001f4d5 ",
    ".docx": "\U0001f4d5 ",
    ".xls": "\U0001f4ca ",    # bar chart
    ".xlsx": "\U0001f4ca ",
    ".ppt": "\U0001f4ca ",
    ".pptx": "\U0001f4ca ",
    # Fonts
    ".woff": "\U0001f520 ",   # ABCD
    ".woff2": "\U0001f520 ",
    ".ttf": "\U0001f520 ",
    ".otf": "\U0001f520 ",
    ".eot": "\U0001f520 ",
    # Security / Keys
    ".pem": "\U0001f511 ",    # key
    ".key": "\U0001f511 ",
    ".crt": "\U0001f511 ",
    ".cer": "\U0001f511 ",
    ".p12": "\U0001f511 ",
    # Lock files
    ".lock": "\U0001f512 ",   # lock
}

# Special filenames that get their own icon regardless of extension
_SPECIAL_FILE_ICONS: dict[str, str] = {
    "Dockerfile": "\U0001f433 ",         # whale (Docker)
    "docker-compose.yml": "\U0001f433 ",
    "docker-compose.yaml": "\U0001f433 ",
    "Makefile": "\U0001f3d7\ufe0f ",     # construction
    "CMakeLists.txt": "\U0001f3d7\ufe0f ",
    "Rakefile": "\U0001f48e ",           # gem (Ruby)
    "Gemfile": "\U0001f48e ",
    "Gemfile.lock": "\U0001f48e ",
    "Cargo.toml": "\U0001f980 ",         # crab (Rust)
    "Cargo.lock": "\U0001f980 ",
    "go.mod": "\U0001f439 ",             # hamster (Go)
    "go.sum": "\U0001f439 ",
    "package.json": "\U0001f4e6 ",       # package (Node)
    "package-lock.json": "\U0001f4e6 ",
    "yarn.lock": "\U0001f4e6 ",
    "pnpm-lock.yaml": "\U0001f4e6 ",
    "tsconfig.json": "\U0001f4d8 ",      # blue book (TS)
    "requirements.txt": "\U0001f40d ",   # snake (Python)
    "setup.py": "\U0001f40d ",
    "setup.cfg": "\U0001f40d ",
    "pyproject.toml": "\U0001f40d ",
    "Pipfile": "\U0001f40d ",
    "Pipfile.lock": "\U0001f40d ",
    "LICENSE": "\U0001f4dc ",            # scroll
    "LICENSE.md": "\U0001f4dc ",
    "LICENSE.txt": "\U0001f4dc ",
    ".gitignore": "\U0001f500 ",         # git
    ".gitmodules": "\U0001f500 ",
    ".gitattributes": "\U0001f500 ",
    ".dockerignore": "\U0001f433 ",
    ".eslintrc.js": "\U0001f9f9 ",       # broom (linter)
    ".eslintrc.json": "\U0001f9f9 ",
    ".prettierrc": "\U0001f9f9 ",
    ".editorconfig": "\u2699\ufe0f ",    # gear
    "CLAUDE.md": "\U0001f916 ",          # robot
}


class InfinidevDirectoryTree(DirectoryTree):
    """DirectoryTree with file-type icons and unsaved-modification highlighting."""

    ICON_NODE = "\U0001f4c1 "       # closed folder
    ICON_NODE_EXPANDED = "\U0001f4c2 "  # open folder
    ICON_FILE = "\U0001f4c4 "       # page

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dirty_paths: set[str] = set()

    def set_dirty_paths(self, paths: set[str]) -> None:
        """Update which file paths should show as modified."""
        if paths != self._dirty_paths:
            self._dirty_paths = paths.copy()
            self.refresh()

    def render_label(self, node, base_style, style):
        node_label = node._label.copy()
        node_label.stylize(style)

        if not self.is_mounted:
            return node_label

        is_dirty = False
        node_path = ""
        if node.data and hasattr(node.data, 'path'):
            node_path = str(node.data.path)
            is_dirty = node_path in self._dirty_paths

        if node._allow_expand:
            # Directory — always use standard folder icons
            icon = self.ICON_NODE_EXPANDED if node.is_expanded else self.ICON_NODE
            from rich.style import Style as RichStyle
            prefix = (icon, base_style + RichStyle(bold=True))
            node_label.stylize_before(
                self.get_component_rich_style("directory-tree--folder", partial=True)
            )
        else:
            # File — pick icon by special filename first, then extension
            fname = node_label.plain.strip()
            ext = pathlib.Path(fname).suffix.lower()
            icon = _SPECIAL_FILE_ICONS.get(fname,
                _FILE_ICONS.get(ext, self.ICON_FILE)
            )
            prefix = (icon, base_style)
            node_label.stylize_before(
                self.get_component_rich_style("directory-tree--file", partial=True),
            )
            node_label.highlight_regex(
                r"\..+$",
                self.get_component_rich_style(
                    "directory-tree--extension", partial=True
                ),
            )

        if node_label.plain.startswith("."):
            node_label.stylize_before(
                self.get_component_rich_style("directory-tree--hidden", partial=True)
            )

        # Dirty indicator
        if is_dirty:
            dirty_marker = Text("\u25cf ", style="bold yellow")
            node_label.stylize("yellow")
            text = Text.assemble(prefix, dirty_marker, node_label)
        else:
            text = Text.assemble(prefix, node_label)

        return text


# ── Main TUI ────────────────────────────────────────────────────────────


_CSS_PATH = pathlib.Path(__file__).parent / "tui.tcss"


class InfinidevTUI(App):
    """The main Infinidev TUI application."""

    CSS_PATH = _CSS_PATH

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", show=True),
        Binding("ctrl+e", "toggle_explorer", "Explorer", show=True, priority=True),
        Binding("ctrl+w", "close_tab", "Close tab", show=True, priority=True),
        Binding("ctrl+s", "save_file", "Save", show=True, priority=True),
        Binding("ctrl+f", "find_in_file", "Find", show=True, priority=True),
        Binding("ctrl+shift+f", "find_in_project", "Search project", show=True, priority=True),
        Binding("f2", "focus_chat", "Chat", show=True),
        Binding("f3", "focus_explorer", "Files", show=True),
        Binding("f4", "focus_sidebar", "Sidebar", show=True),
    ]

    # ── Compose ──────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main-container"):
            # File explorer (hidden by default)
            with Vertical(id="explorer"):
                with Horizontal(id="explorer-header"):
                    yield Label("EXPLORER", id="explorer-title")
                    yield Button("⟳", id="btn-sync-tree", variant="default")
                yield InfinidevDirectoryTree(os.getcwd(), id="file-tree")

            # Content area with tabs
            with Vertical(id="content-area"):
                with TabbedContent(id="content-tabs"):
                    with TabPane("Chat", id="chat-pane"):
                        yield VerticalScroll(id="chat-history")
                        yield OptionList(id="autocomplete-menu")
                        yield ChatInput(id="chat-input")

            # Sidebar
            with VerticalScroll(id="sidebar"):
                # Context window panel
                yield ContextPanel(id="context-panel")
                yield SidebarPanel("PLANNING", id="plan-panel")
                yield SidebarPanel("STEPS", id="steps-panel")
                yield SidebarPanel("ACTIONS", id="actions-panel")
                yield SidebarPanel("LOGS", id="logs-panel")
                yield Static("", id="sidebar-spacer")
        yield Static("", id="status-bar")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────

    def on_mount(self) -> None:
        init_db()
        event_bus.subscribe(self.on_loop_event)
        self.query_one("#chat-input").focus()
        self.add_message("System", "Welcome to Infinidev! Type your instruction or /help.", "system")

        self.session_id = str(uuid.uuid4())
        self._log_lines: list[str] = []
        self._open_files: dict[str, str] = {}  # tab_id → file_path
        self._original_content: dict[str, str] = {}  # tab_id → original content
        self._dirty_files: set[str] = set()  # set of dirty tab_ids
        self._tab_names: dict[str, str] = {}  # tab_id → display name
        self._file_diff_widgets: dict[str, FileChangeDiffWidget] = {}  # path → widget
        self._queued_messages: list = []  # queued message widgets
        self._tree_resolved_lines: list[str] = []  # accumulated tree resolution lines

        self.engine = LoopEngine()
        self.analyst = AnalysisEngine()
        self.reviewer = ReviewEngine()
        self._analysis_waiting = False  # True when waiting for user answer to analysis questions
        self._analysis_original_input = ""  # Original user input being analyzed
        self._analysis_event = None  # threading.Event for blocking run_engine
        self._analysis_answer = ""  # User's answer to analysis questions

        # Engine queue: process one message at a time
        self._engine_running = False
        self._pending_inputs: list[str] = []
        self._gather_next_task = False  # /think enables gather for next task only

        # Permission request state (for execute_command "ask" mode)
        self._permission_waiting = False
        self._permission_event = None  # threading.Event
        self._permission_approved = False

        # Register the permission handler so tools can ask user
        from infinidev.tools.permission import set_permission_handler
        set_permission_handler(self._handle_permission_request)

        self.agent = InfinidevAgent(agent_id="tui_agent")

        # Initialize context window calculator
        from infinidev.ui.context_calculator import calculator
        self.context_calculator = calculator
        self._init_context_display()
        
        self._update_status_bar()

        from infinidev.config.tech_detection import detect_tech_hints
        self.agent._tech_hints = detect_tech_hints(os.getcwd())

        # Initialize file watcher and background indexer
        self._file_watcher: Optional[FileWatcher] = None
        self._index_queue: Optional[IndexQueue] = None
        self._watcher_started = False
        self._expand_handlers: list[Callable] = []
        self._collapse_handlers: list[Callable] = []
        self._visible_paths: set[pathlib.Path] = set()
        self._start_file_watcher()

    def _init_context_display(self) -> None:
        """Initialize context window display widget and fetch model info."""
        self._context_panel = self.query_one("#context-panel", ContextPanel)
        self._fetch_model_context()

    @work(thread=True)
    def _fetch_model_context(self) -> None:
        """Fetch model context size from Ollama in background."""
        import asyncio
        try:
            asyncio.run(self.context_calculator.update_model_context())
        except Exception:
            pass
        # Push initial status to the panel
        status = self.context_calculator.get_context_status()
        self.call_from_thread(self._context_panel.update_status, status)

    def _refresh_context_panel(self, task_tokens: int = 0,
                               prompt_tokens: int = 0,
                               completion_tokens: int = 0) -> None:
        """Refresh the context panel with task prompt tokens from the loop engine."""
        self.context_calculator.update_task(task_prompt_tokens=prompt_tokens)
        self._context_panel.update_status(self.context_calculator.get_context_status())

    # ── Quit with unsaved check ─────────────────────────

    def action_quit(self) -> None:
        """Override quit to warn about unsaved files."""
        event_bus.unsubscribe(self.on_loop_event)
        if self._dirty_files:
            dirty_names = []
            for tid in self._dirty_files:
                fp = self._open_files.get(tid, "")
                dirty_names.append(pathlib.Path(fp).name if fp else "unknown")
            names_str = ", ".join(dirty_names)
            self.push_screen(
                UnsavedChangesScreen(names_str),
                callback=self._handle_quit_unsaved,
            )
        else:
            self.exit()

    def _handle_quit_unsaved(self, result: str) -> None:
        """Handle the unsaved changes modal when quitting."""
        if result == "save":
            # Save all dirty files, then exit
            for tid in list(self._dirty_files):
                fp = self._open_files.get(tid, "")
                if fp:
                    try:
                        editor = self.query_one(f"#editor-{tid}", TextArea)
                        pathlib.Path(fp).write_text(editor.text)
                    except Exception:
                        pass
            self.exit()
        elif result == "discard":
            self.exit()
        # "cancel" → stay

    # ── Focus actions ────────────────────────────────────

    def action_toggle_explorer(self) -> None:
        explorer = self.query_one("#explorer")
        if explorer.has_class("-visible"):
            explorer.remove_class("-visible")
            self.query_one("#chat-input", ChatInput).focus()
        else:
            explorer.add_class("-visible")
            self.query_one("#file-tree", DirectoryTree).focus()

    def action_focus_chat(self) -> None:
        tabs = self.query_one("#content-tabs", TabbedContent)
        tabs.active = "chat-pane"
        self.query_one("#chat-input", ChatInput).focus()

    def action_focus_explorer(self) -> None:
        explorer = self.query_one("#explorer")
        if not explorer.has_class("-visible"):
            explorer.add_class("-visible")
        self.query_one("#file-tree", DirectoryTree).focus()

    def action_focus_sidebar(self) -> None:
        # Focus the first panel's content
        self.query_one("#plan-panel").focus()

    async def action_close_tab(self) -> None:
        """Close the active file editor tab (not the Chat tab)."""
        tabs = self.query_one("#content-tabs", TabbedContent)
        active = tabs.active
        if active == "chat-pane":
            return  # never close the chat tab
        # Check for unsaved changes
        if active in self._dirty_files:
            file_path = self._open_files.get(active, "unknown")
            filename = pathlib.Path(file_path).name
            self.push_screen(
                UnsavedChangesScreen(filename),
                callback=lambda result: self.call_later(
                    self._handle_unsaved_close, active, result
                ),
            )
            return
        await self._do_close_tab(active)

    async def _handle_unsaved_close(self, tab_id: str, result: str) -> None:
        """Handle the result of the unsaved changes modal."""
        if result == "save":
            # Save first, then close
            file_path = self._open_files.get(tab_id, "")
            try:
                editor = self.query_one(f"#editor-{tab_id}", TextArea)
                content = editor.text
                pathlib.Path(file_path).write_text(content)
                self._original_content[tab_id] = content
                self._dirty_files.discard(tab_id)
                self.notify(f"Saved {pathlib.Path(file_path).name}")
            except Exception as e:
                self.notify(f"Error saving: {e}", severity="error")
                return
            await self._do_close_tab(tab_id)
        elif result == "discard":
            self._dirty_files.discard(tab_id)
            await self._do_close_tab(tab_id)
        # "cancel" → do nothing

    async def _do_close_tab(self, tab_id: str) -> None:
        """Actually remove a tab and clean up state."""
        self._open_files.pop(tab_id, None)
        self._original_content.pop(tab_id, None)
        self._dirty_files.discard(tab_id)
        self._tab_names.pop(tab_id, None)
        # Remove search bar if present
        try:
            pane = self.query_one(f"#{tab_id}", TabPane)
            for sb in pane.query("#search-bar"):
                sb.remove()
        except Exception:
            pass
        tabs = self.query_one("#content-tabs", TabbedContent)

        # Find the nearest neighbour tab before removing
        tab_ids = [str(t.id) for t in tabs.query("TabPane")]
        next_tab = "chat-pane"
        if tab_id in tab_ids:
            idx = tab_ids.index(tab_id)
            # Prefer the next file tab, then previous, then chat
            remaining = [t for t in tab_ids if t != tab_id and t != "chat-pane"]
            if remaining:
                # Pick the tab that was adjacent: try same index, else last
                file_idx = [tab_ids.index(t) for t in remaining]
                # Closest after current, or closest before
                after = [t for t in remaining if tab_ids.index(t) > idx]
                before = [t for t in remaining if tab_ids.index(t) < idx]
                next_tab = after[0] if after else before[-1]

        try:
            await tabs.remove_pane(tab_id)
        except Exception:
            pass

        tabs.active = next_tab
        if next_tab == "chat-pane":
            self.query_one("#chat-input", ChatInput).focus()

    # ── File watcher integration ────────────────────────

    def _start_file_watcher(self):
        """Initialize and start the file watcher with background indexing."""
        workspace = os.getcwd()

        # Start background indexing queue
        project_id = getattr(self, '_project_id', None) or 1
        self._index_queue = IndexQueue(project_id=project_id)
        self._index_queue.start()

        self._file_watcher = FileWatcher(
            workspace=workspace,
            callback=self._on_file_change,
            visible_paths_callback=self._get_visible_paths,
            index_callback=self._index_queue.enqueue,
        )
        self._watcher_started = self._file_watcher.start()
        if self._watcher_started:
            self.add_message("System", "File watcher enabled", "system")

    def _on_file_change(self, file_path: str):
        """Callback when a file change is detected in a visible directory."""
        self.call_from_thread(self._refresh_visible_tree, str(file_path))

    def _refresh_visible_tree(self, changed_path: str):
        """Refresh only the visible portion of the file tree containing the changed file."""
        try:
            tree = self.query_one("#file-tree", DirectoryTree)
            # Refresh the entire tree - Textual handles incremental updates efficiently
            tree.refresh()
            logger = logging.getLogger(__name__)
            logger.debug(f"File tree refreshed due to change: {changed_path}")
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.error(f"Failed to refresh file tree: {e}")

    def _get_visible_paths(self) -> Set[str]:
        """Get set of currently expanded/visible directory paths."""
        visible_paths = set()
        try:
            tree = self.query_one("#file-tree", DirectoryTree)
            # Track expanded nodes from the tree
            for node_id in tree._expanded_nodes:
                try:
                    node = tree.get_node(node_id)
                    if node and hasattr(node, 'path'):
                        visible_paths.add(str(node.path))
                except Exception:
                    continue
        except Exception:
            pass
        return visible_paths

    def on_exit_app(self) -> None:
        """Clean up file watcher and index queue on app exit."""
        if self._file_watcher and self._file_watcher.is_running():
            self._file_watcher.stop()
        if self._index_queue and self._index_queue.is_running():
            self._index_queue.stop()
        logger = logging.getLogger(__name__)
        logger.info("File watcher and index queue stopped on app exit")

    # ── File explorer events ─────────────────────────────

    @on(Button.Pressed, "#btn-sync-tree")
    def sync_file_tree(self, event: Button.Pressed) -> None:
        """Reload the file tree from disk."""
        tree = self.query_one("#file-tree", DirectoryTree)
        tree.reload()

    @on(DirectoryTree.FileSelected)
    async def open_file(self, event: DirectoryTree.FileSelected) -> None:
        """Open a file in a new editor tab (or image viewer for images)."""
        file_path = str(event.path)
        tab_id = f"file-{hash(file_path) & 0xFFFFFFFF:08x}"

        # If already open, just switch to it
        if tab_id in self._open_files:
            self.query_one("#content-tabs", TabbedContent).active = tab_id
            return

        # Truncate display name
        rel = os.path.relpath(file_path)
        tab_name = rel if len(rel) < 30 else f".../{pathlib.Path(file_path).name}"
        tabs = self.query_one("#content-tabs", TabbedContent)

        # Image files → open in image viewer
        if is_image_file(file_path):
            viewer = ImageViewer(
                file_path,
                id=f"imgview-{tab_id}",
                classes="image-viewer",
            )
            pane = TabPane(tab_name, viewer, id=tab_id)
            await tabs.add_pane(pane)
            self._open_files[tab_id] = file_path
            self._tab_names[tab_id] = tab_name
            tabs.active = tab_id
            viewer.focus()
            return

        # Read file content
        try:
            content = pathlib.Path(file_path).read_text(errors="replace")
        except Exception as e:
            self.add_message("System", f"Cannot open file: {e}", "system")
            return

        # Determine language for syntax highlighting
        ext = pathlib.Path(file_path).suffix.lower()
        language = _EXT_LANG.get(ext)

        # Create the tab and wait for it to be mounted
        editor = TextArea(
            content,
            language=language,
            theme="monokai",
            show_line_numbers=True,
            id=f"editor-{tab_id}",
            classes="file-editor",
        )
        pane = TabPane(tab_name, editor, id=tab_id)
        await tabs.add_pane(pane)
        self._open_files[tab_id] = file_path
        self._original_content[tab_id] = content
        self._tab_names[tab_id] = tab_name
        tabs.active = tab_id
        editor.focus()

    # ── Dirty state tracking ─────────────────────────────

    @on(TextArea.Changed)
    def _on_editor_changed(self, event: TextArea.Changed) -> None:
        """Track dirty state when a file editor is modified."""
        editor = event.text_area
        if not editor.has_class("file-editor"):
            return
        # Find the tab_id from the editor id (editor-{tab_id})
        editor_id = editor.id or ""
        if not editor_id.startswith("editor-"):
            return
        tab_id = editor_id[len("editor-"):]
        if tab_id not in self._open_files:
            return
        is_dirty = editor.text != self._original_content.get(tab_id, "")
        was_dirty = tab_id in self._dirty_files
        if is_dirty and not was_dirty:
            self._dirty_files.add(tab_id)
            self._update_tab_label(tab_id, dirty=True)
        elif not is_dirty and was_dirty:
            self._dirty_files.discard(tab_id)
            self._update_tab_label(tab_id, dirty=False)

    def _update_tab_label(self, tab_id: str, dirty: bool) -> None:
        """Update the tab label to show/hide the dirty indicator."""
        base_name = self._tab_names.get(tab_id, "")
        new_label = f"\u25cf {base_name}" if dirty else base_name
        try:
            tabs = self.query_one("#content-tabs", TabbedContent)
            tab = tabs.get_tab(tab_id)
            tab.label = new_label
        except Exception:
            pass
        self._refresh_explorer_dirty()

    def _get_dirty_paths(self) -> dict[str, str]:
        """Return {tab_id: file_path} for all dirty files."""
        return {tid: self._open_files[tid] for tid in self._dirty_files if tid in self._open_files}

    def _refresh_explorer_dirty(self) -> None:
        """Update the file explorer to highlight dirty files."""
        try:
            tree = self.query_one("#file-tree", InfinidevDirectoryTree)
            dirty_paths = {self._open_files[tid] for tid in self._dirty_files if tid in self._open_files}
            tree.set_dirty_paths(dirty_paths)
        except Exception:
            pass

    # ── Save file ─────────────────────────────────────────

    async def action_save_file(self) -> None:
        """Save the active file editor tab to disk (Ctrl+S)."""
        tabs = self.query_one("#content-tabs", TabbedContent)
        active = tabs.active
        if active == "chat-pane" or active not in self._open_files:
            return
        file_path = self._open_files[active]
        try:
            editor = self.query_one(f"#editor-{active}", TextArea)
        except Exception:
            return
        content = editor.text
        try:
            pathlib.Path(file_path).write_text(content)
        except Exception as e:
            self.notify(f"Error saving: {e}", severity="error")
            return
        self._original_content[active] = content
        self._dirty_files.discard(active)
        self._update_tab_label(active, dirty=False)
        filename = pathlib.Path(file_path).name
        self.notify(f"Saved {filename}")

    # ── Find in file ──────────────────────────────────────

    async def action_find_in_file(self) -> None:
        """Open search bar for the active file editor (Ctrl+F)."""
        tabs = self.query_one("#content-tabs", TabbedContent)
        active = tabs.active
        if active == "chat-pane" or active not in self._open_files:
            return
        # Don't open a second search bar
        existing = self.query("#search-bar")
        if existing:
            existing.first().query_one("#search-input", Input).focus()
            return
        try:
            editor = self.query_one(f"#editor-{active}", TextArea)
        except Exception:
            return
        pane = self.query_one(f"#{active}", TabPane)
        search_bar = SearchBar(editor)
        await pane.mount(search_bar, before=editor)

    # ── Find in project ───────────────────────────────────

    def action_find_in_project(self) -> None:
        """Open project-wide search modal (Ctrl+Shift+F)."""
        self.push_screen(ProjectSearchScreen())

    def open_file_at_line(self, file_path: str, line_no: int) -> None:
        """Open a file and scroll to a specific line (used by project search)."""
        tab_id = f"file-{hash(file_path) & 0xFFFFFFFF:08x}"
        # Schedule via call_later to allow modal dismiss to complete
        self.call_later(self._open_file_at_line_async, file_path, tab_id, line_no)

    async def _open_file_at_line_async(self, file_path: str, tab_id: str, line_no: int) -> None:
        """Internal: open file and jump to line."""
        tabs = self.query_one("#content-tabs", TabbedContent)
        if tab_id not in self._open_files:
            # Read and open the file
            try:
                content = pathlib.Path(file_path).read_text(errors="replace")
            except Exception as e:
                self.add_message("System", f"Cannot open file: {e}", "system")
                return
            ext = pathlib.Path(file_path).suffix.lower()
            language = _EXT_LANG.get(ext)
            rel = os.path.relpath(file_path)
            tab_name = rel if len(rel) < 30 else f".../{pathlib.Path(file_path).name}"
            editor = TextArea(
                content, language=language, theme="monokai",
                show_line_numbers=True, id=f"editor-{tab_id}", classes="file-editor",
            )
            pane = TabPane(tab_name, editor, id=tab_id)
            await tabs.add_pane(pane)
            self._open_files[tab_id] = file_path
            self._original_content[tab_id] = content
            self._tab_names[tab_id] = tab_name
        tabs.active = tab_id
        # Jump to line
        try:
            editor = self.query_one(f"#editor-{tab_id}", TextArea)
            editor.focus()
            target_line = max(0, line_no - 1)
            editor.move_cursor((target_line, 0))
            editor.scroll_cursor_visible()
        except Exception:
            pass

    # ── Autocomplete ─────────────────────────────────────

    @on(TextArea.Changed, "#chat-input")
    def show_autocomplete(self, event: TextArea.Changed):
        text = event.text_area.text.lstrip()
        menu = self.query_one("#autocomplete-menu", OptionList)
        if text.startswith("/"):
            matches = [
                Option(f"{cmd}  {desc}", id=cmd)
                for cmd, desc in COMMANDS
                if cmd.startswith(text)
            ]
            if matches:
                menu.clear_options()
                menu.add_options(matches)
                menu.add_class("-visible")
                return
        menu.remove_class("-visible")

    @on(OptionList.OptionSelected, "#autocomplete-menu")
    def apply_autocomplete(self, event: OptionList.OptionSelected):
        cmd = event.option.id
        input_widget = self.query_one("#chat-input", ChatInput)
        input_widget.text = f"{cmd} "
        input_widget.focus()
        input_widget.move_cursor(input_widget.document.end)
        self.query_one("#autocomplete-menu", OptionList).remove_class("-visible")

    def on_key(self, event: events.Key) -> None:
        """Dismiss autocomplete on Escape from anywhere."""
        menu = self.query_one("#autocomplete-menu", OptionList)
        if event.key == "escape" and menu.has_class("-visible"):
            event.prevent_default()
            menu.remove_class("-visible")
            self.query_one("#chat-input", ChatInput).focus()

    # ── Engine events ────────────────────────────────────

    def on_loop_event(self, event_type: str, project_id: int, agent_id: str, data: dict[str, Any]):
        self.call_from_thread(self.process_event, event_type, data)

    def process_event(self, event_type: str, data: dict[str, Any]):
        try:
            self._process_event_inner(event_type, data)
        except Exception:
            import traceback
            tb = traceback.format_exc()
            try:
                self._log_lines.append(f"✗ UI event error: {event_type}")
                self._log_lines = self._log_lines[-10:]
                self.query_one("#logs-panel").update_content("\n".join(self._log_lines))
            except Exception:
                pass
            import logging
            logging.getLogger("infinidev.tui").debug("process_event(%s) failed:\n%s", event_type, tb)

    def _process_event_inner(self, event_type: str, data: dict[str, Any]):
        if event_type == "loop_step_update":
            steps = data.get("plan_steps", [])
            if steps:
                steps_text = ""
                for s in steps:
                    icon = "✓" if s["status"] == "done" else "○"
                    if s["status"] == "active":
                        icon = "●"
                    steps_text += f"{icon} {s['description']}\n"
                self.query_one("#steps-panel").update_content(steps_text)
            else:
                self.query_one("#steps-panel").update_content("Waiting for plan...")

            desc = data.get("step_description", "")
            summary = data.get("summary", "")
            iteration = data.get("iteration", 0)
            status = data.get("status", "")
            plan_text = f"Step {iteration}: {desc}" if iteration else desc
            if summary:
                plan_text += f"\n{summary}"
            if status and status != "active":
                plan_text += f"\n[{status}]"
            self.query_one("#plan-panel").update_content(plan_text)

            # Update context window display with real LLM-reported tokens
            self._refresh_context_panel(
                task_tokens=data.get("tokens_total", 0),
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
            )

        elif event_type == "loop_tool_call":
            tool_name = data.get("tool_name", "")
            tool_detail = data.get("tool_detail", "")
            action_text = f"⚙️ {tool_name}\n"
            if tool_detail:
                action_text += f"   {tool_detail}"
            self.query_one("#actions-panel").update_content(action_text)

            # Update context window after each tool call
            self._refresh_context_panel(
                task_tokens=data.get("tokens_total", 0),
                prompt_tokens=data.get("prompt_tokens", 0),
                completion_tokens=data.get("completion_tokens", 0),
            )

        elif event_type == "loop_user_message":
            msg = data.get("message", "")
            if msg:
                self.add_message("Infinidev", msg, "agent")

        elif event_type == "loop_file_changed":
            path = data.get("path", "")
            diff = data.get("diff", "")
            action = data.get("action", "modified")
            num_changes = data.get("num_changes", 1)
            if path in self._file_diff_widgets:
                self._file_diff_widgets[path].update_diff(diff, num_changes, action)
            else:
                widget = FileChangeDiffWidget(path, diff, action)
                self._file_diff_widgets[path] = widget
                history = self.query_one("#chat-history")
                thinking = self.query(".thinking-indicator")
                if thinking:
                    history.mount(widget, before=thinking.first())
                else:
                    history.mount(widget)
                history.scroll_end(animate=False)


        elif event_type == "loop_think":
            reasoning = data.get("reasoning", "").strip()
            if reasoning:
                history = self.query_one("#chat-history")
                container = Vertical(classes="think-msg")
                from rich.markup import escape as _escape_markup
                header = self._safe_static("[bold #b0a0d8]💭 Thinking:[/bold #b0a0d8]")
                body = self._safe_static(f"[dim #c0b8e0]{_escape_markup(reasoning)}[/dim #c0b8e0]")
                thinking_widgets = self.query(".thinking-indicator")
                if thinking_widgets:
                    history.mount(container, before=thinking_widgets.first())
                else:
                    history.mount(container)
                container.mount(header)
                container.mount(body)
                history.scroll_end(animate=False)

        elif event_type == "loop_log":
            level = data.get("level", "warning")
            msg = data.get("message", "")
            icon = "⚠" if level == "warning" else "✗"
            line = f"{icon} {msg}"
            self._log_lines.append(line)
            self._log_lines = self._log_lines[-10:]
            self.query_one("#logs-panel").update_content("\n".join(self._log_lines))
            self.set_timer(10.0, lambda l=line: self._expire_log_line(l))

        # ── Tree engine events ───────────────────────────────
        elif event_type == "tree_init":
            root = data.get("root_problem", "")
            n = data.get("num_children", 0)
            logic = data.get("logic", "AND")
            tree_text = f"🌳 {root[:80]}\n   {logic} → {n} sub-problems"
            self.query_one("#plan-panel").update_content(tree_text)
            self.query_one("#steps-panel").update_content("Initializing tree...")
            self.query_one("#actions-panel").update_content("Tree decomposed")

        elif event_type == "tree_node_exploring":
            node_id = data.get("node_id", "")
            problem = data.get("problem", "")
            depth = data.get("depth", 0)
            indent = "  " * depth
            step_text = f"🔍 [{node_id}]\n{indent}{problem[:100]}"
            self.query_one("#steps-panel").update_content(step_text)
            self.query_one("#actions-panel").update_content(f"Exploring [{node_id}]...")
            self._refresh_context_panel(prompt_tokens=data.get("prompt_tokens", 0))

        elif event_type == "tree_tool_call":
            node_id = data.get("node_id", "")
            tool = data.get("tool_name", "")
            args = data.get("args_preview", "")
            action_text = f"⚙️ [{node_id}] {tool}"
            if args:
                action_text += f"\n   {args[:60]}"
            self.query_one("#actions-panel").update_content(action_text)
            self._refresh_context_panel(prompt_tokens=data.get("prompt_tokens", 0))

        elif event_type == "tree_node_resolved":
            node_id = data.get("node_id", "")
            state = data.get("state", "")
            conf = data.get("confidence", "")
            summary = data.get("summary", "")
            state_icon = {"solvable": "✅", "unsolvable": "❌", "mitigable": "⚠️",
                          "needs_decision": "❓", "needs_experiment": "🧪"}.get(state, "●")
            short_line = f"{state_icon} [{node_id}] {state} ({conf})"
            # Accumulate for plan panel tree view
            self._tree_resolved_lines.append(short_line)
            # Also show in steps panel with summary
            step_text = short_line
            if summary:
                step_text += f"\n   {summary[:80]}"
            self.query_one("#steps-panel").update_content(step_text)
            # Log it
            log_line = short_line
            if summary:
                log_line += f" — {summary[:60]}"
            self._log_lines.append(log_line)
            self._log_lines = self._log_lines[-15:]
            self.query_one("#logs-panel").update_content("\n".join(self._log_lines))
            self._refresh_context_panel(prompt_tokens=data.get("prompt_tokens", 0))

        elif event_type == "tree_propagation":
            root_state = data.get("root_state", "?")
            total = data.get("total_nodes", 0)
            resolved = data.get("resolved_nodes", 0)
            # Update the plan panel with tree progress
            pct = (resolved / total * 100) if total > 0 else 0
            bar_len = 20
            filled = int(bar_len * resolved / total) if total > 0 else 0
            bar = "█" * filled + "░" * (bar_len - filled)
            tree_text = (
                f"🌳 Root: {root_state}\n"
                f"   {bar} {resolved}/{total} ({pct:.0f}%)"
            )
            # Add resolved nodes as tree lines
            if self._tree_resolved_lines:
                tree_text += "\n" + "\n".join(self._tree_resolved_lines[-8:])
            self.query_one("#plan-panel").update_content(tree_text)

        elif event_type == "tree_fact_discovered":
            node_id = data.get("node_id", "")
            fact = data.get("fact_content", "")
            tool = data.get("source_tool", "")
            line = f"💡 [{node_id}] {fact[:60]}"
            if tool:
                line += f" (via {tool})"
            self._log_lines.append(line)
            self._log_lines = self._log_lines[-15:]
            self.query_one("#logs-panel").update_content("\n".join(self._log_lines))

        elif event_type == "tree_synthesizing":
            total = data.get("total_nodes", 0)
            self.query_one("#steps-panel").update_content(f"📝 Synthesizing {total} nodes...")
            self.query_one("#actions-panel").update_content("Generating synthesis...")

        elif event_type == "tree_budget_warning":
            used = data.get("used", 0)
            limit = data.get("limit", 0)
            btype = data.get("type", "")
            line = f"⚠ Budget {btype}: {used}/{limit}"
            self._log_lines.append(line)
            self._log_lines = self._log_lines[-15:]
            self.query_one("#logs-panel").update_content("\n".join(self._log_lines))

        elif event_type == "tree_finished":
            status = data.get("status", "?")
            total = data.get("total_nodes", 0)
            self.query_one("#steps-panel").update_content(
                f"🏁 Complete: {total} nodes\n   Root: {status}"
            )
            self.query_one("#actions-panel").update_content("Idle")
            # Clean up accumulated tree state
            self._tree_resolved_lines.clear()

        # ── Analysis events ───────────────────────────────
        elif event_type == "analysis_start":
            round_num = data.get("round", 1)
            self.query_one("#actions-panel").update_content(
                f"Analyzing request... (round {round_num})"
            )

        elif event_type == "analysis_research":
            queries = data.get("queries", [])
            preview = ", ".join(q[:30] for q in queries[:2])
            self.query_one("#actions-panel").update_content(
                f"Researching: {preview}"
            )

        elif event_type == "analysis_complete":
            action = data.get("action", "")
            self.query_one("#actions-panel").update_content(
                f"Analysis: {action}"
            )

        # ── Review events ─────────────────────────────────
        elif event_type == "review_start":
            self.query_one("#actions-panel").update_content("Code review...")

        elif event_type == "review_complete":
            verdict = data.get("verdict", "")
            issues = data.get("issue_count", 0)
            text = f"Review: {verdict}"
            if issues:
                text += f" ({issues} issues)"
            self.query_one("#actions-panel").update_content(text)

        # ── Gather events ─────────────────────────────────
        elif event_type == "gather_status":
            text = data.get("text", "")
            self.query_one("#actions-panel").update_content(text)

        elif event_type == "gather_error":
            msg = data.get("message", "")
            self.query_one("#actions-panel").update_content(f"Gather skipped: {msg}")

    def _expire_log_line(self, line: str) -> None:
        try:
            self._log_lines.remove(line)
        except ValueError:
            return
        self.query_one("#logs-panel").update_content("\n".join(self._log_lines) if self._log_lines else "")

    # ── Chat messages ────────────────────────────────────

    _SENDER_COLORS = {"user": "#6fbf6f", "agent": "#7a9fd4", "system": "#c8c870"}

    @staticmethod
    def _safe_static(content: str) -> Static:
        """Create a Static widget, falling back to plain text if markup is invalid."""
        import re
        try:
            # Test-render the markup to catch errors before mounting
            from rich.text import Text
            Text.from_markup(content)
            return Static(content)
        except Exception:
            # Strip all Rich markup tags and show plain text
            plain = re.sub(r"\[/?[^\]]*\]", "", content)
            return Static(plain, markup=False)

    def _is_thinking(self) -> bool:
        """Return True if the thinking indicator is currently visible."""
        return len(self.query(".thinking-indicator")) > 0

    def add_message(self, sender: str, text: str, type: str = "agent", queued: bool = False, queue_index: int | None = None):
        """Add a message to the chat history.

        Args:
            sender: Message sender (e.g., "You", "Infinidev")
            text: Message content
            type: Message type ("user", "agent", "system")
            queued: If True, render as a queued message (faded)
            queue_index: Queue position if queued (1-based)
        """
        history = self.query_one("#chat-history")
        color = self._SENDER_COLORS.get(type, "#cccccc")
        text = str(text) if text else ""

        # If thinking is active and this is a user message, show as pending
        is_pending = type == "user" and self._is_thinking()
        msg_class = "pending-msg" if is_pending else f"{type}-msg"

        container = Vertical(classes=msg_class)
        header_text = f"[bold {color}]{sender}:[/bold {color}]"
        if is_pending:
            header_text = f"[dim bold]{sender} (pending):[/dim bold]"
        header = self._safe_static(header_text)

        if type == "agent" and ("```" in text or "**" in text or "# " in text):
            body = Markdown(text)
        else:
            body = self._safe_static(text)

        # Add queue indicator if queued
        if queued:
            queue_label = Static(
                f"[dim]● Queued message #{queue_index} - Not yet processed[/dim]",
                classes="queue-indicator"
            )
            container.mount(queue_label)

        # If thinking, mount read messages before the indicator, pending after
        thinking_widgets = self.query(".thinking-indicator")
        if thinking_widgets and not is_pending:
            # Non-pending message while thinking → place before the indicator
            history.mount(container, before=thinking_widgets.first())
        else:
            history.mount(container)

        container.mount(header)
        container.mount(body)
        history.scroll_end(animate=False)

    def add_queued_message(self, sender: str, text: str, type: str = "agent") -> QueuedMessageWidget:
        """Add a message to the queue (not yet processed).

        Creates a queued message widget that is rendered with a faded appearance
        until the model processes it.
        """
        from infinidev.ui.widgets.context_widgets import QueuedMessageWidget, QueuedMessageStatus

        history = self.query_one("#chat-history")

        widget = QueuedMessageWidget(
            content=text,
            user=sender,
            queued_index=len(self._queued_messages) + 1
        )

        history.mount(widget)
        self._queued_messages.append(widget)

        # Mark this as a user message queuing up the agent
        self._show_thinking()

        return widget

    def process_queued_message(self, widget: QueuedMessageWidget) -> None:
        """Promote a queued message to a normal message and update its status."""
        # Update widget status to processed
        widget.update_status(QueuedMessageStatus.PROCESSED)

        # Remove from queued list
        if widget in self._queued_messages:
            self._queued_messages.remove(widget)

        # Scroll to show the newly processed message
        history = self.query_one("#chat-history")
        history.scroll_into_view(widget, animate=True)

        # Auto-dismiss thinking indicator if this was the first queued message
        if len(self._queued_messages) == 0:
            self._hide_thinking()

    # ── Submit / engine ──────────────────────────────────

    @on(ChatInput.Submitted)
    def handle_submit(self, event: ChatInput.Submitted):
        self.query_one("#autocomplete-menu", OptionList).remove_class("-visible")
        user_text = event.value
        self.add_message("You", user_text, "user")

        # If we're waiting for an analysis answer, feed it back
        if self._analysis_waiting and self._analysis_event is not None:
            self._analysis_answer = user_text
            self._analysis_waiting = False
            self._analysis_event.set()
            return

        if user_text.startswith("!"):
            # Shell command: execute directly
            self._execute_shell_command(user_text[1:])  # Strip the '!' prefix
        elif user_text.startswith("/"):
            self.handle_command(user_text)
        else:
            if self._engine_running:
                self._pending_inputs.append(user_text)
                self.add_message("System", "Queued — waiting for current task to finish.", "system")
            else:
                self._engine_running = True
                self._show_thinking()
                self.run_engine(user_text)

    def _update_status_bar(self) -> None:
        from infinidev.config.settings import settings
        model = settings.LLM_MODEL.split("/", 1)[-1] if "/" in settings.LLM_MODEL else settings.LLM_MODEL
        cwd = os.path.basename(os.getcwd())
        self.query_one("#status-bar", Static).update(f" Model: {model}  │  Project: {cwd}")

    def _show_thinking(self) -> None:
        """Mount the thinking indicator at the bottom of chat history."""
        # Remove any leftover indicators from a previous run
        self._hide_thinking()
        history = self.query_one("#chat-history")
        indicator = Vertical(
            Static("Infinidev is thinking..."),
            LoadingIndicator(),
            classes="thinking-indicator",
        )
        history.mount(indicator)
        history.scroll_end(animate=False)

    def _hide_thinking(self) -> None:
        """Remove all thinking indicators and promote pending messages."""
        for widget in self.query(".thinking-indicator"):
            widget.remove()
        # Promote pending messages to normal user messages
        color = self._SENDER_COLORS.get("user", "#cccccc")
        for widget in list(self.query(".pending-msg")):
            widget.remove_class("pending-msg")
            widget.add_class("user-msg")
            # Update header: first Static child is always the header
            try:
                statics = widget.query(Static)
                if statics:
                    statics.first().update(f"[bold {color}]You:[/bold {color}]")
            except Exception:
                pass

    def _execute_shell_command(self, command: str):
        """Execute a shell command directly and display output."""
        import shlex
        import subprocess

        if not command or not command.strip():
            self.add_message("System", "No command specified.", "system")
            return

        self.add_message("Shell", f"Executing: {command}", "system")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd(),
            )

            output_lines = []
            if result.stdout:
                output_lines.append(f"[bold #00ff88]stdout:[/bold #00ff88]\n{result.stdout}")
            if result.stderr:
                output_lines.append(f"[bold #ff4466]stderr:[/bold #ff4466]\n{result.stderr}")

            output_text = "\n".join(output_lines) if output_lines else "(no output)"
            exit_info = f"\n[bold]Exit code:[/bold] {result.returncode}"

            self.add_message("Shell", output_text + exit_info, "system")

        except subprocess.TimeoutExpired:
            self.add_message("Shell", "Command timed out after 60 seconds.", "system")
        except Exception as e:
            self.add_message("Shell", f"Execution failed: {e}", "system")

    def _handle_permission_request(self, tool_name: str, description: str, details: str) -> bool:
        """Handle a permission request from a tool. Called from worker thread.

        Shows the command in the ACTIONS side panel with Allow/Deny buttons.
        Blocks the engine thread until the user clicks a button.
        """
        import threading

        evt = threading.Event()
        self._permission_event = evt
        self._permission_waiting = True
        self._permission_approved = False

        # Schedule UI update on main thread
        self.call_from_thread(self._show_permission_ui, description, details)

        # Block the engine thread until user clicks a button
        evt.wait()

        approved = self._permission_approved
        self._permission_event = None
        self._permission_waiting = False

        return approved

    def _show_permission_ui(self, description: str, details: str) -> None:
        """Show permission prompt + buttons inside the ACTIONS panel.

        Layout: title → buttons → preview (buttons always visible at top).
        """
        # Store full info for the "View" modal
        self._permission_details = f"{description}\n\n{details}"

        panel = self.query_one("#actions-panel")

        # Always show view button — keep panel text minimal so buttons fit
        preview = details.split("\n")[0][:80]
        if len(details) > len(preview):
            preview += "..."

        panel.update_content(
            f"[bold yellow]Permission Required[/bold yellow]\n"
            f"{preview}"
        )

        # Mount buttons below the text
        btn_row = Horizontal(id="perm-btn-row")
        panel.mount(btn_row, after=panel.content_static)
        view_btn = Button("View", variant="primary", id="btn-perm-view")
        allow_btn = Button("Allow", variant="success", id="btn-perm-allow")
        deny_btn = Button("Deny", variant="error", id="btn-perm-deny")
        btn_row.mount(view_btn)
        btn_row.mount(allow_btn)
        btn_row.mount(deny_btn)
        allow_btn.focus()

    @on(Button.Pressed, "#btn-perm-allow")
    def _on_perm_allow(self, event: Button.Pressed) -> None:
        event.stop()
        self._resolve_permission(True)

    @on(Button.Pressed, "#btn-perm-deny")
    def _on_perm_deny(self, event: Button.Pressed) -> None:
        event.stop()
        self._resolve_permission(False)

    @on(Button.Pressed, "#btn-perm-view")
    def _on_perm_view(self, event: Button.Pressed) -> None:
        """Open modal to view full permission details with syntax highlighting."""
        event.stop()
        details = getattr(self, "_permission_details", "")
        self.push_screen(PermissionDetailScreen(details))

    def _resolve_permission(self, approved: bool) -> None:
        """Handle a permission button click."""
        # Remove permission UI elements
        for widget_id in ("#perm-btn-row", "#btn-perm-view"):
            try:
                self.query_one(widget_id).remove()
            except Exception:
                pass
        self.query_one("#actions-panel").update_content(
            "[green]Approved — continuing...[/green]" if approved else "[red]Denied[/red]"
        )
        # Unblock the engine thread
        if self._permission_event is not None:
            self._permission_approved = approved
            self._permission_waiting = False
            self._permission_event.set()

    def _drain_pending_inputs(self) -> None:
        """Process the next queued input, if any."""
        if self._pending_inputs:
            next_input = self._pending_inputs.pop(0)
            self._engine_running = True
            self._show_thinking()
            self.run_engine(next_input)

    @work(exclusive=True, thread=True)
    def run_engine(self, user_input: str):
        import threading

        self._file_diff_widgets = {}
        try:
            reload_all()
            store_conversation_turn(self.session_id, 'user', user_input)
            summaries = get_recent_summaries(self.session_id, limit=10)
            self.agent._session_summaries = summaries

            # Update chat context usage
            self.context_calculator.update_chat(user_input, summaries)
            self.call_from_thread(
                self._context_panel.update_status,
                self.context_calculator.get_context_status(),
            )

            # --- Analysis phase ---
            from infinidev.config.settings import settings as _settings
            if _settings.ANALYSIS_ENABLED:
                self.analyst.reset()

                analysis_input = user_input
                analysis = self.analyst.analyze(
                    analysis_input,
                    session_summaries=summaries,
                )

                # Handle question loop
                while analysis.action == "ask" and self.analyst.can_ask_more:
                    questions_text = analysis.format_questions_for_user()
                    self.call_from_thread(self._hide_thinking)
                    self.call_from_thread(
                        self.add_message, "Analyst", questions_text, "agent"
                    )

                    # Wait for user to answer
                    self._analysis_event = threading.Event()
                    self._analysis_waiting = True
                    self._analysis_answer = ""
                    self._analysis_original_input = user_input

                    self._analysis_event.wait()  # Blocks until user responds

                    answer = self._analysis_answer
                    self._analysis_event = None

                    if not answer.strip():
                        break

                    self.analyst.add_answer(questions_text, answer)
                    self.call_from_thread(self._show_thinking)
                    analysis_input = user_input + "\n\nUser clarification: " + answer
                    analysis = self.analyst.analyze(
                        analysis_input,
                        session_summaries=summaries,
                    )

                # Build enriched task prompt
                task_prompt = analysis.build_flow_prompt()

                # Handle "done" pseudo-flow (greetings, simple questions)
                if analysis.flow == "done":
                    self.call_from_thread(
                        self._context_panel.set_flow, "done"
                    )
                    self.call_from_thread(self._hide_thinking)
                    self.call_from_thread(
                        self.add_message, "Infinidev",
                        analysis.reason or analysis.original_input, "agent"
                    )
                    store_conversation_turn(
                        self.session_id, 'assistant',
                        analysis.reason or analysis.original_input,
                        (analysis.reason or analysis.original_input)[:200],
                    )
                    self.call_from_thread(
                        self.query_one("#actions-panel").update_content, "Idle"
                    )
                    return

                if analysis.action == "proceed":
                    # Show spec and wait for user confirmation
                    spec = analysis.specification
                    spec_parts = []
                    if spec.get("summary"):
                        spec_parts.append(f"**Summary:** {spec['summary']}")
                    for key in ("requirements", "hidden_requirements", "assumptions", "out_of_scope"):
                        items = spec.get(key, [])
                        if items:
                            spec_parts.append(f"\n**{key.replace('_', ' ').title()}:**")
                            for item in items:
                                spec_parts.append(f"  • {item}")
                    if spec.get("technical_notes"):
                        spec_parts.append(f"\n**Technical Notes:** {spec['technical_notes']}")
                    spec_parts.append("\n*Proceed with implementation? Type **y** to proceed, **n** to cancel, or add feedback.*")

                    self.call_from_thread(self._hide_thinking)
                    self.call_from_thread(
                        self.add_message, "Analyst", "\n".join(spec_parts), "agent"
                    )

                    # Wait for user confirmation
                    self._analysis_event = threading.Event()
                    self._analysis_waiting = True
                    self._analysis_answer = ""
                    self._analysis_original_input = user_input
                    self._analysis_event.wait()

                    confirm = self._analysis_answer.strip()
                    self._analysis_event = None

                    if confirm.lower() in ("n", "no", "cancel"):
                        self.call_from_thread(self._hide_thinking)
                        self.call_from_thread(
                            self.add_message, "System", "Development skipped.", "system"
                        )
                        self.call_from_thread(
                            self.query_one("#actions-panel").update_content, "Idle"
                        )
                        return
                    if confirm and confirm.lower() not in ("y", "yes", ""):
                        # User gave extra feedback — append to the task prompt
                        desc, expected = task_prompt
                        desc += f"\n\n## Additional User Feedback\n{confirm}"
                        task_prompt = (desc, expected)

                    self.call_from_thread(self._show_thinking)

                # Get flow config and configure agent
                from infinidev.engine.flows import get_flow_config
                flow_config = get_flow_config(analysis.flow)
                self.agent._system_prompt_identity = flow_config.identity_prompt
                self.agent.backstory = flow_config.backstory

                # Override expected output with flow-specific template
                desc, _ = task_prompt
                task_prompt = (desc, flow_config.expected_output)
            else:
                task_prompt = (user_input, "Complete the task and report findings.")
                flow_config = None
            # --- End analysis phase ---

            # --- Gather phase ---
            flow_label = analysis.flow if _settings.ANALYSIS_ENABLED else "develop"
            _do_gather = _settings.GATHER_ENABLED or self._gather_next_task
            self._gather_next_task = False  # Reset after use
            if _do_gather and flow_label == "develop":
                try:
                    from infinidev.gather import run_gather

                    chat_history = [
                        {"role": "user" if "[user]" in s.lower() else "assistant", "content": s}
                        for s in get_recent_summaries(self.session_id, limit=10)
                    ]
                    brief = run_gather(user_input, chat_history, analysis, self.agent)
                    desc, expected = task_prompt
                    task_prompt = (brief.render() + "\n\n" + desc, expected)
                    from infinidev.flows.event_listeners import event_bus as _eb
                    _eb.emit("gather_status", 0, "", {"text": f"Gathered: {brief.summary()}"})
                except Exception as exc:
                    from infinidev.flows.event_listeners import event_bus as _eb
                    _eb.emit("gather_error", 0, "", {"message": str(exc)})
            # --- End gather phase ---

            # --- Development phase ---
            self.call_from_thread(
                self._context_panel.set_flow, flow_label
            )
            self.call_from_thread(
                self.query_one("#actions-panel").update_content,
                f"Running [{flow_label}]..."
            )

            self.agent.activate_context(session_id=self.session_id)
            try:
                if flow_label == "explore":
                    from infinidev.engine.tree_engine import TreeEngine
                    tree_engine = TreeEngine()
                    result = tree_engine.execute(
                        agent=self.agent,
                        task_prompt=task_prompt,
                        verbose=True,
                    )
                else:
                    result = self.engine.execute(
                        agent=self.agent,
                        task_prompt=task_prompt,
                        verbose=True,
                    )
                if not result or not result.strip():
                    result = "Done. (no additional output)"
            finally:
                self.agent.deactivate()

            # --- Code review phase (single pass, no retry loop) ---
            run_review = flow_config.run_review if flow_config else True
            if _settings.REVIEW_ENABLED and run_review and flow_label != "explore" and self.engine.has_file_changes():
                self.reviewer.reset()
                review = self.reviewer.review(
                    task_description=task_prompt[0],
                    developer_result=result,
                    file_changes_summary=self.engine.get_changed_files_summary(),
                    file_reasons=self.engine.get_file_change_reasons(),
                    file_contents=self.engine.get_file_contents(),
                    recent_messages=get_recent_summaries(self.session_id, limit=5),
                )

                if review.is_approved:
                    self.call_from_thread(
                        self.add_message,
                        "Reviewer",
                        f"Code review: APPROVED. {review.summary}",
                        "system",
                    )
                elif review.is_rejected:
                    self.call_from_thread(
                        self.add_message,
                        "Reviewer",
                        review.format_for_user(),
                        "system",
                    )
            # --- End code review phase ---

            self.call_from_thread(self._hide_thinking)
            self.call_from_thread(self.add_message, "Infinidev", result, "agent")
            store_conversation_turn(self.session_id, 'assistant', result, result[:200])
            self.call_from_thread(self._hide_thinking)
            self.call_from_thread(self.query_one("#actions-panel").update_content, "Idle")
        except Exception as e:
            self._analysis_waiting = False
            self.call_from_thread(self._hide_thinking)
            self.call_from_thread(self.add_message, "Error", str(e), "system")
        finally:
            self._engine_running = False
            self.call_from_thread(
                self._context_panel.set_flow, ""
            )
            self.call_from_thread(self._drain_pending_inputs)

    # ── Commands ─────────────────────────────────────────

    # ── Settings Helper Methods ───────────────────────────────

    def _get_settings_info(self):
        """Get formatted settings information."""
        from infinidev.config.settings import settings, SETTINGS_FILE

        lines = [
            f"[bold]Infinidev Settings (from {SETTINGS_FILE})[/bold]",
            "",
            "[bold]LLM[/bold]",
            f"  {settings.LLM_MODEL:<50} (LLM_MODEL)",
            f"  {settings.LLM_BASE_URL:<50} (LLM_BASE_URL)",
            "",
            "[bold]Loop Engine[/bold]",
            f"  {settings.LOOP_MAX_ITERATIONS:<50} (LOOP_MAX_ITERATIONS)",
            f"  {settings.LOOP_MAX_TOTAL_TOOL_CALLS:<50} (LOOP_MAX_TOTAL_TOOL_CALLS)",
            "",
            "[bold]Code Interpreter[/bold]",
            f"  {settings.CODE_INTERPRETER_TIMEOUT:<50} (CODE_INTERPRETER_TIMEOUT)",
            "",
            "[bold]Phases[/bold]",
            f"  {str(settings.ANALYSIS_ENABLED):<50} (ANALYSIS_ENABLED)",
            f"  {str(settings.REVIEW_ENABLED):<50} (REVIEW_ENABLED)",
            "",
            "[bold]Permissions[/bold]",
            f"  {settings.EXECUTE_COMMANDS_PERMISSION:<50} (EXECUTE_COMMANDS_PERMISSION)",
            f"  {str(settings.ALLOWED_COMMANDS_LIST):<50} (ALLOWED_COMMANDS_LIST)",
            f"  {settings.FILE_OPERATIONS_PERMISSION:<50} (FILE_OPERATIONS_PERMISSION)",
            f"  {str(settings.ALLOWED_FILE_PATHS):<50} (ALLOWED_FILE_PATHS)",
            "",
            "[bold]UI[/bold]",
            f"  {settings.model_dump().get('LOG_LEVEL', 'warning'):<50} (LOG_LEVEL)",
        ]
        return "\n".join(lines)

    def _convert_value_to_type(self, key: str, value: str):
        """Convert a string value to the appropriate type based on the setting key."""
        type_map = {
            "LLM_MODEL": str,
            "LLM_BASE_URL": str,
            "DB_PATH": str,
            "WORKSPACE_BASE_DIR": str,
            "EMBEDDING_PROVIDER": str,
            "EMBEDDING_MODEL": str,
            "EMBEDDING_BASE_URL": str,
            "FORGEJO_API_URL": str,
            "FORGEJO_OWNER": str,
            "LOOP_MAX_ITERATIONS": int,
            "LOOP_MAX_TOOL_CALLS_PER_ACTION": int,
            "LOOP_MAX_TOTAL_TOOL_CALLS": int,
            "LOOP_HISTORY_WINDOW": int,
            "MAX_RETRIES": int,
            "RETRY_BASE_DELAY": float,
            "COMMAND_TIMEOUT": int,
            "WEB_TIMEOUT": int,
            "GIT_PUSH_TIMEOUT": int,
            "MAX_FILE_SIZE_BYTES": int,
            "MAX_DIR_LISTING": int,
            "WEB_CACHE_TTL_SECONDS": int,
            "WEB_RPM_LIMIT": int,
            "WEB_ROBOTS_CACHE_TTL": int,
            "DEDUP_SIMILARITY_THRESHOLD": float,
            "CODE_INTERPRETER_TIMEOUT": int,
            "CODE_INTERPRETER_MAX_OUTPUT": int,
            "SANDBOX_ENABLED": bool,
            "ANALYSIS_ENABLED": bool,
            "REVIEW_ENABLED": bool,
            "EXECUTE_COMMANDS_PERMISSION": str,
            "ALLOWED_COMMANDS_LIST": list,
            "FILE_OPERATIONS_PERMISSION": str,
            "ALLOWED_FILE_PATHS": list,
        }
        type_class = type_map.get(key, str)
        if type_class == bool:
            return value.lower() in ("true", "1", "yes")
        if type_class == list:
            # Split comma-separated string into list, strip whitespace
            return [item.strip() for item in value.split(",") if item.strip()]
        try:
            return type_class(value)
        except ValueError:
            raise ValueError(f"Invalid value for {key}: {value}")

    def _handle_settings(self, parts: list[str]):
        """Handle the /settings command."""
        from infinidev.config.settings import settings, SETTINGS_FILE, reload_all
        import shutil

        subcmd = parts[1].lower() if len(parts) > 1 else "info"

        if subcmd == "reset":
            if SETTINGS_FILE.exists():
                SETTINGS_FILE.unlink()
            reload_all()
            self.add_message("System", "Settings reset to defaults. Reloaded.", "system")
        elif subcmd == "export" and len(parts) > 2:
            export_path = parts[2]
            try:
                shutil.copy(SETTINGS_FILE, export_path)
                self.add_message("System", f"Settings exported to: {export_path}", "system")
            except Exception as e:
                self.add_message("System", f"Export failed: {e}", "system")
        elif subcmd == "import" and len(parts) > 2:
            import_path = parts[2]
            try:
                if not SETTINGS_FILE.exists():
                    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(import_path, SETTINGS_FILE)
                reload_all()
                self.add_message("System", f"Settings imported from: {import_path}. Reloaded.", "system")
            except FileNotFoundError:
                self.add_message("System", f"Import failed: File not found: {import_path}", "system")
            except Exception as e:
                self.add_message("System", f"Import failed: {e}", "system")
        elif subcmd == "info" or subcmd == "" or len(parts) == 1:
            if len(parts) == 1:
                # No subcommand provided - open the interactive modal
                self._browse_settings()
            else:
                self.add_message("System", self._get_settings_info(), "system")
        else:
            # Show or set specific setting
            setting_key = subcmd.upper()
            value_to_set = parts[2] if len(parts) > 2 else None

            try:
                if value_to_set:
                    # Try to convert to appropriate type
                    converted = self._convert_value_to_type(setting_key, value_to_set)
                    settings.save_user_settings({setting_key: converted})
                    reload_all()
                    self.add_message("System", f"Updated {setting_key} to: {converted}", "system")
                else:
                    # Show current value
                    value = getattr(settings, setting_key, None)
                    if value is not None:
                        self.add_message("System", f"{setting_key}: {value}", "system")
                    else:
                        self.add_message("System", f"Unknown setting: {setting_key}", "system")
            except ValueError as e:
                self.add_message("System", f"Error: {e}", "system")

    def handle_command(self, cmd_text: str):
        parts = cmd_text.split()
        cmd = parts[0].lower()
        if cmd == "/exit" or cmd == "/quit":
            self.exit()
        elif cmd == "/clear":
            self.query_one("#chat-history").remove_children()
            self._log_lines.clear()
            self.query_one("#plan-panel").update_content("")
            self.query_one("#steps-panel").update_content("")
            self.query_one("#actions-panel").update_content("Idle")
            self.query_one("#logs-panel").update_content("")
        elif cmd == "/help":
            self.add_message(
                "System",
                "^E                   Toggle file explorer\n"
                "F2 / F3 / F4         Focus: Chat / Explorer / Sidebar\n"
                "^W                   Close file tab\n"
                "─────────────────────────────────────\n"
                "!ls                  Execute shell command\n"
                "!grep foo *.py       Run shell with piping\n"
                "/models              Show current model\n"
                "/models list         List available Ollama models\n"
                "/models set <name>   Change model\n"
                "/models manage       Pick a model interactively\n"
                "/settings            Show current settings\n"
                "/settings <key>      Show specific setting\n"
                "/settings <key> <val> Change setting\n"
                "/settings reset      Reset to defaults\n"
                "/settings export     Export settings to file\n"
                "/settings import     Import settings from file\n"
                "/explore <problem>   Decompose and explore a complex problem\n"
                "/init                Explore and document the current project\n"
                "/findings            Browse all findings\n"
                "/knowledge           Browse project knowledge\n"
                "/documentation       Browse cached library docs\n"
                "/clear               Clear chat\n"
                "/exit, /quit         Exit",
                "system",
            )
        elif cmd == "/settings":
            self._handle_settings(parts)
        elif cmd == "/models":
            from infinidev.config.settings import settings
            subcmd = parts[1].lower() if len(parts) > 1 else "info"
            if subcmd == "set" and len(parts) > 2:
                new_model = parts[2]
                if "/" not in new_model:
                    new_model = f"ollama_chat/{new_model}"
                settings.save_user_settings({"LLM_MODEL": new_model})
                from infinidev.config.settings import reload_all as _reload
                _reload()
                self.add_message("System", f"Model updated to: {settings.LLM_MODEL}", "system")
                self._update_status_bar()
            elif subcmd == "list":
                self._list_models()
            elif subcmd == "manage":
                self._manage_models()
            else:
                self.add_message("System", f"Current model: {settings.LLM_MODEL}\nBase URL: {settings.LLM_BASE_URL}", "system")
        elif cmd == "/findings":
            self._browse_findings(filter_type=None)
        elif cmd == "/knowledge":
            self._browse_findings(filter_type="project_context")
        elif cmd == "/documentation" or cmd == "/docs":
            self._browse_documentation()
        elif cmd == "/explore":
            problem = " ".join(parts[1:]) if len(parts) > 1 else ""
            if not problem:
                self.add_message("System", "Usage: /explore <problem description>", "system")
            elif self._engine_running:
                self.add_message("System", "Cannot run /explore while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", f"Exploring: {problem}", "system")
                self._show_thinking()
                self._run_explore(problem)
        elif cmd == "/brainstorm":
            problem = " ".join(parts[1:]) if len(parts) > 1 else ""
            if not problem:
                self.add_message("System", "Usage: /brainstorm <problem description>", "system")
            elif self._engine_running:
                self.add_message("System", "Cannot run /brainstorm while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", f"Brainstorming: {problem}", "system")
                self._show_thinking()
                self._run_brainstorm(problem)
        elif cmd == "/think":
            self._gather_next_task = True
            self.add_message("System", "Gather mode enabled for the next task. Send your prompt and infinidev will deeply analyze the codebase before acting.", "system")
        elif cmd == "/init":
            if self._engine_running:
                self.add_message("System", "Cannot run /init while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", "Exploring and documenting project...", "system")
                self._show_thinking()
                self._run_init()
        else:
            self.add_message("System", f"Unknown command: {cmd}", "system")

    # ── Background workers ───────────────────────────────

    @work(exclusive=True, thread=True)
    def _run_init(self):
        """Run /init — explore and document the current project."""
        from infinidev.prompts.init_project import INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT
        from infinidev.engine.flows import get_flow_config

        reload_all()
        self.call_from_thread(
            self._context_panel.set_flow, "init"
        )
        self.call_from_thread(
            self.query_one("#actions-panel").update_content,
            "Running [init]..."
        )

        flow_config = get_flow_config("document")
        self.agent._system_prompt_identity = flow_config.identity_prompt
        self.agent.backstory = flow_config.backstory

        self.agent.activate_context(session_id=self.session_id)
        try:
            result = self.engine.execute(
                agent=self.agent,
                task_prompt=(INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT),
                verbose=True,
            )
            if not result or not result.strip():
                result = "Project initialization complete."
        except Exception as e:
            result = f"Init failed: {e}"
        finally:
            self.agent.deactivate()

        self.call_from_thread(self._hide_thinking)
        self.call_from_thread(self.add_message, "Infinidev", result, "agent")
        store_conversation_turn(self.session_id, 'assistant', result, result[:200])
        self.call_from_thread(
            self.query_one("#actions-panel").update_content, "Idle"
        )
        self._engine_running = False
        self.call_from_thread(self._drain_pending_inputs)

    @work(exclusive=True, thread=True)
    def _run_explore(self, problem: str):
        """Run /explore — decompose and explore a complex problem."""
        from infinidev.engine.tree_engine import TreeEngine
        from infinidev.engine.flows import get_flow_config

        reload_all()
        self.call_from_thread(
            self._context_panel.set_flow, "explore"
        )
        self.call_from_thread(
            self.query_one("#actions-panel").update_content,
            "Running [explore]..."
        )

        flow_config = get_flow_config("explore")
        self.agent._system_prompt_identity = flow_config.identity_prompt
        self.agent.backstory = flow_config.backstory

        self.agent.activate_context(session_id=self.session_id)
        try:
            tree_engine = TreeEngine()
            result = tree_engine.execute(
                agent=self.agent,
                task_prompt=(problem, flow_config.expected_output),
                verbose=True,
            )
            if not result or not result.strip():
                result = "Exploration complete (no synthesis produced)."
        except Exception as e:
            result = f"Exploration failed: {e}"
        finally:
            self.agent.deactivate()

        self.call_from_thread(self._hide_thinking)
        self.call_from_thread(self.add_message, "Infinidev", result, "agent")
        store_conversation_turn(self.session_id, 'assistant', result, result[:200])
        self.call_from_thread(
            self.query_one("#actions-panel").update_content, "Idle"
        )
        self._engine_running = False
        self.call_from_thread(self._drain_pending_inputs)

    @work(exclusive=True, thread=True)
    def _run_brainstorm(self, problem: str):
        """Run /brainstorm — brainstorm ideas and solutions for a problem."""
        from infinidev.engine.tree_engine import TreeEngine
        from infinidev.engine.flows import get_flow_config

        reload_all()
        self.call_from_thread(
            self._context_panel.set_flow, "brainstorm"
        )
        self.call_from_thread(
            self.query_one("#actions-panel").update_content,
            "Running [brainstorm]..."
        )

        flow_config = get_flow_config("brainstorm")
        self.agent._system_prompt_identity = flow_config.identity_prompt
        self.agent.backstory = flow_config.backstory

        self.agent.activate_context(session_id=self.session_id)
        try:
            tree_engine = TreeEngine()
            result = tree_engine.execute(
                agent=self.agent,
                task_prompt=(problem, flow_config.expected_output),
                verbose=True,
            )
            if not result or not result.strip():
                result = "Brainstorm complete (no synthesis produced)."
        except Exception as e:
            result = f"Brainstorm failed: {e}"
        finally:
            self.agent.deactivate()

        self.call_from_thread(self._hide_thinking)
        self.call_from_thread(self.add_message, "Infinidev", result, "agent")
        store_conversation_turn(self.session_id, 'assistant', result, result[:200])
        self.call_from_thread(
            self.query_one("#actions-panel").update_content, "Idle"
        )
        self._engine_running = False
        self.call_from_thread(self._drain_pending_inputs)

    @work(thread=True)
    def _browse_findings(self, filter_type: str | None):
        from infinidev.db.service import get_all_findings
        findings = get_all_findings()
        if filter_type:
            findings = [f for f in findings if f["finding_type"] == filter_type]
        title = "Project Knowledge" if filter_type == "project_context" else "All Findings"
        self.call_from_thread(self.push_screen, FindingsBrowserScreen(findings, title=title))

    @work(thread=True)
    def _browse_settings(self):
        """Open the settings editor modal."""
        from infinidev.config.settings import settings as global_settings
        self.call_from_thread(self.push_screen, SettingsEditorScreen(global_settings))

    @work(thread=True)
    def _browse_documentation(self):
        from infinidev.db.service import execute_with_retry

        def _load(conn):
            rows = conn.execute(
                """\
                SELECT library_name, language, version, section_title, section_order, content
                FROM library_docs
                ORDER BY library_name, language, version, section_order
                """
            ).fetchall()
            return rows

        rows = execute_with_retry(_load) or []

        # Group by library
        libs: list[dict] = []
        sections: dict[str, list[dict]] = {}
        seen_keys: set[str] = set()

        for row in rows:
            key = f"{row['library_name']}|{row['language']}|{row['version']}"
            if key not in seen_keys:
                seen_keys.add(key)
                libs.append({
                    "key": key,
                    "library_name": row["library_name"],
                    "language": row["language"],
                    "version": row["version"],
                    "section_count": 0,
                })
            sections.setdefault(key, []).append({
                "section_title": row["section_title"],
                "section_order": row["section_order"],
                "content": row["content"],
            })

        for lib in libs:
            lib["section_count"] = len(sections.get(lib["key"], []))

        self.call_from_thread(self.push_screen, DocsBrowserScreen(libs, sections))

    @work(thread=True)
    def _list_models(self):
        import httpx
        from infinidev.config.settings import settings
        try:
            base_url = settings.LLM_BASE_URL.rstrip("/")
            resp = httpx.get(f"{base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                self.call_from_thread(self.add_message, "System", f"Error: HTTP {resp.status_code}", "system")
                return
            models = resp.json().get("models", [])
            if not models:
                self.call_from_thread(self.add_message, "System", "No models found.", "system")
                return
            current_tag = settings.LLM_MODEL.split("/", 1)[-1]
            lines = [f"  {'*' if m.get('name') == current_tag else ' '} {m.get('name')} ({m.get('size', 0) / (1024**3):.1f} GB)" for m in models]
            self.call_from_thread(self.add_message, "System", "Available models:\n" + "\n".join(lines), "system")
        except Exception as e:
            self.call_from_thread(self.add_message, "System", f"Could not connect to Ollama: {e}", "system")

    @work(thread=True)
    def _manage_models(self):
        import httpx
        from infinidev.config.settings import settings
        try:
            base_url = settings.LLM_BASE_URL.rstrip("/")
            resp = httpx.get(f"{base_url}/api/tags", timeout=10)
            if resp.status_code != 200:
                self.call_from_thread(self.add_message, "System", f"Error: HTTP {resp.status_code}", "system")
                return
            models = resp.json().get("models", [])
            if not models:
                self.call_from_thread(self.add_message, "System", "No models found.", "system")
                return
            current_tag = settings.LLM_MODEL.split("/", 1)[-1]
            self.call_from_thread(self._open_model_picker, models, current_tag)
        except Exception as e:
            self.call_from_thread(self.add_message, "System", f"Could not connect to Ollama: {e}", "system")

    def _open_model_picker(self, models: list[dict], current_tag: str) -> None:
        def on_dismiss(selected: str | None) -> None:
            if selected is None:
                return
            from infinidev.config.settings import settings, reload_all as _reload
            new_model = f"ollama_chat/{selected}"
            settings.save_user_settings({"LLM_MODEL": new_model})
            _reload()
            self.add_message("System", f"Model changed to: {new_model}", "system")
            self._update_status_bar()
            self.query_one("#chat-input", ChatInput).focus()
        self.push_screen(ModelPickerScreen(models, current_tag), callback=on_dismiss)


if __name__ == "__main__":
    app = InfinidevTUI()
    app.run()
