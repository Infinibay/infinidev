"""TUI implementation for Infinidev using Textual."""

import logging
import os
import pathlib
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
    ProgressBar, Button, Select,
)
from textual.widgets.option_list import Option
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from textual import on, events, work

# Infinidev imports
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop_engine import LoopEngine, set_event_callback
from infinidev.engine.analysis_engine import AnalysisEngine
from infinidev.engine.review_engine import ReviewEngine
from infinidev.db.service import (
    init_db, store_conversation_turn, get_recent_summaries,
)
from infinidev.config.settings import reload_all, Settings
from infinidev.cli.file_watcher import FileWatcher
import infinidev.prompts.flows  # noqa: F401 — registers flows
from infinidev.ui.widgets.context_widgets import QueuedMessageWidget, QueuedMessageStatus
from infinidev.ui.widgets.file_diff_widget import FileChangeDiffWidget, colorize_diff

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

    CSS = """
    ContextPanel {
        width: 100%;
        margin-bottom: 1;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $primary;
    }
    #context-model-name {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    .ctx-section-title {
        text-style: bold;
        margin-top: 1;
        color: $text;
    }
    .ctx-detail {
        margin-left: 2;
        color: $text;
    }
    .ctx-bar {
        margin-left: 2;
    }
    """

    def __init__(self, id: str = None):
        super().__init__(id=id)
        self._status: dict[str, Any] = {}

    def compose(self) -> ComposeResult:
        yield Static("Loading...", id="context-model-name")
        # Chat section — last prompt_tokens from LLM
        yield Static("[bold]Chat Usage[/bold]", classes="ctx-section-title")
        yield Static("", id="chat-details", classes="ctx-detail")
        yield Static("", id="chat-bar", classes="ctx-bar")
        # Task section — last prompt tokens during task execution
        yield Static("[bold]Task Usage[/bold]", classes="ctx-section-title")
        yield Static("", id="task-details", classes="ctx-detail")
        yield Static("", id="task-bar", classes="ctx-bar")

    def update_status(self, status: dict[str, Any]) -> None:
        """Update the panel with new status data."""
        self._status = status
        max_ctx = status.get("max_context", 4096)

        # Model name
        model = status.get("model", "unknown")
        self.query_one("#context-model-name", Static).update(
            f"[bold]{model}[/bold]  [dim]({max_ctx} ctx)[/dim]"
        )

        # Chat window - show last prompt tokens instead of cumulative
        chat = status.get("chat", {})
        self._update_section(
            "#chat-details", "#chat-bar",
            chat.get("current_tokens", 0),
            chat.get("remaining_tokens", max_ctx),
            chat.get("usage_percentage", 0.0),
        )

        # Task window - show cumulative tokens
        tasks = status.get("tasks", {})
        self._update_section(
            "#task-details", "#task-bar",
            tasks.get("current_tokens", 0),
            tasks.get("remaining_tokens", max_ctx),
            tasks.get("usage_percentage", 0.0),
        )

    def _update_section(self, details_id: str, bar_id: str,
                        used: int, available: int, pct: float) -> None:
        """Update one context section (chat or task)."""
        pct_val = min(pct, 1.0)
        pct_str = f"{pct_val * 100:.1f}%"

        # Color based on usage
        if pct_val > 0.8:
            color = "red"
        elif pct_val > 0.5:
            color = "yellow"
        else:
            color = "green"

        self.query_one(details_id, Static).update(
            f"Used: [bold]{used}[/bold]  "
            f"Available: [bold {color}]{available}[/bold {color}]  "
            f"({pct_str})"
        )

        # Progress bar
        bar_width = 20
        filled = int(bar_width * pct_val)
        empty = bar_width - filled
        bar = "\\[" + "█" * filled + "░" * empty + "\\]"
        self.query_one(bar_id, Static).update(
            f"[bold {color}]{bar}[/bold {color}] {pct_str}"
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

    CSS = """
    ModelPickerScreen { align: center middle; }
    #model-picker-box { width: 60; max-height: 80%; background: $surface; border: tall $primary; padding: 1 2; }
    #model-picker-title { text-align: center; text-style: bold; margin-bottom: 1; }
    #model-picker-list { height: 1fr; max-height: 20; }
    #model-picker-hint { text-align: center; color: $text-muted; margin-top: 1; }
    """
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
    """Modal to view and edit all Infinidev settings."""

    CSS = """
    SettingsEditorScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.65);
    }
    #settings-box {
        width: 80%;
        height: 80%;
        background: $surface;
        border: round $primary;
        padding: 0;
    }
    #settings-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 2;
        background: $surface-darken-2;
        border-bottom: solid $primary-darken-2;
    }
    #settings-list {
        height: 1fr;
        background: $surface;
        padding: 0 1;
        scrollbar-size-vertical: 1;
        scrollbar-color: $primary-darken-2;
        scrollbar-color-hover: $primary;
        scrollbar-color-active: $primary;
    }
    #settings-list > .option-list--option-highlighted {
        background: $primary 20%;
    }
    #settings-footer {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 2;
        background: $surface-darken-2;
        border-top: solid $primary-darken-2;
    }
    #settings-footer Horizontal {
        width: auto;
        height: auto;
        align: center middle;
    }
    #settings-footer Button {
        margin: 0 1;
    }
    #settings-hint {
        width: 100%;
        text-align: center;
        color: $text-muted;
        margin-top: 1;
        height: auto;
    }
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("ctrl+enter", "save", "Save", show=True),
    ]

    # Define all settings with their metadata.
    # "type" controls the editor widget: "string"=text, "int"/"float"=text,
    # "bool"=select(true/false), "select"=select from "choices", "list"=text(comma-sep).
    SETTINGS_INFO = {
        "LLM_MODEL": {"section": "LLM", "default": "ollama_chat/qwen2.5-coder:7b", "type": "string",
                       "desc": "LiteLLM model identifier. Format: provider/model (e.g. ollama_chat/qwen2.5-coder:7b)."},
        "LLM_BASE_URL": {"section": "LLM", "default": "http://localhost:11434", "type": "string",
                         "desc": "Base URL for the LLM provider (Ollama, OpenAI-compatible, etc)."},
        "LLM_API_KEY": {"section": "LLM", "default": "ollama", "type": "string",
                        "desc": "API key for the LLM provider. Use 'ollama' for local Ollama."},
        "EMBEDDING_PROVIDER": {"section": "Embedding", "default": "ollama", "type": "string",
                               "desc": "Provider for text embeddings used in semantic search."},
        "EMBEDDING_MODEL": {"section": "Embedding", "default": "nomic-embed-text", "type": "string",
                            "desc": "Embedding model name (must be available in the provider)."},
        "EMBEDDING_BASE_URL": {"section": "Embedding", "default": "http://localhost:11434", "type": "string",
                               "desc": "Base URL for the embedding provider."},
        "LOOP_MAX_ITERATIONS": {"section": "Loop Engine", "default": 50, "type": "int",
                                "desc": "Maximum plan-execute-summarize iterations per task."},
        "LOOP_MAX_TOOL_CALLS_PER_ACTION": {"section": "Loop Engine", "default": 0, "type": "int",
                                           "desc": "Max tool calls per step. 0 = unlimited (only global limit applies)."},
        "LOOP_MAX_TOTAL_TOOL_CALLS": {"section": "Loop Engine", "default": 200, "type": "int",
                                      "desc": "Hard limit on total tool calls per task across all steps."},
        "LOOP_HISTORY_WINDOW": {"section": "Loop Engine", "default": 0, "type": "int",
                                "desc": "Number of past action summaries to keep in context. 0 = keep all."},
        "ANALYSIS_ENABLED": {"section": "Phases", "default": True, "type": "bool",
                             "desc": "Enable the pre-development analyst phase that explores code and produces a spec."},
        "REVIEW_ENABLED": {"section": "Phases", "default": True, "type": "bool",
                           "desc": "Enable the post-development code review phase."},
        "EXECUTE_COMMANDS_PERMISSION": {"section": "Permissions", "default": "auto_approve", "type": "select",
                                        "choices": ["auto_approve", "ask", "allowed_list"],
                                        "desc": "How to handle shell command execution. auto_approve=allow all, ask=prompt user, allowed_list=only commands in ALLOWED_COMMANDS_LIST."},
        "ALLOWED_COMMANDS_LIST": {"section": "Permissions", "default": [], "type": "list",
                                  "desc": "Commands allowed when EXECUTE_COMMANDS_PERMISSION=allowed_list. Comma-separated."},
        "FILE_OPERATIONS_PERMISSION": {"section": "Permissions", "default": "ask", "type": "select",
                                       "choices": ["ask", "auto_approve", "allowed_paths"],
                                       "desc": "How to handle file write/edit. ask=prompt user, auto_approve=allow all, allowed_paths=only within listed paths."},
        "ALLOWED_FILE_PATHS": {"section": "Permissions", "default": [], "type": "list",
                               "desc": "Paths allowed when FILE_OPERATIONS_PERMISSION=allowed_paths. Comma-separated."},
        "SANDBOX_ENABLED": {"section": "Sandbox", "default": False, "type": "bool",
                            "desc": "Enable sandbox mode. Restricts file access to ALLOWED_BASE_DIRS."},
        "ALLOWED_BASE_DIRS": {"section": "Sandbox", "default": ["/"], "type": "list",
                              "desc": "Directories the agent can access when sandbox is enabled. Comma-separated."},
        "ALLOWED_COMMANDS": {"section": "Sandbox", "default": [], "type": "list",
                             "desc": "Legacy: shell commands allowed in sandbox mode. Comma-separated."},
        "COMMAND_TIMEOUT": {"section": "Timeouts", "default": 120, "type": "int",
                            "desc": "Max seconds for a shell command before it is killed."},
        "WEB_TIMEOUT": {"section": "Timeouts", "default": 30, "type": "int",
                        "desc": "Max seconds for web fetch/search requests."},
        "GIT_PUSH_TIMEOUT": {"section": "Timeouts", "default": 120, "type": "int",
                             "desc": "Max seconds for git push operations."},
        "MAX_FILE_SIZE_BYTES": {"section": "File Limits", "default": 5242880, "type": "int",
                                "desc": "Max file size (bytes) the agent can read. Default 5 MB."},
        "MAX_DIR_LISTING": {"section": "File Limits", "default": 1000, "type": "int",
                            "desc": "Max entries returned by list_directory."},
        "DB_PATH": {"section": "Database", "default": "~/.infinidev/infinidev.db", "type": "string",
                    "desc": "Path to the SQLite database for projects, tasks, and findings."},
        "MAX_RETRIES": {"section": "Database", "default": 5, "type": "int",
                        "desc": "Max retries for database operations on WAL contention."},
        "RETRY_BASE_DELAY": {"section": "Database", "default": 0.1, "type": "float",
                             "desc": "Base delay (seconds) for exponential backoff on DB retries."},
        "WEB_CACHE_TTL_SECONDS": {"section": "Web Tools", "default": 3600, "type": "int",
                                  "desc": "Cache duration (seconds) for web search/fetch results."},
        "WEB_RPM_LIMIT": {"section": "Web Tools", "default": 20, "type": "int",
                          "desc": "Max web requests per minute (rate limiting)."},
        "WEB_ROBOTS_CACHE_TTL": {"section": "Web Tools", "default": 3600, "type": "int",
                                 "desc": "Cache duration (seconds) for robots.txt lookups."},
        "DEDUP_SIMILARITY_THRESHOLD": {"section": "Knowledge", "default": 0.82, "type": "float",
                                       "desc": "Cosine similarity threshold for deduplicating findings (0-1)."},
        "WORKSPACE_BASE_DIR": {"section": "Workspace", "default": ".", "type": "string",
                               "desc": "Base directory for the agent's workspace."},
        "CODE_INTERPRETER_TIMEOUT": {"section": "Code Interpreter", "default": 120, "type": "int",
                                     "desc": "Max seconds for code interpreter execution."},
        "CODE_INTERPRETER_MAX_OUTPUT": {"section": "Code Interpreter", "default": 50000, "type": "int",
                                        "desc": "Max characters of output captured from code interpreter."},
    }

    def __init__(self, settings: Settings):
        super().__init__()
        self._settings = settings
        self._edited_values: dict[str, str] = {}

    def compose(self) -> ComposeResult:
        with Vertical(id="settings-box"):
            yield Label("Settings", id="settings-title")
            yield OptionList(id="settings-list")
            with Vertical(id="settings-footer"):
                with Horizontal():
                    yield Button("Save", variant="success", id="btn-save")
                    yield Button("Cancel", variant="error", id="btn-cancel")
                yield Label("[dim]↑↓ navigate   Enter edit[/dim]", id="settings-hint")

    def on_mount(self) -> None:
        ol = self.query_one("#settings-list", OptionList)

        # Group settings by section
        sections: dict[str, list[tuple[str, dict]]] = {}
        for key, info in self.SETTINGS_INFO.items():
            section = info["section"]
            if section not in sections:
                sections[section] = []
            sections[section].append((key, info))

        # Add section headers and settings
        first = True
        for section_name in sorted(sections.keys()):
            # Blank line separator between sections (except before first)
            if not first:
                ol.add_option(Option("", id=f"__blank__{section_name}"))
            first = False
            # Section header with line decoration
            header = f"[bold cyan]── {section_name} ──[/bold cyan]"
            ol.add_option(Option(header, id=f"__section__{section_name}"))

            for key, info in sorted(sections[section_name], key=lambda x: x[0]):
                current_value = getattr(self._settings, key, info.get("default", ""))
                display = str(current_value) if current_value not in (None, "", []) else "[italic dim]not set[/italic dim]"
                value_markup = f"[dim]{display}[/dim]" if current_value not in (None, "", []) else display
                ol.add_option(
                    Option(
                        f"  [bold]{key}[/bold]  {value_markup}",
                        id=key,
                    )
                )

        ol.focus()

    def _is_non_setting(self, option_id: str | None) -> bool:
        """Check if an option is a section header or blank separator."""
        if not option_id:
            return True
        return option_id.startswith("__section__") or option_id.startswith("__blank__")

    @on(OptionList.OptionHighlighted, "#settings-list")
    def on_highlight(self, event: OptionList.OptionHighlighted) -> None:
        if self._is_non_setting(event.option.id):
            return

    @on(OptionList.OptionSelected, "#settings-list")
    def on_select(self, event: OptionList.OptionSelected) -> None:
        if self._is_non_setting(event.option.id):
            return
        self._show_setting_editor(event.option.id)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-save":
            self.action_save()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def _show_setting_editor(self, setting_key: str) -> None:
        info = self.SETTINGS_INFO.get(setting_key, {})
        current_value = getattr(self._settings, setting_key, None)
        setting_type = info.get("type", "string")
        description = info.get("desc", "")
        choices = info.get("choices", None)

        editor = SettingValueEditor(
            setting_key, current_value, setting_type,
            description=description, choices=choices,
        )
        self.app.push_screen(editor, lambda value: self._on_value_updated(setting_key, value))

    def _on_value_updated(self, key: str, value: str | None) -> None:
        if value is not None:
            # Convert list-type settings from comma-separated string
            setting_type = self.SETTINGS_INFO.get(key, {}).get("type", "string")
            if setting_type == "list" and isinstance(value, str):
                value = [item.strip() for item in value.split(",") if item.strip()]
            self._edited_values[key] = value
            # Update the option label to show edited value
            ol = self.query_one("#settings-list", OptionList)
            display_value = self._format_value(value, self.SETTINGS_INFO.get(key, {}).get("type", "string"))
            for i in range(ol.option_count):
                opt = ol.get_option_at_index(i)
                if opt.id == key:
                    ol.replace_option_prompt(opt.id, f"  [bold white]{key}[/bold white]  [bold green]{display_value}[/bold green] [dim italic](edited)[/dim italic]")
                    break
        self.query_one("#settings-list", OptionList).focus()

    def _format_value(self, value: str, value_type: str) -> str:
        """Format value for display based on its type."""
        try:
            if value_type == "int":
                return str(int(value))
            elif value_type == "float":
                return str(float(value))
            elif value_type == "bool":
                return "true" if value.lower() in ("true", "1", "yes") else "false"
            return value
        except (ValueError, AttributeError):
            return value

    def action_save(self) -> None:
        """Save all edited settings."""
        if not self._edited_values:
            self.dismiss(None)
            return
        
        self.query_one("#settings-list", OptionList).blur()
        
        from infinidev.config.settings import reload_all
        try:
            self._settings.save_user_settings(self._edited_values)
            reload_all()
            self.add_message("System", f"Saved {len(self._edited_values)} settings. Reloaded.", "system")
            self.dismiss(None)
        except Exception as e:
            self.add_message("System", f"Failed to save settings: {e}", "system")
            self.dismiss(None)

    def action_cancel(self) -> None:
        """Cancel and discard changes."""
        self.dismiss(None)

    def add_message(self, sender: str, text: str, type: str = "agent") -> None:
        """Show a transient notification for save status."""
        self.notify(f"{sender}: {text}", timeout=3)


class PermissionDetailScreen(ModalScreen[None]):
    """Modal to view full permission details with syntax highlighting."""

    CSS = """
    PermissionDetailScreen {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #perm-detail-box {
        width: 80%;
        height: 80%;
        background: $surface;
        border: round $warning;
        padding: 0;
    }
    #perm-detail-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $warning;
        padding: 1 2;
        background: $surface-darken-2;
        border-bottom: solid $warning-darken-2;
    }
    #perm-detail-code {
        height: 1fr;
        margin: 1 2;
    }
    #perm-detail-hint {
        width: 100%;
        text-align: center;
        color: $text-muted;
        padding: 1 2;
        background: $surface-darken-2;
        border-top: solid $warning-darken-2;
    }
    """
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

    CSS = """
    SettingValueEditor {
        align: center middle;
        background: rgba(0, 0, 0, 0.7);
    }
    #setting-editor-box {
        width: 60%;
        max-height: 50%;
        background: $surface;
        border: round $primary;
        padding: 0;
    }
    #setting-editor-title {
        width: 100%;
        text-align: center;
        text-style: bold;
        color: $primary;
        padding: 1 2;
        background: $surface-darken-2;
        border-bottom: solid $primary-darken-2;
    }
    #setting-editor-desc {
        width: 100%;
        padding: 1 2;
        color: $text-muted;
    }
    #setting-editor-input {
        width: 100%;
        height: auto;
        max-height: 8;
        margin: 0 2 1 2;
    }
    #setting-editor-select {
        width: 100%;
        margin: 0 2 1 2;
    }
    #setting-editor-footer {
        width: 100%;
        height: auto;
        align: center middle;
        padding: 1 2;
        background: $surface-darken-2;
        border-top: solid $primary-darken-2;
    }
    #setting-editor-footer Horizontal {
        width: auto;
        height: auto;
        align: center middle;
    }
    #setting-editor-footer Button {
        margin: 0 1;
    }
    """
    BINDINGS = [
        Binding("escape", "cancel", "Cancel", show=True),
    ]

    def __init__(self, key: str, current_value, value_type: str,
                 description: str = "", choices: list[str] | None = None):
        super().__init__()
        self._key = key
        self._current_value = current_value
        self._value_type = value_type
        self._description = description
        self._choices = choices
        self._new_value = str(current_value)

    def compose(self) -> ComposeResult:
        from textual.widgets import Select
        with Vertical(id="setting-editor-box"):
            yield Label(f"Edit: {self._key}", id="setting-editor-title")
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

    CSS = """
    FindingsBrowserScreen { align: center middle; }
    #findings-box { width: 90%; height: 85%; background: $surface; border: tall $primary; padding: 1 2; }
    #findings-title { text-align: center; text-style: bold; margin-bottom: 1; }
    #findings-body { height: 1fr; }
    #findings-list { width: 40%; height: 100%; }
    #findings-detail { width: 60%; height: 100%; border-left: tall $primary; padding: 0 1; overflow-y: auto; }
    #findings-hint { text-align: center; color: $text-muted; margin-top: 1; }
    """
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

    CSS = """
    DocsBrowserScreen { align: center middle; }
    #docs-box { width: 90%; height: 85%; background: $surface; border: tall $primary; padding: 1 2; }
    #docs-title { text-align: center; text-style: bold; margin-bottom: 1; }
    #docs-body { height: 1fr; }
    #docs-lib-list { width: 30%; height: 100%; }
    #docs-section-list { width: 25%; height: 100%; border-left: tall $primary; }
    #docs-content { width: 45%; height: 100%; border-left: tall $primary; padding: 0 1; overflow-y: auto; }
    #docs-hint { text-align: center; color: $text-muted; margin-top: 1; }
    """
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


# ── Main TUI ────────────────────────────────────────────────────────────


class InfinidevTUI(App):
    """The main Infinidev TUI application."""

    CSS = """
    Screen { background: $surface; }

    /* ── Layout ─────────────────────────────────────────── */
    #main-container { height: 100%; }

    /* ── Context Panel ─────────────────────────────────── */
    #context-panel {
        width: 100%;
        padding: 1 2;
        background: $surface-darken-2;
        border: solid $secondary;
        margin-bottom: 1;
    }
    #context-panel.sidebar-title {
        text-align: center;
        margin-bottom: 1;
        text-style: bold;
    }
    #context-model-row {
        align: center middle;
        margin-bottom: 0;
    }
    #context-model-label {
        text-align: right;
        padding-left: 1;
    }
    #chat-progress-row, #tasks-progress-row {
        align: center middle;
        margin: 0;
    }
    #chat-progress, #tasks-progress {
        width: 1fr;
        margin-left: 1;
    }
    .context-label {
        width: auto;
        padding-right: 1;
    }
    .context-separator {
        height: 1;
        border: solid $secondary;
        margin: 1 0;
        opacity: 0.3;
    }
    #queued-messages-label {
        text-align: center;
        color: $text-muted;
        margin-top: 1;
    }
    .queued-message {
        background: $surface-darken-1;
        border: solid $warning;
        padding: 0 1;
        margin-top: 1;
        opacity: 0.6;
        text-style: dim;
    }

    /* ── Layout ─────────────────────────────────────────── */
    #explorer {
        width: 25;
        height: 100%;
        display: none;
        border-right: tall $primary;
        background: $surface-darken-1;
    }
    #explorer.-visible { display: block; }
    #explorer-title {
        background: $primary; color: white; padding: 0 1; text-align: center;
    }
    #file-tree { height: 1fr; }

    #content-area { width: 1fr; height: 100%; }

    /* Tabs styling */
    #content-tabs { height: 100%; }
    TabbedContent > ContentSwitcher { height: 1fr; }
    TabPane { height: 100%; }

    /* Chat tab */
    #chat-pane { height: 100%; }
    #chat-history { height: 1fr; border-bottom: solid $primary; padding: 1; overflow-y: scroll; }
    #autocomplete-menu { display: none; height: auto; max-height: 8; border-top: solid $accent; background: $surface-lighten-1; }
    #autocomplete-menu.-visible { display: block; }
    ChatInput { height: 4; border: solid $primary; }

    /* File editor tab */
    .file-editor { height: 100%; }

    /* Sidebar */
    #sidebar {
        width: 30%;
        height: 100%;
        border-left: tall $primary;
        background: $surface-darken-1;
        padding: 1;
    }
    .sidebar-title { background: $primary; color: white; padding: 0 1; margin-bottom: 0; }
    .sidebar-content { margin-bottom: 1; padding: 1; background: $surface-lighten-1; height: auto; max-height: 10; overflow-y: scroll; color: $text; }

    /* ── Chat messages ───────────────────────────────── */
    .user-msg { height: auto; color: #a8ffc8; margin-bottom: 1; background: #0a1a0a; padding: 0 1; border-left: tall #2df97f; }
    .pending-msg { height: auto; color: #7aad7a; margin-bottom: 1; background: #0a1a0a; padding: 0 1; border-left: dashed #2a8a4f; opacity: 80%; }
    .agent-msg { height: auto; color: #d0e4ff; margin-bottom: 1; background: #0a101a; padding: 0 1; border-left: tall #4da6ff; }
    .agent-msg Markdown { margin: 0; padding: 0; background: transparent; }
    .agent-msg MarkdownFence { margin: 1 0; max-height: 20; overflow-y: auto; }
    .agent-msg MarkdownH1, .agent-msg MarkdownH2, .agent-msg MarkdownH3 { margin: 1 0 0 0; padding: 0; background: transparent; border: none; }
    .system-msg { height: auto; color: #ffcc4d; text-style: italic; margin-bottom: 1; padding: 0 1; border-left: tall #ffaa00; background: #1a1500; }

    /* Status bar */
    #status-bar { height: 1; background: $surface-darken-1; color: $text-muted; padding: 0 2; dock: bottom; }

    /* Thinking indicator */
    .thinking-indicator { height: auto; margin: 1 0; color: #7b9fdf; align: center middle; }
    .thinking-indicator Static { text-align: center; width: 100%; }
    .thinking-indicator LoadingIndicator { height: 1; }

    /* ── Context Windows ────────────────────────────────────── */
    #context-panel {
        width: 100%;
        margin-bottom: 1;
        padding: 1;
        background: $surface-lighten-1;
        border: solid $primary;
    }
    #context-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
        color: $primary;
    }
    .context-label {
        width: auto;
        text-style: bold;
        color: $text;
    }
    .context-value {
        width: auto;
        color: $text-muted;
    }
    .context-usage {
        width: 1fr;
    }
    .context-usage-text {
        width: auto;
        margin-left: 1;
    }
    .context-remaining {
        width: auto;
        color: $success;
    }
    .context-remaining.low {
        color: $error;
    }

    /* Queued messages styling */
    .queued-msg {
        height: auto;
        color: $text-muted;
        margin-bottom: 1;
        padding: 0 1;
        border-left: tall $accent;
        background: $surface-darken-2;
        opacity: 0.6;
    }
    .queued-msg .message-content {
        color: $text-muted;
    }
    .queued-msg .message-status {
        color: $warning;
        text-style: italic;
    }
    .queued-msg.reading {
        opacity: 1.0;
        border-left: tall $primary;
        background: $surface;
    }
    .queued-msg.processed {
        opacity: 1.0;
        border-left: tall $success;
        background: #0a1a0a;
    }

    /* Permission buttons */
    #perm-btn-row {
        height: auto;
        width: 100%;
        padding: 1 0;
    }
    #perm-btn-row Button {
        margin: 0 1;
        min-width: 10;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Exit", show=True),
        Binding("ctrl+e", "toggle_explorer", "Explorer", show=True, priority=True),
        Binding("ctrl+w", "close_tab", "Close tab", show=True, priority=True),
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
                yield Label("EXPLORER", id="explorer-title")
                yield DirectoryTree(os.getcwd(), id="file-tree")

            # Content area with tabs
            with Vertical(id="content-area"):
                with TabbedContent(id="content-tabs"):
                    with TabPane("Chat", id="chat-pane"):
                        yield VerticalScroll(id="chat-history")
                        yield OptionList(id="autocomplete-menu")
                        yield ChatInput(id="chat-input")

            # Sidebar
            with Vertical(id="sidebar"):
                # Context window panel
                yield ContextPanel(id="context-panel")
                yield SidebarPanel("PLANNING", id="plan-panel")
                yield SidebarPanel("STEPS", id="steps-panel")
                yield SidebarPanel("ACTIONS", id="actions-panel")
                yield SidebarPanel("LOGS", id="logs-panel")
        yield Static("", id="status-bar")
        yield Footer()

    # ── Lifecycle ─────────────────────────────────────────

    def on_mount(self) -> None:
        init_db()
        set_event_callback(self.on_loop_event)
        self.query_one("#chat-input").focus()
        self.add_message("System", "Welcome to Infinidev! Type your instruction or /help.", "system")

        self.session_id = str(uuid.uuid4())
        self._log_lines: list[str] = []
        self._open_files: dict[str, str] = {}  # tab_id → file_path
        self._file_diff_widgets: dict[str, FileChangeDiffWidget] = {}  # path → widget

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

        # Initialize file watcher
        self._file_watcher: Optional[FileWatcher] = None
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
        if active in self._open_files:
            del self._open_files[active]
        try:
            await tabs.remove_pane(active)
        except Exception:
            pass
        tabs.active = "chat-pane"
        self.query_one("#chat-input", ChatInput).focus()

    # ── File watcher integration ────────────────────────

    def _start_file_watcher(self):
        """Initialize and start the file watcher."""
        workspace = os.getcwd()
        self._file_watcher = FileWatcher(
            workspace=workspace,
            callback=self._on_file_change,
            visible_paths_callback=self._get_visible_paths
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
        """Clean up file watcher on app exit."""
        if self._file_watcher and self._file_watcher.is_running():
            self._file_watcher.stop()
            logger = logging.getLogger(__name__)
            logger.info("File watcher stopped on app exit")

    # ── File explorer events ─────────────────────────────

    @on(DirectoryTree.FileSelected)
    async def open_file(self, event: DirectoryTree.FileSelected) -> None:
        """Open a file in a new editor tab."""
        file_path = str(event.path)
        tab_id = f"file-{hash(file_path) & 0xFFFFFFFF:08x}"

        # If already open, just switch to it
        if tab_id in self._open_files:
            self.query_one("#content-tabs", TabbedContent).active = tab_id
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

        # Truncate display name
        rel = os.path.relpath(file_path)
        tab_name = rel if len(rel) < 30 else f".../{pathlib.Path(file_path).name}"

        # Create the tab and wait for it to be mounted
        tabs = self.query_one("#content-tabs", TabbedContent)
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
        tabs.active = tab_id
        editor.focus()

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

        elif event_type == "loop_log":
            level = data.get("level", "warning")
            msg = data.get("message", "")
            icon = "⚠" if level == "warning" else "✗"
            line = f"{icon} {msg}"
            self._log_lines.append(line)
            self._log_lines = self._log_lines[-10:]
            self.query_one("#logs-panel").update_content("\n".join(self._log_lines))
            self.set_timer(10.0, lambda l=line: self._expire_log_line(l))

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
            return Static(plain)

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

        # Track queued messages
        if not hasattr(self, '_queued_messages'):
            self._queued_messages: list[QueuedMessageWidget] = []

        widget = QueuedMessageWidget(
            sender=sender,
            text=text,
            sender_color=self._SENDER_COLORS.get(type, "#cccccc"),
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
        if hasattr(self, '_queued_messages') and widget in self._queued_messages:
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
                self.call_from_thread(
                    self.query_one("#actions-panel").update_content,
                    "Analyzing request..."
                )

                analysis_input = user_input
                analysis = self.analyst.analyze(
                    analysis_input,
                    session_summaries=summaries,
                    event_callback=self.on_loop_event if hasattr(self, 'on_loop_event') else None,
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

            # --- Development phase ---
            flow_label = analysis.flow if _settings.ANALYSIS_ENABLED else "develop"
            self.call_from_thread(
                self.query_one("#actions-panel").update_content,
                f"Running [{flow_label}]..."
            )

            self.agent.activate_context(session_id=self.session_id)
            try:
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
            if _settings.REVIEW_ENABLED and run_review and self.engine.has_file_changes():
                self.call_from_thread(
                    self.query_one("#actions-panel").update_content,
                    "Code review..."
                )

                self.reviewer.reset()
                review = self.reviewer.review(
                    task_description=task_prompt[0],
                    developer_result=result,
                    file_changes_summary=self.engine.get_changed_files_summary(),
                    event_callback=self.on_loop_event if hasattr(self, 'on_loop_event') else None,
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
