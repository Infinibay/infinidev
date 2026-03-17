"""TUI implementation for Infinidev using Textual."""

import logging
import os
import pathlib
import uuid
from typing import Any

logger = logging.getLogger(__name__)
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from typing import Optional, Callable, Set
from textual.widgets import (
    Header, Footer, Static, TextArea, Label, OptionList, Markdown,
    DirectoryTree, TabbedContent, TabPane, LoadingIndicator,
    ProgressBar,
)
from textual.widgets.option_list import Option
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from textual import on, events, work

# Infinidev imports
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop_engine import LoopEngine, set_event_callback
from infinidev.db.service import (
    init_db, store_conversation_turn, get_recent_summaries,
)
from infinidev.config.settings import reload_all
from infinidev.cli.file_watcher import FileWatcher
from infinidev.ui.widgets.context_widgets import QueuedMessageWidget, QueuedMessageStatus

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
    ("/findings", "Browse all findings"),
    ("/knowledge", "Browse project knowledge"),
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
        # Chat section
        yield Static("[bold]Chat[/bold]", classes="ctx-section-title")
        yield Static("", id="chat-details", classes="ctx-detail")
        yield Static("", id="chat-bar", classes="ctx-bar")
        # Task section
        yield Static("[bold]Task[/bold]", classes="ctx-section-title")
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

        # Chat window
        chat = status.get("chat", {})
        self._update_section(
            "#chat-details", "#chat-bar",
            chat.get("current_tokens", 0),
            chat.get("remaining_tokens", max_ctx),
            chat.get("usage_percentage", 0.0),
        )

        # Task window
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
    .agent-msg { height: auto; color: #d0e4ff; margin-bottom: 1; background: #0a101a; padding: 0 1; border-left: tall #4da6ff; }
    .agent-msg Markdown { margin: 0; padding: 0; background: transparent; }
    .agent-msg MarkdownFence { margin: 1 0; max-height: 20; overflow-y: auto; }
    .agent-msg MarkdownH1, .agent-msg MarkdownH2, .agent-msg MarkdownH3 { margin: 1 0 0 0; padding: 0; background: transparent; border: none; }
    .system-msg { height: auto; color: #ffcc4d; text-style: italic; margin-bottom: 1; padding: 0 1; border-left: tall #ffaa00; background: #1a1500; }

    /* Status bar */
    #status-bar { height: 1; background: $surface-darken-1; color: $text-muted; padding: 0 2; dock: bottom; }

    /* Thinking indicator */
    #thinking-indicator { height: auto; margin: 1 0; color: #7b9fdf; align: center middle; }
    #thinking-indicator Static { text-align: center; width: 100%; }
    #thinking-indicator LoadingIndicator { height: 1; }

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

        self.engine = LoopEngine()
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

    def _refresh_context_panel(self, task_tokens: int = 0) -> None:
        """Recalculate and refresh the context panel display."""
        # Collect chat text from message history
        history = self.query_one("#chat-history")
        chat_text = ""
        for msg in history.query(Static):
            content = str(msg._Static__content) if hasattr(msg, '_Static__content') else ""
            chat_text += content

        # Use litellm token_counter for accurate counting
        try:
            import litellm
            from infinidev.config.settings import settings
            chat_tokens = litellm.token_counter(
                model=settings.LLM_MODEL,
                text=chat_text,
            ) if chat_text else 0
        except Exception:
            # Fallback if litellm tokenizer fails
            chat_tokens = len(chat_text) // 4

        self.context_calculator.calculate_usage(
            chat_tokens=chat_tokens,
            task_tokens=task_tokens,
        )
        status = self.context_calculator.get_context_status()
        self._context_panel.update_status(status)

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

            # Update context window display with token usage from engine
            task_tokens = data.get("tokens_total", 0)
            self._refresh_context_panel(task_tokens=task_tokens)

        elif event_type == "loop_tool_call":
            tool_name = data.get("tool_name", "")
            tool_detail = data.get("tool_detail", "")
            action_text = f"⚙️ {tool_name}\n"
            if tool_detail:
                action_text += f"   {tool_detail}"
            self.query_one("#actions-panel").update_content(action_text)

        elif event_type == "loop_user_message":
            msg = data.get("message", "")
            if msg:
                self.add_message("Infinidev", msg, "agent")

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
        msg_class = f"{type}-msg"
        color = self._SENDER_COLORS.get(type, "#cccccc")
        text = str(text) if text else ""

        container = Vertical(classes=msg_class)
        header = self._safe_static(f"[bold {color}]{sender}:[/bold {color}]")
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

        history.mount(container)
        container.mount(header)
        container.mount(body)
        history.scroll_end(animate=False)

        # Refresh context panel to reflect updated chat tokens
        if hasattr(self, '_context_panel'):
            self._refresh_context_panel()

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

    def _hide_thinking(self) -> None:
        """Hide the thinking indicator."""
        try:
            indicator = self.query_one("#thinking-indicator", Vertical)
            indicator.remove()
        except Exception:
            pass

    # ── Submit / engine ──────────────────────────────────

    @on(ChatInput.Submitted)
    def handle_submit(self, event: ChatInput.Submitted):
        self.query_one("#autocomplete-menu", OptionList).remove_class("-visible")
        user_text = event.value
        self.add_message("You", user_text, "user")
        if user_text.startswith("!"):
            # Shell command: execute directly
            self._execute_shell_command(user_text[1:])  # Strip the '!' prefix
        elif user_text.startswith("/"):
            self.handle_command(user_text)
        else:
            self._show_thinking()
            self.run_engine(user_text)

    def _update_status_bar(self) -> None:
        from infinidev.config.settings import settings
        model = settings.LLM_MODEL.split("/", 1)[-1] if "/" in settings.LLM_MODEL else settings.LLM_MODEL
        cwd = os.path.basename(os.getcwd())
        self.query_one("#status-bar", Static).update(f" Model: {model}  │  Project: {cwd}")

    def _show_thinking(self) -> None:
        """Mount the thinking indicator at the bottom of chat history."""
        history = self.query_one("#chat-history")
        indicator = Vertical(
            Static("Infinidev is thinking..."),
            LoadingIndicator(),
            id="thinking-indicator",
        )
        history.mount(indicator)
        history.scroll_end(animate=False)

    def _hide_thinking(self) -> None:
        """Remove the thinking indicator."""
        try:
            self.query_one("#thinking-indicator").remove()
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

    @work(exclusive=True, thread=True)
    def run_engine(self, user_input: str):
        try:
            reload_all()
            store_conversation_turn(self.session_id, 'user', user_input)
            self.agent._session_summaries = get_recent_summaries(self.session_id, limit=10)
            self.agent.activate_context(session_id=self.session_id)
            try:
                result = self.engine.execute(
                    agent=self.agent,
                    task_prompt=(user_input, "Complete the task and report findings."),
                    verbose=True,
                )
                if not result or not result.strip():
                    result = "Done. (no additional output)"
                self.call_from_thread(self._hide_thinking)
                self.call_from_thread(self.add_message, "Infinidev", result, "agent")
                store_conversation_turn(self.session_id, 'assistant', result, result[:200])
            finally:
                self.agent.deactivate()
                self.call_from_thread(self._hide_thinking)
                self.call_from_thread(self.query_one("#actions-panel").update_content, "Idle")
        except Exception as e:
            self.call_from_thread(self._hide_thinking)
            self.call_from_thread(self.add_message, "Error", str(e), "system")

    # ── Commands ─────────────────────────────────────────

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
                "/findings            Browse all findings\n"
                "/knowledge           Browse project knowledge\n"
                "/clear               Clear chat\n"
                "/exit, /quit         Exit",
                "system",
            )
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
        else:
            self.add_message("System", f"Unknown command: {cmd}", "system")

    # ── Background workers ───────────────────────────────

    @work(thread=True)
    def _browse_findings(self, filter_type: str | None):
        from infinidev.db.service import get_all_findings
        findings = get_all_findings()
        if filter_type:
            findings = [f for f in findings if f["finding_type"] == filter_type]
        title = "Project Knowledge" if filter_type == "project_context" else "All Findings"
        self.call_from_thread(self.push_screen, FindingsBrowserScreen(findings, title=title))

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
        self.call_from_thread(self.push_screen, ModelPickerScreen(models, current_tag), callback=on_dismiss)


if __name__ == "__main__":
    app = InfinidevTUI()
    app.run()
