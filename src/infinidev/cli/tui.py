"""TUI implementation for Infinidev using Textual."""

import os
import pathlib
import uuid
from typing import Any
from textual.app import App, ComposeResult
from textual.screen import ModalScreen
from typing import Optional, Callable, Set
from textual.widgets import (
    Header, Footer, Static, TextArea, Label, OptionList, Markdown,
    DirectoryTree, TabbedContent, TabPane,
)
from textual.widgets.option_list import Option
from textual.containers import Vertical, Horizontal, VerticalScroll
from textual.binding import Binding
from textual import on, events, work
from watchfiles import FileChange

# Infinidev imports
from infinidev.agents.base import InfinidevAgent
from infinidev.engine.loop_engine import LoopEngine, set_event_callback
from infinidev.db.service import (
    init_db, store_conversation_turn, get_recent_summaries,
)
from infinidev.config.settings import reload_all
from infinidev.cli.file_watcher import FileWatcher

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
        self.content_static.update(text)


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

    /* ── Chat messages ──────────────────────────────────── */
    .user-msg { height: auto; color: #c4c4c4; margin-bottom: 1; background: #1a2a1a; padding: 0 1; border-left: tall #4a9f4a; }
    .agent-msg { height: auto; color: #e0e0e0; margin-bottom: 1; background: #1a1a2a; padding: 0 1; border-left: tall #5b7fbf; }
    .agent-msg Markdown { margin: 0; padding: 0; background: transparent; }
    .agent-msg MarkdownFence { margin: 1 0; max-height: 20; overflow-y: auto; }
    .agent-msg MarkdownH1, .agent-msg MarkdownH2, .agent-msg MarkdownH3 { margin: 1 0 0 0; padding: 0; background: transparent; border: none; }
    .system-msg { height: auto; color: #a0a060; text-style: italic; margin-bottom: 1; padding: 0 1; }
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
                yield SidebarPanel("PLANNING", id="plan-panel")
                yield SidebarPanel("STEPS", id="steps-panel")
                yield SidebarPanel("ACTIONS", id="actions-panel")
                yield SidebarPanel("LOGS", id="logs-panel")
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

        from infinidev.config.tech_detection import detect_tech_hints
        self.agent._tech_hints = detect_tech_hints(os.getcwd())

        # Initialize file watcher
        self._file_watcher: Optional[FileWatcher] = None
        self._watcher_started = False
        self._expand_handlers: list[Callable] = []
        self._collapse_handlers: list[Callable] = []
        self._visible_paths: set[pathlib.Path] = set()
        self._start_file_watcher()

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

        elif event_type == "loop_tool_call":
            tool_name = data.get("tool_name", "")
            tool_detail = data.get("tool_detail", "")
            action_text = f"⚙️ {tool_name}\n"
            if tool_detail:
                action_text += f"   {tool_detail}"
            self.query_one("#actions-panel").update_content(action_text)

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

    def add_message(self, sender: str, text: str, type: str = "agent"):
        history = self.query_one("#chat-history")
        msg_class = f"{type}-msg"
        color = self._SENDER_COLORS.get(type, "#cccccc")
        text = str(text) if text else ""

        container = Vertical(classes=msg_class)
        header = Static(f"[bold {color}]{sender}:[/bold {color}]")
        if type == "agent" and ("```" in text or "**" in text or "# " in text):
            body = Markdown(text)
        else:
            body = Static(text)
        history.mount(container)
        container.mount(header)
        container.mount(body)
        history.scroll_end(animate=False)

    # ── Submit / engine ──────────────────────────────────

    @on(ChatInput.Submitted)
    def handle_submit(self, event: ChatInput.Submitted):
        self.query_one("#autocomplete-menu", OptionList).remove_class("-visible")
        user_text = event.value
        self.add_message("You", user_text, "user")
        if user_text.startswith("/"):
            self.handle_command(user_text)
        else:
            self.run_engine(user_text)

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
                self.call_from_thread(self.add_message, "Infinidev", result, "agent")
                store_conversation_turn(self.session_id, 'assistant', result, result[:200])
            finally:
                self.agent.deactivate()
                self.call_from_thread(self.query_one("#actions-panel").update_content, "Idle")
        except Exception as e:
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
            self.query_one("#chat-input", ChatInput).focus()
        self.push_screen(ModelPickerScreen(models, current_tag), callback=on_dismiss)


if __name__ == "__main__":
    app = InfinidevTUI()
    app.run()
