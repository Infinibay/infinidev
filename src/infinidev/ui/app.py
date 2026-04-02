"""Main Infinidev TUI Application built on prompt_toolkit.

This replaces the Textual-based InfinidevTUI in cli/tui.py.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import deque
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import (
    ConditionalContainer, HSplit, Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.filters import Condition
from prompt_toolkit.mouse_events import MouseEventType

from infinidev.ui.theme import (
    TEXT, TEXT_MUTED, PRIMARY, ACCENT, SURFACE, SURFACE_LIGHT,
    STYLE_SIDEBAR_CONTENT, CHAT_INPUT_HEIGHT,
    SCROLLBAR_BG, SCROLLBAR_FG,
)
from infinidev.ui.keybindings import create_global_keybindings
from infinidev.ui.layout import build_layout
from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.controls.chat_input import create_chat_input
from infinidev.ui.controls.autocomplete import AutocompleteState, COMMANDS

logger = logging.getLogger(__name__)


class InfinidevApp:
    """Application state and controller for the Infinidev TUI.

    Holds all mutable state. The prompt_toolkit Application references this
    object via layout lambdas and keybinding handlers.
    """

    def __init__(self) -> None:
        # ── UI state ────────────────────────────────────────
        self.explorer_visible: bool = False
        self.active_tab: str = "chat"  # "chat" or tab_id for files
        self.active_dialog: str | None = None

        # ── Chat state ──────────────────────────────────────
        self.chat_messages: list[dict[str, Any]] = []
        self._chat_history_control = ChatHistoryControl(self.chat_messages)
        self._autocomplete = AutocompleteState(on_select=self._apply_autocomplete)

        # Chat input
        self._chat_buffer, self._chat_input_control, self._chat_input_kb = \
            create_chat_input(
                on_submit=self._handle_submit,
                on_text_changed=lambda text: self._autocomplete.update(text),
                is_autocomplete_visible=lambda: self._autocomplete.visible,
                on_autocomplete_apply=lambda: self._autocomplete.apply_selected(),
                on_autocomplete_next=lambda: (self._autocomplete.select_next(), self.invalidate()),
                on_autocomplete_prev=lambda: (self._autocomplete.select_prev(), self.invalidate()),
                on_autocomplete_dismiss=lambda: (self._autocomplete.dismiss(), self.invalidate()),
                chat_history_control=self._chat_history_control,
            )

        # ── Sidebar state ───────────────────────────────────
        self._context_status: dict[str, Any] = {}
        self._context_flow: str = ""
        self._plan_text: str = ""
        self._steps_text: str = ""
        self._actions_text: str = ""
        self._log_lines: deque[str] = deque(maxlen=15)

        # ── File state ──────────────────────────────────────
        self._open_files: dict[str, str] = {}      # tab_id -> file_path
        self._tab_names: dict[str, str] = {}        # tab_id -> display name
        self._dirty_files: set[str] = set()         # dirty tab_ids
        self._file_diffs: dict[str, dict] = {}      # path -> diff info
        self._editors: dict[str, Any] = {}           # tab_id -> FileEditor
        self._editor_windows: dict[str, Any] = {}   # tab_id -> stable Window
        self._tree_control = None                    # DirectoryTreeControl (lazy)
        self._tree_window = None                     # Stable Window for the tree
        # Stable placeholder window (created once, reused until tree inits)
        from prompt_toolkit.layout.containers import Window as _W
        from prompt_toolkit.layout.controls import FormattedTextControl as _FTC
        self._explorer_placeholder = _W(
            content=_FTC(lambda: [(f"{TEXT_MUTED}", " Press Ctrl+E to load explorer")]),
        )

        # Search bar state
        from infinidev.ui.controls.search_bar import SearchBarState
        self._search_bar = SearchBarState()

        # ── Engine state ────────────────────────────────────
        self._engine_running: bool = False
        self._pending_inputs: list[str] = []
        self._gather_next_task: bool = False
        self._tree_resolved_lines: list[str] = []
        self.session_id: str = str(uuid.uuid4())

        # ── Analysis state ──────────────────────────────────
        self._analysis_waiting: bool = False
        self._analysis_event = None
        self._analysis_answer: str = ""
        self._analysis_original_input: str = ""

        # ── Plan review state ──────────────────────────────
        self._plan_review_waiting: bool = False
        self._plan_review_event = None
        self._plan_review_answer: str = ""

        # ── Engine objects (lazy-initialized on first run) ──
        self.engine = None       # LoopEngine
        self.analyst = None      # AnalysisEngine
        self.reviewer = None     # ReviewEngine
        self.agent = None        # InfinidevAgent
        self.context_calculator = None

        # ── Settings dialog state ───────────────────────────
        self._settings_state = None
        self._settings_sections_ctrl_window = None
        self._settings_settings_ctrl_window = None

        # ── Findings dialog state ───────────────────────────
        self._findings_list_ctrl = None
        self._findings_list_window = None
        self._findings_detail_window = None

        # ── Permission state ────────────────────────────────
        self._permission_waiting: bool = False
        self._permission_event = None
        self._permission_approved: bool = False

        # ── Refs set by layout.py ───────────────────────────
        self._float_container = None
        self.status_bar_control = None
        self.footer_control = None

        # ── Build stable content windows (created once) ─────
        from prompt_toolkit.layout.margins import ScrollbarMargin
        self._chat_content_window = HSplit([
            Window(
                content=self._chat_history_control,
                wrap_lines=False,
                right_margins=[ScrollbarMargin(display_arrows=True)],
            ),
            ConditionalContainer(
                content=Window(
                    content=FormattedTextControl(lambda: self._autocomplete.get_fragments()),
                    height=8,
                    style=f"bg:{SURFACE_LIGHT}",
                ),
                filter=Condition(lambda: self._autocomplete.visible),
            ),
        ])

        # ── Build the Application ───────────────────────────
        global_kb = create_global_keybindings(self)
        self._layout = build_layout(self)

        from prompt_toolkit.styles import Style as PTStyle
        app_style = PTStyle.from_dict({
            "scrollbar.background": f"bg:{SCROLLBAR_BG}",
            "scrollbar.button":    f"bg:{SCROLLBAR_FG}",
            "scrollbar.arrow":     f"{SCROLLBAR_FG}",
        })

        self.app = Application(
            layout=self._layout,
            key_bindings=global_kb,
            style=app_style,
            full_screen=True,
            mouse_support=True,
        )

        # Welcome message
        self.add_message("System", "Welcome to Infinidev! Type your instruction or /help.", "system")

    # ── Run ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Launch the full-screen TUI."""
        self.app.run()

    def invalidate(self) -> None:
        """Request a screen redraw (thread-safe)."""
        self.app.invalidate()

    # ── Message management ───────────────────────────────────────────

    def add_message(self, sender: str, text: str, msg_type: str = "agent") -> None:
        """Append a message to the chat history and trigger redraw."""
        self.chat_messages.append({
            "sender": sender,
            "text": str(text) if text else "",
            "type": msg_type,
        })
        self._chat_history_control.invalidate_cache()
        # Thread-safe invalidate — safe to call from worker threads
        try:
            self.invalidate()
        except Exception:
            pass  # App might not be running yet during init

    # ── Engine initialization ────────────────────────────────────────

    def _ensure_engine(self) -> None:
        """Lazy-initialize engine objects on first use."""
        if self.engine is not None:
            return

        from infinidev.db.service import init_db
        from infinidev.engine.loop_engine import LoopEngine
        from infinidev.engine.analysis_engine import AnalysisEngine
        from infinidev.engine.review_engine import ReviewEngine
        from infinidev.agents.base import InfinidevAgent
        from infinidev.flows.event_listeners import event_bus

        init_db()

        # Register UI hooks
        from infinidev.engine.ui_hooks import register_ui_hooks
        register_ui_hooks()

        # Subscribe to engine events
        event_bus.subscribe(self.on_loop_event)

        self.engine = LoopEngine()
        self.analyst = AnalysisEngine()
        self.reviewer = ReviewEngine()
        self.agent = InfinidevAgent(agent_id="tui_agent")

        # Register permission handler
        from infinidev.tools.permission import set_permission_handler
        set_permission_handler(self._handle_permission_request)

        # Context calculator — fetch actual model context window size
        from infinidev.ui.context_calculator import calculator
        self.context_calculator = calculator
        import asyncio
        try:
            asyncio.run(self.context_calculator.update_model_context())
        except Exception:
            pass  # Falls back to default 4096
        self._context_status = self.context_calculator.get_context_status()

        # Tech detection
        from infinidev.config.tech_detection import detect_tech_hints
        self.agent._tech_hints = detect_tech_hints(os.getcwd())

        # Update status bar with model info
        self._update_status_bar()

    # ── Permission handler ───────────────────────────────────────────

    def _handle_permission_request(self, tool_name: str, description: str, details: str) -> bool:
        """Handle permission request from worker thread. Blocks until user responds."""
        import threading

        evt = threading.Event()
        self._permission_event = evt
        self._permission_waiting = True
        self._permission_approved = False

        # Show permission UI — update state and invalidate
        preview = details.split("\n")[0][:80]
        if len(details) > len(preview):
            preview += "..."
        self._actions_text = f"PERMISSION REQUIRED\n{preview}\n\n[Allow] [Deny] — respond with 'allow' or 'deny'"
        self._permission_details = f"{description}\n\n{details}"
        self.invalidate()

        # For now, permission is handled via chat input
        # The user types "allow" or "deny" and it's intercepted in _handle_submit
        evt.wait()

        approved = self._permission_approved
        self._permission_event = None
        self._permission_waiting = False
        return approved

    # ── Submit handler ───────────────────────────────────────────────

    def _apply_autocomplete(self, cmd: str) -> None:
        """Apply an autocomplete selection to the chat input."""
        from prompt_toolkit.document import Document
        self._chat_buffer.set_document(Document(f"{cmd} "), bypass_readonly=True)
        self._autocomplete.dismiss()
        self.invalidate()

    def _handle_submit(self, user_text: str) -> None:
        """Called when user presses Enter in chat input."""
        self._autocomplete.dismiss()
        self.add_message("You", user_text, "user")

        # Permission response
        if self._permission_waiting and self._permission_event is not None:
            lower = user_text.strip().lower()
            if lower in ("allow", "y", "yes"):
                self._permission_approved = True
                self._permission_event.set()
                self._actions_text = "Approved -- continuing..."
                self.invalidate()
                return
            elif lower in ("deny", "n", "no"):
                self._permission_approved = False
                self._permission_event.set()
                self._actions_text = "Denied"
                self.invalidate()
                return

        # Analysis answer feed-back
        if self._analysis_waiting and self._analysis_event is not None:
            self._analysis_answer = user_text
            self._analysis_waiting = False
            self._analysis_event.set()
            return

        # Plan review feed-back
        if self._plan_review_waiting and self._plan_review_event is not None:
            self._plan_review_answer = user_text
            self._plan_review_waiting = False
            self._plan_review_event.set()
            return

        if user_text.startswith("!"):
            self._execute_shell_command(user_text[1:])
        elif user_text.startswith("/"):
            self.handle_command(user_text)
        else:
            if self._engine_running:
                self._pending_inputs.append(user_text)
                self.add_message("System", "Queued -- waiting for current task to finish.", "system")
            else:
                self._engine_running = True
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_engine_task
                run_in_background(self, run_engine_task, self, user_text, exclusive=True)

    # ── Command handler ──────────────────────────────────────────────

    def handle_command(self, cmd_text: str) -> None:
        """Handle /commands from chat input."""
        parts = cmd_text.split()
        cmd = parts[0].lower()

        if cmd in ("/exit", "/quit"):
            self.app.exit()

        elif cmd == "/clear":
            self.chat_messages.clear()
            self._chat_history_control.invalidate_cache()
            self._log_lines.clear()
            self._plan_text = ""
            self._steps_text = ""
            self._actions_text = ""
            self.invalidate()

        elif cmd == "/help":
            self.add_message(
                "System",
                "Ctrl+E                Toggle file explorer\n"
                "F2 / F3 / F4          Focus: Chat / Explorer / Sidebar\n"
                "Ctrl+W                Close file tab\n"
                "--------------------------------------------\n"
                "!ls                   Execute shell command\n"
                "!grep foo *.py        Run shell with piping\n"
                "/models               Show current model\n"
                "/models list          List available Ollama models\n"
                "/models set <name>    Change model\n"
                "/models manage        Pick a model interactively\n"
                "/settings             Show current settings\n"
                "/settings <key>       Show specific setting\n"
                "/settings <key> <val> Change setting\n"
                "/settings reset       Reset to defaults\n"
                "/plan <task>          Generate plan, review, then execute\n"
                "/explore <problem>    Decompose and explore a complex problem\n"
                "/init                 Explore and document the current project\n"
                "/findings             Browse all findings\n"
                "/knowledge            Browse project knowledge\n"
                "/documentation        Browse cached library docs\n"
                "/clear                Clear chat\n"
                "/exit, /quit          Exit",
                "system",
            )

        elif cmd == "/settings":
            self._handle_settings(parts)

        elif cmd == "/models":
            self._handle_models(parts)

        elif cmd == "/findings":
            self._open_findings_dialog(filter_type=None)

        elif cmd == "/knowledge":
            self._open_findings_dialog(filter_type="project_context")

        elif cmd in ("/documentation", "/docs"):
            self.add_message("System", "[Docs browser — coming soon]", "system")

        elif cmd == "/explore":
            problem = " ".join(parts[1:]) if len(parts) > 1 else ""
            if not problem:
                self.add_message("System", "Usage: /explore <problem description>", "system")
            elif self._engine_running:
                self.add_message("System", "Cannot run /explore while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", f"Exploring: {problem}", "system")
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_explore_task
                run_in_background(self, run_explore_task, self, problem, exclusive=True)

        elif cmd == "/brainstorm":
            problem = " ".join(parts[1:]) if len(parts) > 1 else ""
            if not problem:
                self.add_message("System", "Usage: /brainstorm <problem description>", "system")
            elif self._engine_running:
                self.add_message("System", "Cannot run /brainstorm while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", f"Brainstorming: {problem}", "system")
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_brainstorm_task
                run_in_background(self, run_brainstorm_task, self, problem, exclusive=True)

        elif cmd == "/plan":
            task = " ".join(parts[1:]) if len(parts) > 1 else ""
            if not task:
                self.add_message("System", "Usage: /plan <task description>", "system")
            elif self._engine_running:
                self.add_message("System", "Cannot run /plan while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", f"Planning: {task}", "system")
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_plan_task
                run_in_background(self, run_plan_task, self, task, exclusive=True)

        elif cmd == "/think":
            self._gather_next_task = True
            self.add_message(
                "System",
                "Gather mode enabled for the next task. Send your prompt and "
                "infinidev will deeply analyze the codebase before acting.",
                "system",
            )

        elif cmd == "/init":
            if self._engine_running:
                self.add_message("System", "Cannot run /init while a task is running.", "system")
            else:
                self._engine_running = True
                self.add_message("System", "Exploring and documenting project...", "system")
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_init_task
                run_in_background(self, run_init_task, self, exclusive=True)

        else:
            self.add_message("System", f"Unknown command: {cmd}", "system")

    # ── Settings handler ─────────────────────────────────────────────

    def _handle_settings(self, parts: list[str]) -> None:
        """Handle /settings subcommands."""
        from infinidev.config.settings import settings, reload_all

        if len(parts) == 1 or (len(parts) == 2 and parts[1].lower() == "browse"):
            # Open settings modal
            self._open_settings_dialog()
            return

        subcmd = parts[1].lower()

        if subcmd == "reset":
            settings.reset_to_defaults()
            reload_all()
            self.add_message("System", "Settings reset to defaults.", "system")

        elif subcmd == "export" and len(parts) > 2:
            path = parts[2]
            settings.export_to_file(path)
            self.add_message("System", f"Settings exported to {path}", "system")

        elif subcmd == "import" and len(parts) > 2:
            path = parts[2]
            settings.import_from_file(path)
            reload_all()
            self.add_message("System", f"Settings imported from {path}", "system")

        elif len(parts) == 2:
            # Show specific setting
            key = parts[1].upper()
            val = getattr(settings, key, None)
            if val is not None:
                self.add_message("System", f"{key}: {val}", "system")
            else:
                self.add_message("System", f"Unknown setting: {key}", "system")

        elif len(parts) >= 3:
            # Set a value
            key = parts[1].upper()
            value = " ".join(parts[2:])
            try:
                settings.save_user_settings({key: value})
                reload_all()
                self.add_message("System", f"{key} = {value}", "system")
                self._update_status_bar()
            except Exception as e:
                self.add_message("System", f"Error setting {key}: {e}", "system")

    # ── Models handler ───────────────────────────────────────────────

    def _handle_models(self, parts: list[str]) -> None:
        """Handle /models subcommands."""
        from infinidev.config.settings import settings, reload_all
        from infinidev.config.providers import get_provider, fetch_models

        subcmd = parts[1].lower() if len(parts) > 1 else "info"

        if subcmd == "set" and len(parts) > 2:
            new_model = parts[2]
            # Use current provider's prefix if model has no prefix
            if "/" not in new_model:
                provider = get_provider(settings.LLM_PROVIDER)
                new_model = f"{provider.prefix}{new_model}"
            settings.save_user_settings({"LLM_MODEL": new_model})
            reload_all()
            self.add_message("System", f"Model updated to: {settings.LLM_MODEL}", "system")
            self._update_status_bar()

        elif subcmd == "list":
            provider = get_provider(settings.LLM_PROVIDER)
            self.add_message("System", f"Fetching models for {provider.display_name}...", "system")
            try:
                models = fetch_models(settings.LLM_PROVIDER, settings.LLM_API_KEY, settings.LLM_BASE_URL)
                if models:
                    model_list = "\n".join(f"  {m}" for m in models)
                    self.add_message("System", f"Available models:\n{model_list}", "system")
                else:
                    self.add_message("System", "No models found. Check API key and connection.", "system")
            except Exception as e:
                self.add_message("System", f"Error fetching models: {e}", "system")

        elif subcmd == "manage":
            self.add_message("System", "[Model picker — coming soon]", "system")

        else:
            provider = get_provider(settings.LLM_PROVIDER)
            self.add_message(
                "System",
                f"Provider: {provider.display_name}\n"
                f"Model: {settings.LLM_MODEL}\n"
                f"Base URL: {settings.LLM_BASE_URL}",
                "system",
            )

    # ── Shell commands ───────────────────────────────────────────────

    def _execute_shell_command(self, cmd: str) -> None:
        """Execute a !command and show output in chat."""
        import subprocess
        self.add_message("System", f"$ {cmd}", "system")
        try:
            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=30,
                cwd=os.getcwd(),
                stdin=subprocess.DEVNULL,
            )
            output = result.stdout
            if result.stderr:
                output += f"\n{result.stderr}" if output else result.stderr
            if output.strip():
                self.add_message("Shell", output.rstrip(), "system")
            if result.returncode != 0:
                self.add_message("Shell", f"Exit code: {result.returncode}", "system")
        except subprocess.TimeoutExpired:
            self.add_message("Shell", "Command timed out (30s)", "system")
        except Exception as e:
            self.add_message("Shell", f"Error: {e}", "system")

    # ── Context tokens ───────────────────────────────────────────────

    def update_context_tokens(self, task_tokens: int = 0,
                              prompt_tokens: int = 0,
                              completion_tokens: int = 0) -> None:
        """Update context token counts from engine events."""
        if self.context_calculator is None:
            return
        if prompt_tokens:
            self.context_calculator.update_task(prompt_tokens)
        self._context_status = self.context_calculator.get_context_status()

    # ── Keybinding handlers ──────────────────────────────────────────

    def request_quit(self, event=None) -> None:
        if event:
            event.app.exit()
        else:
            self.app.exit()

    def toggle_explorer(self) -> None:
        self.explorer_visible = not self.explorer_visible
        if self.explorer_visible:
            if self._tree_control is None:
                self._init_tree()
            # Focus the tree so arrow keys work immediately
            if self._tree_window is not None:
                try:
                    self.app.layout.focus(self._tree_window)
                except Exception:
                    pass
        else:
            # Return focus to chat when closing explorer
            self.focus_chat()
        self.invalidate()

    def _init_tree(self) -> None:
        """Lazy-initialize the directory tree on first explorer open."""
        from prompt_toolkit.layout.containers import Window
        from infinidev.ui.controls.directory_tree import DirectoryTreeControl
        self._tree_control = DirectoryTreeControl(
            root_path=os.getcwd(),
            on_file_selected=self._on_file_selected,
        )
        # Create the Window ONCE so it can hold stable focus
        self._tree_window = Window(content=self._tree_control)

    def _on_file_selected(self, file_path: str) -> None:
        """Handle file double-click: open in editor tab."""
        self._open_file(file_path)

    def _open_file(self, file_path: str) -> None:
        """Open a file in a new editor tab (or switch to existing)."""
        import os
        from prompt_toolkit.layout.containers import Window as _W, HSplit as _H, ConditionalContainer as _CC
        from prompt_toolkit.layout.controls import FormattedTextControl as _FTC
        from prompt_toolkit.filters import Condition as _Cond
        from infinidev.ui.controls.file_editor import FileEditor

        tab_id = file_path
        name = os.path.basename(file_path)

        # Already open? Just switch to it
        if tab_id in self._editors:
            self.active_tab = tab_id
            try:
                self.app.layout.focus(self._editor_windows[tab_id])
            except Exception:
                pass
            self.invalidate()
            return

        # Create editor
        editor = FileEditor(file_path, on_dirty_change=self._on_dirty_change)
        self._editors[tab_id] = editor
        self._open_files[tab_id] = file_path
        self._tab_names[tab_id] = name

        # Build stable window: editor + conditional search bar
        search = self._search_bar
        editor_window = _H([
            _CC(
                content=_H([
                    _W(content=search.search_control, height=1),
                    _W(content=_FTC(lambda: search.get_status_fragments()), height=1),
                ]),
                filter=_Cond(lambda tid=tab_id: search.visible and self.active_tab == tid),
            ),
            _W(content=editor.control),
        ])
        self._editor_windows[tab_id] = editor_window

        # Switch to new tab and focus
        self.active_tab = tab_id
        try:
            self.app.layout.focus(editor_window)
        except Exception:
            pass
        self.invalidate()

    def _on_dirty_change(self, tab_id: str, dirty: bool) -> None:
        """Track dirty state for tab indicators."""
        if dirty:
            self._dirty_files.add(tab_id)
        else:
            self._dirty_files.discard(tab_id)
        self.invalidate()

    def get_explorer_content(self):
        """Return the stable explorer Window (same instance every call)."""
        if self._tree_window is not None:
            return self._tree_window
        return self._explorer_placeholder

    def close_active_tab(self) -> None:
        """Close the active file tab."""
        tab_id = self.active_tab
        if tab_id == "chat":
            return  # Can't close chat

        editor = self._editors.get(tab_id)
        if editor and editor.is_dirty:
            # For now, warn and don't close. Full unsaved dialog in Phase 8.
            self.add_message("System", f"Unsaved changes in {editor.name}. Save first (Ctrl+S).", "system")
            return

        # Remove editor and tab
        self._editors.pop(tab_id, None)
        self._editor_windows.pop(tab_id, None)
        self._open_files.pop(tab_id, None)
        self._tab_names.pop(tab_id, None)
        self._dirty_files.discard(tab_id)

        # Switch to previous tab or chat
        remaining = list(self._tab_names.keys())
        self.active_tab = remaining[-1] if remaining else "chat"
        if self.active_tab == "chat":
            self.focus_chat()
        self.invalidate()

    def save_active_file(self) -> None:
        """Save the active file editor."""
        tab_id = self.active_tab
        editor = self._editors.get(tab_id)
        if not editor:
            return
        if editor.save():
            self.add_message("System", f"Saved: {editor.name}", "system")
        else:
            self.add_message("System", f"Failed to save: {editor.name}", "system")
        self.invalidate()

    def toggle_search_bar(self) -> None:
        """Toggle the in-file search bar for the active editor."""
        tab_id = self.active_tab
        editor = self._editors.get(tab_id)
        if not editor:
            return
        self._search_bar.set_target(editor.buffer)
        self._search_bar.toggle()
        if self._search_bar.visible:
            try:
                self.app.layout.focus(self._search_bar.search_control)
            except Exception:
                pass
        self.invalidate()

    def show_project_search(self) -> None:
        # Phase 8
        pass

    def focus_chat(self) -> None:
        """Focus the chat input buffer."""
        try:
            self.app.layout.focus(self._chat_input_control)
        except Exception:
            pass
        self.invalidate()

    def focus_explorer(self) -> None:
        if not self.explorer_visible:
            self.explorer_visible = True
            if self._tree_control is None:
                self._init_tree()
        if self._tree_window is not None:
            try:
                self.app.layout.focus(self._tree_window)
            except Exception:
                pass
        self.invalidate()

    def focus_sidebar(self) -> None:
        pass

    def handle_escape(self) -> None:
        if self.active_dialog:
            self.active_dialog = None
            self.focus_chat()
            self.invalidate()
            return
        if self._autocomplete.visible:
            self._autocomplete.dismiss()
            self.invalidate()
            return
        # Phase 4: task cancellation

    # ── Settings dialog ──────────────────────────────────────────────

    def _open_settings_dialog(self) -> None:
        """Open the settings editor modal."""
        if self._settings_state is None:
            self._init_settings_dialog()
        self.active_dialog = "settings_editor"
        self._settings_state.focus_panel = "sections"
        self._settings_state.section_cursor = 0
        self._settings_state.setting_cursor = 0
        self._settings_state.editing = False
        # Focus the sections panel
        try:
            self.app.layout.focus(self._settings_sections_ctrl_window)
        except Exception:
            pass
        self.invalidate()

    def _init_settings_dialog(self) -> None:
        """Lazy-create the settings dialog and register as a Float."""
        from prompt_toolkit.layout.containers import Float, ConditionalContainer
        from prompt_toolkit.filters import Condition
        from infinidev.ui.dialogs.settings_editor import create_settings_editor

        def _on_focus_change(panel: str) -> None:
            try:
                if panel == "sections":
                    self.app.layout.focus(self._settings_sections_ctrl_window)
                elif panel == "dropdown":
                    self.app.layout.focus(self._settings_dropdown_window)
                else:
                    self.app.layout.focus(self._settings_settings_ctrl_window)
            except Exception:
                pass
            self.invalidate()

        def _on_edit_start():
            """Focus the edit buffer when inline editing starts."""
            try:
                self.app.layout.focus(self._settings_edit_buffer_window)
            except Exception:
                pass
            self.invalidate()

        from infinidev.ui.dialogs.settings_editor import DropdownControl

        frame, state, sections_ctrl, settings_ctrl = create_settings_editor(
            on_save=lambda k, v: self.invalidate(),
            on_focus_change=_on_focus_change,
            on_edit_start=_on_edit_start,
        )
        self._settings_state = state

        # Create stable windows for focus management
        from prompt_toolkit.layout.containers import Window
        from prompt_toolkit.layout.dimension import Dimension as D
        self._settings_sections_ctrl_window = Window(content=sections_ctrl, width=16)
        self._settings_settings_ctrl_window = Window(content=settings_ctrl)

        # Edit buffer window (stable ref for focus)
        from prompt_toolkit.layout.containers import HSplit, VSplit
        from prompt_toolkit.layout.controls import BufferControl
        self._settings_edit_buffer_window = Window(
            content=BufferControl(buffer=state.edit_buffer, focusable=True),
            height=1,
            style=f"bg:{SURFACE_LIGHT} #ffffff",
        )

        # Dropdown picker window
        dropdown_ctrl = DropdownControl(state)
        self._settings_dropdown_window = Window(
            content=dropdown_ctrl,
            style=f"bg:{SURFACE}",
        )

        # Rebuild the frame body with our stable windows
        # The dropdown overlays the settings panel when open
        settings_area = HSplit([
            ConditionalContainer(
                content=self._settings_dropdown_window,
                filter=Condition(lambda: state.dropdown_open),
            ),
            ConditionalContainer(
                content=HSplit([
                    self._settings_settings_ctrl_window,
                    ConditionalContainer(
                        content=self._settings_edit_buffer_window,
                        filter=Condition(lambda: state.editing),
                    ),
                ]),
                filter=Condition(lambda: not state.dropdown_open),
            ),
        ], width=D(weight=1))

        body = VSplit([
            self._settings_sections_ctrl_window,
            Window(width=1, char="│", style=f"{PRIMARY}"),
            settings_area,
        ])

        from infinidev.ui.dialogs.base import dialog_frame as _df
        actual_frame = _df("Settings", body, width=90, height=30, border_color=PRIMARY)

        # Add to float container
        dialog_float = Float(
            content=ConditionalContainer(
                content=actual_frame,
                filter=Condition(lambda: self.active_dialog == "settings_editor"),
            ),
            transparent=False,
        )
        if self._float_container:
            self._float_container.floats.append(dialog_float)

    # ── Findings dialog ────────────────────────────────────────────

    def _open_findings_dialog(self, filter_type: str | None = None) -> None:
        """Open the findings browser modal."""
        if self._findings_list_ctrl is None:
            self._init_findings_dialog()

        # Load findings from DB
        from infinidev.db.service import get_all_findings
        try:
            findings = get_all_findings()
        except Exception:
            findings = []
        if filter_type:
            findings = [f for f in findings if f.get("finding_type") == filter_type]

        self._findings_list_ctrl.findings = findings
        self._findings_list_ctrl.cursor = 0
        self.active_dialog = "findings_browser"

        try:
            self.app.layout.focus(self._findings_list_window)
        except Exception:
            pass
        self.invalidate()

    def _init_findings_dialog(self) -> None:
        """Lazy-create the findings dialog and register as a Float."""
        from prompt_toolkit.layout.containers import (
            Float, ConditionalContainer, VSplit, Window,
        )
        from prompt_toolkit.layout.dimension import Dimension as D
        from prompt_toolkit.filters import Condition
        from infinidev.ui.dialogs.findings_browser import (
            FindingsListControl, FindingsDetailControl,
        )
        from infinidev.ui.dialogs.base import dialog_frame as _df

        list_ctrl = FindingsListControl()
        detail_ctrl = FindingsDetailControl(list_ctrl)
        self._findings_list_ctrl = list_ctrl

        # Stable windows
        self._findings_list_window = Window(content=list_ctrl, width=D(weight=40))
        self._findings_detail_window = Window(content=detail_ctrl, width=D(weight=60))

        # Add keybindings for navigation and panel switching
        from prompt_toolkit.key_binding import KeyBindings
        nav_kb = KeyBindings()

        @nav_kb.add("up")
        def _up(event):
            list_ctrl.move_cursor(-1)

        @nav_kb.add("down")
        def _down(event):
            list_ctrl.move_cursor(1)

        @nav_kb.add("escape")
        def _close(event):
            self.active_dialog = None
            self.focus_chat()
            self.invalidate()

        # Attach keybindings to the list control
        list_ctrl._nav_kb = nav_kb
        _orig_get_kb = list_ctrl.get_key_bindings if hasattr(list_ctrl, 'get_key_bindings') else None

        def _get_kb():
            return nav_kb
        list_ctrl.get_key_bindings = _get_kb
        list_ctrl.is_focusable = lambda: True

        # Mouse click on list
        from prompt_toolkit.mouse_events import MouseEventType as _MET

        def _mouse(mouse_event):
            if mouse_event.event_type == _MET.MOUSE_UP:
                row = mouse_event.position.y
                if 0 <= row < len(list_ctrl.findings):
                    list_ctrl.cursor = row
            return None
        list_ctrl.mouse_handler = _mouse

        title = "Findings"
        body = VSplit([
            self._findings_list_window,
            Window(width=1, char="│", style=f"{PRIMARY}"),
            self._findings_detail_window,
        ])

        frame = _df(title, body, width=90, height=30, border_color=PRIMARY)

        dialog_float = Float(
            content=ConditionalContainer(
                content=frame,
                filter=Condition(lambda: self.active_dialog == "findings_browser"),
            ),
            transparent=False,
        )
        if self._float_container:
            self._float_container.floats.append(dialog_float)

    # ── Tab bar fragments ────────────────────────────────────────────

    def _switch_tab(self, tab_id: str) -> None:
        """Switch to a tab and focus its content."""
        self.active_tab = tab_id
        if tab_id == "chat":
            self.focus_chat()
        else:
            win = self._editor_windows.get(tab_id)
            if win:
                try:
                    self.app.layout.focus(win)
                except Exception:
                    pass
        self.invalidate()

    def get_tab_bar_fragments(self) -> FormattedText:
        fragments: list = []

        # Chat tab
        def _click_chat(mouse_event):
            if mouse_event.event_type == MouseEventType.MOUSE_UP:
                self._switch_tab("chat")

        if self.active_tab == "chat":
            fragments.append((f"bg:{PRIMARY} #ffffff bold", " Chat ", _click_chat))
        else:
            fragments.append((f"{TEXT_MUTED} bg:{SURFACE_LIGHT}", " Chat ", _click_chat))

        # File tabs
        for tab_id, name in self._tab_names.items():
            dirty = "* " if tab_id in self._dirty_files else ""

            def _click_tab(mouse_event, tid=tab_id):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self._switch_tab(tid)

            if self.active_tab == tab_id:
                fragments.append((f"bg:{PRIMARY} #ffffff bold", f" {dirty}{name} ", _click_tab))
            else:
                fragments.append((f"{TEXT_MUTED} bg:{SURFACE_LIGHT}", f" {dirty}{name} ", _click_tab))
            fragments.append(("", " "))

        return FormattedText(fragments)

    # ── Content area ─────────────────────────────────────────────────

    def get_active_content(self):
        """Return the stable Window/container for the currently active tab."""
        if self.active_tab == "chat":
            return self._chat_content_window
        editor_win = self._editor_windows.get(self.active_tab)
        if editor_win is not None:
            return editor_win
        # Fallback
        return self._chat_content_window

    # ── Sidebar fragments ────────────────────────────────────────────

    def get_context_fragments(self) -> FormattedText:
        model = self._context_status.get("model", "unknown")
        max_ctx = self._context_status.get("max_context", 4096)
        flow_part = f"  {self._context_flow}" if self._context_flow else ""

        fragments: list[tuple[str, str]] = []
        fragments.append((f"{TEXT} bold", f"{model}"))
        fragments.append((f"{TEXT_MUTED}", f" ({max_ctx} ctx)"))
        if flow_part:
            fragments.append((f"{ACCENT} bold", flow_part))
        fragments.append(("", "\n"))

        chat = self._context_status.get("chat", {})
        fragments.extend(self._usage_bar_fragments(
            "Chat", chat.get("current_tokens", 0),
            chat.get("remaining_tokens", 0),
            chat.get("usage_percentage", 0.0),
        ))
        fragments.append(("", "\n"))

        tasks = self._context_status.get("tasks", {})
        fragments.extend(self._usage_bar_fragments(
            "Task", tasks.get("current_tokens", 0),
            tasks.get("remaining_tokens", 0),
            tasks.get("usage_percentage", 0.0),
        ))

        return FormattedText(fragments)

    def _usage_bar_fragments(self, label: str, used: int,
                             available: int, pct: float) -> list[tuple[str, str]]:
        from infinidev.ui.theme import (
            PROGRESS_GOOD, PROGRESS_WARNING, PROGRESS_CRITICAL,
            BAR_WIDTH, BAR_FILLED, BAR_EMPTY,
        )
        pct_val = min(pct, 1.0)
        if pct_val > 0.8:
            color = PROGRESS_CRITICAL
        elif pct_val > 0.5:
            color = PROGRESS_WARNING
        else:
            color = PROGRESS_GOOD

        filled = int(BAR_WIDTH * pct_val)
        empty = BAR_WIDTH - filled

        return [
            (f"{TEXT} bold", f"{label} "),
            (f"{TEXT_MUTED}", f"{used}/{available} "),
            (f"{color}", BAR_FILLED * filled),
            (f"{TEXT_MUTED}", BAR_EMPTY * empty),
            (f"{color} bold", f" {pct_val * 100:.0f}%"),
        ]

    def get_plan_fragments(self) -> FormattedText:
        if not self._plan_text:
            return FormattedText([(f"{TEXT_MUTED}", " No active plan")])
        return FormattedText([(STYLE_SIDEBAR_CONTENT, self._plan_text)])

    def get_steps_fragments(self) -> FormattedText:
        if not self._steps_text:
            return FormattedText([(f"{TEXT_MUTED}", " Waiting...")])
        return FormattedText([(STYLE_SIDEBAR_CONTENT, self._steps_text)])

    def get_actions_fragments(self) -> FormattedText:
        if not self._actions_text:
            return FormattedText([(f"{TEXT_MUTED}", " Idle")])
        return FormattedText([(STYLE_SIDEBAR_CONTENT, self._actions_text)])

    def get_logs_fragments(self) -> FormattedText:
        if not self._log_lines:
            return FormattedText([(f"{TEXT_MUTED}", " No logs")])
        text = "\n".join(list(self._log_lines)[-5:])
        return FormattedText([(STYLE_SIDEBAR_CONTENT, text)])

    # ── Status bar ───────────────────────────────────────────────────

    def _update_status_bar(self) -> None:
        if self.status_bar_control:
            try:
                from infinidev.config.settings import settings
                model = settings.LLM_MODEL.split("/", 1)[-1] if "/" in settings.LLM_MODEL else settings.LLM_MODEL
                self.status_bar_control.set_model(model)
            except Exception:
                pass
            self.invalidate()

    # ── Event bus integration ────────────────────────────────────────

    def on_loop_event(self, event_type: str, project_id: int,
                      agent_id: str, data: dict[str, Any]) -> None:
        """EventBus subscriber callback — called from engine worker threads."""
        # Pass agent_id through so the event handler can differentiate
        data["_agent_id"] = agent_id

        from infinidev.ui.event_handler import process_event
        process_event(self, event_type, data)
        self.invalidate()


def run_tui() -> None:
    """Entry point to launch the Infinidev TUI."""
    import sys

    # Redirect stderr to suppress subprocess output that would corrupt
    # the full-screen terminal. Log messages go to file handler instead.
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, "w")
    try:
        app_state = InfinidevApp()
        if app_state.status_bar_control:
            app_state.status_bar_control.set_model("loading...")
            app_state.status_bar_control.set_project(os.path.basename(os.getcwd()))
        app_state.run()
    finally:
        sys.stderr.close()
        sys.stderr = original_stderr
