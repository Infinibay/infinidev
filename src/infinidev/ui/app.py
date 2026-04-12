"""Main Infinidev TUI Application built on prompt_toolkit.

This replaces the Textual-based InfinidevTUI in cli/tui.py.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections import deque
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import (
    ConditionalContainer, HSplit, VSplit, Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.filters import Condition
from prompt_toolkit.mouse_events import MouseEventType

from infinidev.ui.theme import (
    TEXT, TEXT_MUTED, TEXT_DIM, PRIMARY, ACCENT, SUCCESS,
    SURFACE, SURFACE_LIGHT,
    STYLE_SIDEBAR_CONTENT, CHAT_INPUT_HEIGHT,
    SCROLLBAR_BG, SCROLLBAR_FG,
)
from infinidev.ui.keybindings import create_global_keybindings
from infinidev.ui.layout import build_layout
from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.controls.chat_input import create_chat_input
from infinidev.ui.controls.autocomplete import AutocompleteState, COMMANDS
from infinidev.ui.handlers.files import FileManager
from infinidev.ui.handlers.dialogs import DialogManager

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
        self._thinking_text: str = ""  # Live streaming thinking content
        self._steps_text: str = ""
        self._actions_text: str = ""
        self._streaming_tool_name: str | None = None  # Tool detected during streaming
        self._streaming_token_count: int = 0           # Tokens received so far
        self._log_entries: deque[tuple[float, str]] = deque(maxlen=15)  # (timestamp, text)

        # ── File management (delegated to FileManager) ──────
        self.file_manager = FileManager(self)
        # Backward-compat aliases used by layout.py / keybindings.py
        self._open_files = self.file_manager.open_files
        self._tab_names = self.file_manager.tab_names
        self._dirty_files = self.file_manager.dirty_files
        self._editors = self.file_manager.editors
        self._editor_windows = self.file_manager.editor_windows
        self._search_bar = self.file_manager.search_bar

        # ── Engine state ────────────────────────────────────
        self._engine_running: bool = False
        self._pending_inputs: list[str] = []
        # Hold-Escape-to-cancel state
        self._cancel_hold_start: float | None = None
        self._cancel_last_escape: float = 0.0
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

        # ── Dialog management (delegated to DialogManager) ───
        self.dialog_manager = DialogManager(self)

        # ── Permission state ────────────────────────────────
        self._permission_waiting: bool = False
        self._permission_event = None
        self._permission_approved: bool = False

        # ── Refs set by layout.py ───────────────────────────
        self._float_container = None
        self.status_bar_control = None
        self.footer_control = None

        # ── Build stable content windows (created once) ─────
        from infinidev.ui.controls.clickable_scrollbar import ClickableScrollbar
        self._chat_history_window = Window(
            content=self._chat_history_control,
            wrap_lines=False,
        )
        self._chat_scrollbar = ClickableScrollbar(self._chat_history_window)
        self._chat_content_window = HSplit([
            VSplit([
                self._chat_history_window,
                Window(content=self._chat_scrollbar, width=1,
                       style=f"bg:{SCROLLBAR_BG}"),
            ]),
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

        # ── Animation refresh timer ─────────────────────────────
        self._start_animation_timer()

        # ── Start background workspace indexing ────────────────
        self._index_ready = False
        self._start_background_index()

    # ── Run ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Launch the full-screen TUI."""
        try:
            self.app.run()
        finally:
            self.file_manager.stop_file_watcher()

    def invalidate(self) -> None:
        """Request a screen redraw (thread-safe)."""
        self.app.invalidate()

    # ── Background workspace indexing ───────────────────────────────

    def _start_background_index(self) -> None:
        """Kick off workspace indexing in a background thread.

        Shows progress in the chat history so the user knows what's happening.
        The index runs once at startup; subsequent file changes are handled
        by the file watcher.
        """
        import threading

        def _index_worker():
            from infinidev.config.settings import settings
            if not settings.CODE_INTEL_ENABLED:
                self._index_ready = True
                return

            if self.status_bar_control:
                self.status_bar_control.set_status("indexing...")
                self.invalidate()

            self.add_message(
                "System",
                "Indexing workspace — code intelligence will be ready shortly...",
                "system",
            )
            try:
                from infinidev.cli.initial_index import run_initial_index

                stats = run_initial_index(project_id=1)

                files = stats.get("files_indexed", 0)
                symbols = stats.get("symbols_total", 0)
                elapsed = stats.get("elapsed_ms", 0)

                if files > 0:
                    self.add_message(
                        "System",
                        f"Index ready: {files} files, {symbols} symbols ({elapsed}ms)",
                        "system",
                    )
                else:
                    self.add_message("System", "Index up to date.", "system")

                if self.status_bar_control:
                    self.status_bar_control.set_status("ready")
                    self.invalidate()

            except Exception as exc:
                logger.warning("Background indexing failed: %s", exc)
                self.add_message(
                    "System",
                    f"Indexing failed: {exc} — code intelligence may be limited.",
                    "system",
                )
            finally:
                self._index_ready = True

        t = threading.Thread(target=_index_worker, daemon=True, name="infinidev-index")
        t.start()

    def _start_animation_timer(self) -> None:
        """Periodic invalidation for sidebar animations while engine runs.

        Without this, time-based animations (spinners, blinking) would freeze
        because prompt_toolkit only redraws on explicit invalidate() calls.
        Runs at ~6 FPS — enough for smooth spinners, cheap enough to be idle.
        """
        import threading

        def _tick():
            while True:
                time.sleep(0.16)  # ~6 FPS
                if self._engine_running or self._streaming_token_count > 0:
                    try:
                        self.invalidate()
                    except Exception:
                        pass

        t = threading.Thread(target=_tick, daemon=True, name="infinidev-anim")
        t.start()

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
        from infinidev.engine.loop import LoopEngine
        from infinidev.engine.analysis.analysis_engine import AnalysisEngine
        from infinidev.engine.analysis.review_engine import ReviewEngine
        from infinidev.agents.base import InfinidevAgent
        from infinidev.flows.event_listeners import event_bus

        init_db()

        # Register UI hooks
        from infinidev.engine.hooks.ui_hooks import register_ui_hooks
        register_ui_hooks()
        from infinidev.engine.behavior.hook import register_behavior_hooks
        register_behavior_hooks()

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
                # Inject message into the running loop — will appear in next iteration
                if self.engine is not None:
                    self.engine.inject_message(user_text)
                    self.add_message("System", "Message injected — the agent will see it on its next LLM call.", "system")
                else:
                    self._pending_inputs.append(user_text)
                    self.add_message("System", "Queued — waiting for current task to finish.", "system")
            else:
                self._engine_running = True
                self._chat_history_control.show_thinking = True
                self.invalidate()
                self._ensure_engine()
                from infinidev.ui.workers import run_in_background, run_engine_task
                run_in_background(self, run_engine_task, self, user_text, exclusive=True)

    # ── Command handler ──────────────────────────────────────────────

    def handle_command(self, cmd_text: str) -> None:
        """Handle /commands from chat input (delegated to command router)."""
        from infinidev.ui.handlers.commands import handle_command
        handle_command(self, cmd_text)

    # ── Settings handler ─────────────────────────────────────────────

    def _handle_settings(self, parts: list[str]) -> None:
        """Handle /settings subcommands (delegated to command router)."""
        from infinidev.ui.handlers.commands import handle_settings
        handle_settings(self, parts)

    def _handle_models(self, parts: list[str]) -> None:
        """Handle /models subcommands (delegated to command router)."""
        from infinidev.ui.handlers.commands import handle_models
        handle_models(self, parts)

    def _execute_shell_command(self, cmd: str) -> None:
        """Execute a shell command (delegated to command router)."""
        from infinidev.ui.handlers.commands import execute_shell_command
        execute_shell_command(self, cmd)

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
        self.file_manager.toggle_explorer()

    def _init_tree(self) -> None:
        self.file_manager._init_tree()

    def _on_file_selected(self, file_path: str) -> None:
        self.file_manager.open_file(file_path)

    def _open_file(self, file_path: str) -> None:
        self.file_manager.open_file(file_path)

    def _on_dirty_change(self, tab_id: str, dirty: bool) -> None:
        self.file_manager._on_dirty_change(tab_id, dirty)

    def get_explorer_content(self):
        return self.file_manager.get_explorer_content()

    def close_active_tab(self) -> None:
        self.file_manager.close_active_tab()

    def save_active_file(self) -> None:
        self.file_manager.save_active_file()

    def toggle_search_bar(self) -> None:
        self.file_manager.toggle_search_bar()

    def show_project_search(self) -> None:
        pass  # Phase 8

    def toggle_line_numbers(self) -> None:
        self.file_manager.toggle_line_numbers()

    def open_file_picker(self) -> None:
        self.file_manager.open_file_picker()

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
        # Hold-Escape to cancel running task
        if self._engine_running and self.engine is not None:
            now = time.monotonic()
            if self._cancel_hold_start is None:
                self._cancel_hold_start = now
                self._cancel_last_escape = now
                import threading as _th
                _th.Thread(target=self._cancel_hold_watcher, daemon=True,
                           name="cancel-hold").start()
            else:
                self._cancel_last_escape = now
                elapsed = now - self._cancel_hold_start
                if elapsed >= 3.0:
                    self._execute_cancel()
                    return
            self._update_cancel_bar()
            self.invalidate()

    # ── Hold-to-cancel helpers ──────────────────────────────────────

    _CANCEL_HOLD_SECONDS = 3.0
    _CANCEL_BAR_WIDTH = 6

    def _cancel_hold_watcher(self) -> None:
        """Background thread: detect key release and animate progress bar."""
        while self._cancel_hold_start is not None:
            time.sleep(0.1)
            now = time.monotonic()
            # User released Escape (no event in last 300ms)
            if now - self._cancel_last_escape > 0.3:
                self._cancel_hold_start = None
                self._update_cancel_bar()
                self.invalidate()
                return
            # 3 seconds reached — trigger cancel
            if now - self._cancel_hold_start >= self._CANCEL_HOLD_SECONDS:
                self._execute_cancel()
                return
            # Engine finished on its own while holding
            if not self._engine_running:
                self._cancel_hold_start = None
                self._update_cancel_bar()
                self.invalidate()
                return
            self._update_cancel_bar()
            self.invalidate()

    def _update_cancel_bar(self) -> None:
        """Update the status bar with the cancel progress or clear it."""
        if self.status_bar_control is None:
            return
        if self._cancel_hold_start is None:
            self.status_bar_control.set_status("")
            return
        elapsed = time.monotonic() - self._cancel_hold_start
        progress = min(1.0, elapsed / self._CANCEL_HOLD_SECONDS)
        filled = int(progress * self._CANCEL_BAR_WIDTH)
        bar = "\u2588" * filled + "\u2591" * (self._CANCEL_BAR_WIDTH - filled)
        self.status_bar_control.set_status(f"Cancelling... [{bar}]")

    def _execute_cancel(self) -> None:
        """Trigger engine cancellation and queue a response turn."""
        if self._cancel_hold_start is None and not self._engine_running:
            return
        self._cancel_hold_start = None
        self._update_cancel_bar()
        # Signal the engine to stop
        if self.engine:
            self.engine.cancel()
        # Hidden message — stays in chat_messages for context but not rendered
        self.chat_messages.append({
            "sender": "System",
            "text": "[Task cancelled by user]",
            "type": "system",
            "visible": False,
        })
        self._chat_history_control.invalidate_cache()
        # Queue a cancel-acknowledgement turn (runs after engine finishes)
        cancel_prompt = (
            "[SYSTEM] The user just cancelled the task you were working on. "
            "Acknowledge the cancellation very briefly (1-2 sentences, in the "
            "language the user has been using) and ask what they need."
        )
        self._pending_inputs.insert(0, cancel_prompt)
        self.invalidate()

    # ── Settings dialog ──────────────────────────────────────────────

    def _open_settings_dialog(self) -> None:
        self.dialog_manager.open_settings()

    def _open_findings_dialog(self, filter_type: str | None = None) -> None:
        self.dialog_manager.open_findings(filter_type=filter_type)

    def _switch_tab(self, tab_id: str) -> None:
        self.file_manager.switch_tab(tab_id)

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
        from infinidev.ui.managers.context_render import build_context_fragments
        return build_context_fragments(self._context_status, self._context_flow)

    def _usage_bar_fragments(self, label: str, used: int,
                             available: int, pct: float) -> list[tuple[str, str]]:
        # Kept as a thin shim so any external caller (tests, plugins)
        # that reached into the private method still works.
        from infinidev.ui.managers.context_render import build_usage_bar_fragments
        return build_usage_bar_fragments(label, used, available, pct)

    def get_plan_fragments(self) -> FormattedText:
        if not self._thinking_text and not self._plan_text:
            return FormattedText([(f"{TEXT_MUTED}", " Idle")])
        # Show streaming thinking first, then plan context
        fragments = []
        if self._thinking_text:
            fragments.append((f"#c0b8e0 bg:{SURFACE_LIGHT}", f" {self._thinking_text}"))
        if self._plan_text:
            if fragments:
                fragments.append((STYLE_SIDEBAR_CONTENT, "\n"))
            fragments.append((f"{TEXT_DIM} bg:{SURFACE_LIGHT}", f" {self._plan_text}"))
        return FormattedText(fragments)

    def get_steps_fragments(self) -> FormattedText:
        if not self._steps_text:
            return FormattedText([(f"{TEXT_MUTED}", " Waiting...")])
        # Parse step lines and render with styled icons
        fragments = []
        for line in self._steps_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            if line.startswith("v "):  # done
                fragments.append((f"{SUCCESS} bg:{SURFACE_LIGHT}", " \u2714 "))
                fragments.append((f"{TEXT_MUTED} bg:{SURFACE_LIGHT}", f"{line[2:]}\n"))
            elif line.startswith("> "):  # active
                fragments.append((f"{ACCENT} bold bg:{SURFACE_LIGHT}", " \u25b6 "))
                fragments.append((f"#ffffff bold bg:{SURFACE_LIGHT}", f"{line[2:]}\n"))
            elif line.startswith("o "):  # pending
                fragments.append((f"{TEXT_DIM} bg:{SURFACE_LIGHT}", " \u25cb "))
                fragments.append((f"{TEXT_DIM} bg:{SURFACE_LIGHT}", f"{line[2:]}\n"))
            else:
                fragments.append((STYLE_SIDEBAR_CONTENT, f" {line}\n"))
        return FormattedText(fragments) if fragments else FormattedText([(f"{TEXT_MUTED}", " Waiting...")])

    def get_actions_fragments(self) -> FormattedText:
        # Streaming tool detection — show animated indicator
        if self._streaming_tool_name:
            import time
            # Cycle through animation frames based on time
            frames = ["◐", "◓", "◑", "◒"]
            frame = frames[int(time.monotonic() * 4) % len(frames)]
            tokens = self._streaming_token_count
            return FormattedText([
                (f"{ACCENT} bold", f" {frame} "),
                (f"{ACCENT}", f"{self._streaming_tool_name}"),
                (f"{TEXT_MUTED}", f"\n   streaming... {tokens} tokens"),
            ])

        # Active streaming without tool detection — show token count
        if self._streaming_token_count > 0 and self._engine_running and not self._actions_text:
            import time
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = frames[int(time.monotonic() * 8) % len(frames)]
            tokens = self._streaming_token_count
            return FormattedText([
                (f"{TEXT_MUTED}", f" {frame} receiving... {tokens} tokens"),
            ])

        # Engine running but no streaming data yet — show waiting animation
        if self._engine_running and not self._actions_text:
            import time
            frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
            frame = frames[int(time.monotonic() * 6) % len(frames)]
            return FormattedText([
                (f"{TEXT_MUTED}", f" {frame} waiting for LLM..."),
            ])

        if not self._actions_text:
            return FormattedText([(f"{TEXT_MUTED}", " Idle")])
        return FormattedText([(STYLE_SIDEBAR_CONTENT, self._actions_text)])

    def get_logs_fragments(self) -> FormattedText:
        now = time.monotonic()
        # Filter entries older than 30s
        active = [(ts, text) for ts, text in self._log_entries if now - ts < 30.0]
        if not active:
            return FormattedText([(f"{TEXT_MUTED}", " No logs")])
        # Show last 5
        lines = [text for _, text in active[-5:]]
        return FormattedText([(STYLE_SIDEBAR_CONTENT, "\n".join(lines))])

    def add_log(self, text: str) -> None:
        """Add a timestamped log entry (auto-expires after 30s)."""
        self._log_entries.append((time.monotonic(), text))

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
