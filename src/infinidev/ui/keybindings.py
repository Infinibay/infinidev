"""Global keybindings for the Infinidev TUI.

All bindings are registered here and merged into the Application.
Control-specific bindings (e.g., chat input Enter) live in their
respective control modules and are merged separately.
"""

from __future__ import annotations

import time

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


# ── Ctrl+C multi-press state ──────────────────────────────────────────
_cc_stage: int = 0
_cc_last: float = 0.0
_CC_TIMEOUT: float = 2.0  # seconds before the sequence resets


def create_global_keybindings(app_state) -> KeyBindings:
    """Create the global key bindings registry.

    Args:
        app_state: The InfinidevApp instance (or its state object)
                   so handlers can call methods on it.
    """
    kb = KeyBindings()

    # Reset the Ctrl+C counter whenever the user types in chat input.
    def _reset_cc_on_input(_b) -> None:
        global _cc_stage
        _cc_stage = 0

    try:
        app_state._chat_buffer.on_text_changed += _reset_cc_on_input
    except Exception:
        pass
    @kb.add("c-c")
    def quit_(event):
        """Three-stage Ctrl+C:

        *If text is selected in the buffer, copy it instead of quitting.*
        Otherwise:
        1st press — if chat input has text, clear it.
        2nd press — show 'Press Ctrl+C again to quit'.
        3rd press — actually quit.
        Resets after 2 s or when the user types.
        """
        global _cc_stage, _cc_last
        now = time.monotonic()

        # Reset stage if too much time has passed
        if now - _cc_last > _CC_TIMEOUT:
            _cc_stage = 0

        # ── Fallback: if the terminal lumped Ctrl+Shift+C into Ctrl+C,
        #    copy the selection when present instead of clearing/quitting.
        buf = app_state._chat_buffer
        focused = event.app.current_buffer
        if focused and focused.selection_state:
            start = min(focused.selection_state.original_cursor_position,
                        focused.cursor_position)
            end = max(focused.selection_state.original_cursor_position,
                      focused.cursor_position)
            selected_text = focused.text[start:end]
            if selected_text:
                from infinidev.ui.clipboard import copy_to_clipboard
                ok = copy_to_clipboard(selected_text)
                app_state.flash_status("Copied to clipboard" if ok else "Copy failed")
                return

        if _cc_stage == 0:
            if buf.text:
                # Stage 0 → 1: clear the input
                from prompt_toolkit.document import Document
                buf.set_document(Document(""), bypass_readonly=True)
                app_state.flash_status("Input cleared")
                _cc_stage = 1
            else:
                # Input already empty, skip straight to hint
                app_state.flash_status("Press Ctrl+C again to quit")
                _cc_stage = 2
        elif _cc_stage == 1:
            # Stage 1 → 2: show quit hint
            app_state.flash_status("Press Ctrl+C again to quit")
            _cc_stage = 2
        else:
            # Stage 2 → quit
            _cc_stage = 0
            app_state.request_quit(event)

        _cc_last = now
    @kb.add("c-e")
    def toggle_explorer(event):
        """Toggle the file explorer panel."""
        app_state.toggle_explorer()

    @kb.add("c-w")
    def close_tab(event):
        """Close the active file tab."""
        app_state.close_active_tab()

    @kb.add("c-s")
    def save_file(event):
        """Save the active file."""
        app_state.save_active_file()

    @kb.add("c-f")
    def find_in_file(event):
        """Open in-file search bar."""
        app_state.toggle_search_bar()

    @kb.add("c-g")  # Ctrl+Shift+F isn't reliable in all terminals; use Ctrl+G
    def find_in_project(event):
        """Open project-wide search dialog."""
        app_state.show_project_search()

    @kb.add("c-l", eager=True)
    @kb.add("f6")
    def toggle_line_numbers(event):
        """Toggle line numbers in file editors."""
        app_state.toggle_line_numbers()

    @kb.add("c-p", eager=True)
    @kb.add("c-o", eager=True)
    def quick_open(event):
        """Open the quick file picker."""
        app_state.open_file_picker()

    @kb.add("f2")
    def focus_chat(event):
        """Move focus to the chat input."""
        app_state.focus_chat()

    @kb.add("f3")
    def focus_explorer(event):
        """Move focus to the file explorer."""
        app_state.focus_explorer()

    @kb.add("f4")
    def focus_sidebar(event):
        """Move focus to the sidebar."""
        app_state.focus_sidebar()

    @kb.add("c-y")
    def copy_last_agent(event):
        """Copy the last agent message to the system clipboard."""
        app_state.copy_last_agent_message()

    @kb.add("escape", "y")
    def toggle_select_mode(event):
        """Enter/exit message selection mode to pick a message to copy."""
        app_state.toggle_select_mode()

    @kb.add("escape")
    def cancel_task(event):
        """Cancel the currently running task, dismiss dialog, or exit select mode."""
        if app_state._chat_history_control.select_mode:
            app_state._chat_history_control.exit_select_mode()
            app_state.flash_status("")
            app_state.invalidate()
            return
        app_state.handle_escape()

    return kb


# ── Keybinding hints for the footer ────────────────────────────────────

FOOTER_HINTS = [
    ("Ctrl+C", "Clear/Quit"),
    ("Ctrl+O", "Open file"),
    ("Ctrl+E", "Explorer"),
    ("F6", "Line #"),
    ("Ctrl+W", "Close tab"),
    ("Ctrl+S", "Save"),
    ("Ctrl+F", "Find"),
    ("Ctrl+Y", "Copy msg"),
    ("Esc Y", "Select msg"),
    ("F2", "Chat"),
    ("Esc", "Stop task"),
]
