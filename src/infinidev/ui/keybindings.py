"""Global keybindings for the Infinidev TUI.

All bindings are registered here and merged into the Application.
Control-specific bindings (e.g., chat input Enter) live in their
respective control modules and are merged separately.
"""

from __future__ import annotations

from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


def create_global_keybindings(app_state) -> KeyBindings:
    """Create the global key bindings registry.

    Args:
        app_state: The InfinidevApp instance (or its state object)
                   so handlers can call methods on it.
    """
    kb = KeyBindings()

    @kb.add("c-c")
    def quit_(event):
        """Exit the application."""
        app_state.request_quit(event)

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

    @kb.add("escape")
    def cancel_task(event):
        """Cancel the currently running task, or dismiss active dialog."""
        app_state.handle_escape()

    return kb


# ── Keybinding hints for the footer ────────────────────────────────────

FOOTER_HINTS = [
    ("Ctrl+C", "Exit"),
    ("Ctrl+O", "Open file"),
    ("Ctrl+E", "Explorer"),
    ("F6", "Line #"),
    ("Ctrl+W", "Close tab"),
    ("Ctrl+S", "Save"),
    ("Ctrl+F", "Find"),
    ("F2", "Chat"),
    ("Esc", "Stop task"),
]
