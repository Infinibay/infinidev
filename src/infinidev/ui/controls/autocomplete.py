"""Autocomplete for /commands in the chat input.

Provides both a prompt_toolkit Completer (for inline completion) and a
standalone filtered list renderer for the autocomplete overlay.
"""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.formatted_text import FormattedText

from infinidev.ui.theme import TEXT, TEXT_MUTED, PRIMARY, ACCENT, SURFACE_LIGHT

# Commands list — imported from tui.py's COMMANDS
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
    ("/plan", "Generate plan, review it, then execute on approval"),
    ("/refactor", "Refactor code: modularize, clean, order, restructure"),
    ("/init", "Explore and document the current project"),
    ("/debug", "Inspect agent internals: notes, history, plan, state"),
    ("/notes", "Show agent notes (alias for /debug)"),
    ("/findings", "Browse all findings"),
    ("/knowledge", "Browse project knowledge"),
    ("/documentation", "Browse cached library documentation"),
    ("/docs", "Browse cached library documentation (alias)"),
    ("/clear", "Clear chat history"),
    ("/help", "Show this help"),
    ("/exit", "Exit the CLI"),
    ("/quit", "Exit the CLI"),
]


class AutocompleteState:
    """Manages the autocomplete overlay state and rendering."""

    def __init__(self, on_select: Callable[[str], None] | None = None) -> None:
        self.visible: bool = False
        self.matches: list[tuple[str, str]] = []  # (cmd, desc)
        self.selected_index: int = 0
        self._on_select = on_select

    def update(self, text: str) -> None:
        """Update matches based on current input text."""
        text = text.lstrip()
        if text.startswith("/"):
            self.matches = [
                (cmd, desc) for cmd, desc in COMMANDS
                if cmd.startswith(text)
            ]
            self.visible = len(self.matches) > 0
            self.selected_index = 0
        else:
            self.visible = False
            self.matches = []

    def dismiss(self) -> None:
        self.visible = False
        self.matches = []
        self.selected_index = 0

    def select_next(self) -> None:
        if self.matches:
            self.selected_index = (self.selected_index + 1) % len(self.matches)

    def select_prev(self) -> None:
        if self.matches:
            self.selected_index = (self.selected_index - 1) % len(self.matches)

    def get_selected_command(self) -> str | None:
        if self.matches and 0 <= self.selected_index < len(self.matches):
            return self.matches[self.selected_index][0]
        return None

    def apply_selected(self) -> None:
        """Apply the selected command via callback."""
        cmd = self.get_selected_command()
        if cmd and self._on_select:
            self._on_select(cmd)
            self.dismiss()

    def get_fragments(self) -> FormattedText:
        """Render the autocomplete overlay as FormattedText with click support."""
        from prompt_toolkit.mouse_events import MouseEventType

        if not self.visible or not self.matches:
            return FormattedText([])

        fragments: list = []
        for i, (cmd, desc) in enumerate(self.matches[:8]):  # max 8 visible
            def _click(mouse_event, idx=i, c=cmd):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self.selected_index = idx
                    if self._on_select:
                        self._on_select(c)
                        self.dismiss()

            if i == self.selected_index:
                fragments.append((f"bg:{PRIMARY} #ffffff bold", f" {cmd} ", _click))
                fragments.append((f"bg:{PRIMARY} #cccccc", f" {desc} ", _click))
            else:
                fragments.append((f"bg:{SURFACE_LIGHT} {ACCENT}", f" {cmd} ", _click))
                fragments.append((f"bg:{SURFACE_LIGHT} {TEXT_MUTED}", f" {desc} ", _click))
            fragments.append(("", "\n"))

        return FormattedText(fragments)
