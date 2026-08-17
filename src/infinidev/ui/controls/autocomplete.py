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
    ("/effort", "Reasoning depth: show the levels this model accepts"),
    ("/effort high", "Set reasoning effort (levels vary per model)"),
    ("/engine", "Task engine: show or set auto|task|react|staged|graph_beta"),
    ("/engine task", "Set the task engine (auto|task|react|staged|graph_beta)"),
    ("/settings", "Show or edit settings configuration"),
    ("/settings browse", "Open settings editor modal"),
    ("/think", "Gather context deeply before next task (enables gather phase)"),
    ("/explore", "Decompose and explore a complex problem"),
    ("/brainstorm", "Brainstorm ideas and solutions for a problem"),
    ("/plan", "Generate plan, review it, then execute on approval"),
    ("/refactor", "Refactor code: modularize, clean, order, restructure"),
    ("/init", "Explore and document the current project"),
    ("/debug", "Inspect agent internals: notes, history, plan, state"),
    ("/agents", "Inspect council debates and individual agent chats"),
    ("/notes", "Show agent notes (alias for /debug)"),
    ("/findings", "Browse all findings"),
    ("/knowledge", "Browse project knowledge"),
    ("/documentation", "Browse cached library documentation"),
    ("/docs", "Browse cached library documentation (alias)"),
    ("/reindex", "Re-index the workspace (incremental)"),
    ("/reindex --full", "Drop the index and rebuild from scratch"),
    ("/mcp", "Show MCP server health (Ken semantic index, and others)"),
    ("/mcp restart", "Restart an MCP server: /mcp restart ken"),
    ("/auto", "Start autonomous chain with instructions (or show usage)"),
    ("/auto pause", "Pause the running autonomous chain"),
    ("/auto stop", "Stop autonomous mode entirely"),
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

        # Slide a window of up to 8 entries that follows selected_index so
        # the highlighted item is always on-screen even when the match list
        # is longer than the window.
        window_size = 8
        total = len(self.matches)
        start = 0
        if total > window_size and self.selected_index >= window_size:
            start = min(self.selected_index - window_size + 1, total - window_size)
        window = self.matches[start:start + window_size]

        fragments: list = []
        for offset, (cmd, desc) in enumerate(window):
            idx = start + offset

            def _click(mouse_event, idx=idx, c=cmd):
                if mouse_event.event_type == MouseEventType.MOUSE_UP:
                    self.selected_index = idx
                    if self._on_select:
                        self._on_select(c)
                        self.dismiss()

            if offset > 0:
                # Newline separates entries; none trailing the last one.
                fragments.append(("", "\n"))

            if idx == self.selected_index:
                fragments.append((f"bg:{PRIMARY} #ffffff bold", f" {cmd} ", _click))
                fragments.append((f"bg:{PRIMARY} #cccccc", f" {desc} ", _click))
            else:
                fragments.append((f"bg:{SURFACE_LIGHT} {ACCENT}", f" {cmd} ", _click))
                fragments.append((f"bg:{SURFACE_LIGHT} {TEXT_MUTED}", f" {desc} ", _click))

        return FormattedText(fragments)
