"""Tab bar control for switching between chat and open files.

Renders a single-line tab bar with active tab highlighting and dirty indicators.
Tab switching is handled by the main app via callbacks.
"""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.formatted_text import FormattedText

from infinidev.ui.theme import PRIMARY, TEXT_MUTED, SURFACE_LIGHT, WARNING


class TabBar:
    """Manages tab state and renders the tab bar."""

    def __init__(self, on_tab_select: Callable[[str], None] | None = None) -> None:
        self._tabs: list[tuple[str, str]] = [("chat", "Chat")]  # (tab_id, display_name)
        self._active: str = "chat"
        self._dirty: set[str] = set()
        self._on_tab_select = on_tab_select

    @property
    def active_tab(self) -> str:
        return self._active

    @active_tab.setter
    def active_tab(self, tab_id: str) -> None:
        self._active = tab_id
        if self._on_tab_select:
            self._on_tab_select(tab_id)

    def add_tab(self, tab_id: str, name: str) -> None:
        """Add a new tab. If it already exists, just activate it."""
        for tid, _ in self._tabs:
            if tid == tab_id:
                self._active = tab_id
                return
        self._tabs.append((tab_id, name))
        self._active = tab_id

    def remove_tab(self, tab_id: str) -> None:
        """Remove a tab. Falls back to chat if active tab is removed."""
        self._tabs = [(tid, name) for tid, name in self._tabs if tid != tab_id]
        self._dirty.discard(tab_id)
        if self._active == tab_id:
            self._active = self._tabs[-1][0] if self._tabs else "chat"

    def set_dirty(self, tab_id: str, dirty: bool) -> None:
        if dirty:
            self._dirty.add(tab_id)
        else:
            self._dirty.discard(tab_id)

    def get_fragments(self) -> FormattedText:
        """Render the tab bar as FormattedText."""
        fragments: list[tuple[str, str]] = []

        for tab_id, name in self._tabs:
            dirty = "* " if tab_id in self._dirty else ""
            if tab_id == self._active:
                fragments.append((f"bg:{PRIMARY} #ffffff bold", f" {dirty}{name} "))
            else:
                fragments.append((f"{TEXT_MUTED} bg:{SURFACE_LIGHT}", f" {dirty}{name} "))
            fragments.append(("", " "))

        return FormattedText(fragments)

    def has_tab(self, tab_id: str) -> bool:
        return any(tid == tab_id for tid, _ in self._tabs)

    def get_tab_ids(self) -> list[str]:
        return [tid for tid, _ in self._tabs]
