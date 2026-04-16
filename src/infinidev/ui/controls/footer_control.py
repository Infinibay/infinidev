"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.keybindings import FOOTER_HINTS, get_active_contexts


class FooterControl(FormattedTextControl):
    """Bottom bar showing keybinding hints filtered by current context."""

    def __init__(self, app_state=None) -> None:
        self._app_state = app_state
        super().__init__(self._get_text)

    def _get_text(self) -> FormattedText:
        if self._app_state is not None:
            active = get_active_contexts(self._app_state)
            hints = [
                (key, desc) for key, desc, ctx in FOOTER_HINTS
                if ctx & active  # any overlap
            ]
        else:
            # Fallback: show all
            hints = [(key, desc) for key, desc, _ in FOOTER_HINTS]

        fragments: list[tuple[str, str]] = []
        for i, (key, desc) in enumerate(hints):
            if i > 0:
                fragments.append((f"{TEXT_MUTED}", "  "))
            fragments.append((f"{PRIMARY} bold", f" {key} "))
            fragments.append((f"{TEXT_MUTED}", f" {desc}"))
        return FormattedText(fragments)
