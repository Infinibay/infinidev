"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.keybindings import FOOTER_HINTS


class FooterControl(FormattedTextControl):
    """Bottom bar showing keybinding hints. Computed once (static content)."""

    def __init__(self) -> None:
        # Build once — footer never changes
        fragments: list[tuple[str, str]] = []
        for i, (key, desc) in enumerate(FOOTER_HINTS):
            if i > 0:
                fragments.append((f"{TEXT_MUTED}", "  "))
            fragments.append((f"{PRIMARY} bold", f" {key} "))
            fragments.append((f"{TEXT_MUTED}", f" {desc}"))
        self._cached = FormattedText(fragments)
        super().__init__(lambda: self._cached)

