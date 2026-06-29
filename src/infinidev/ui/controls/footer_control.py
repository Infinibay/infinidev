"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.utils import get_cwidth

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.keybindings import FOOTER_HINTS, get_active_contexts
from infinidev.ui.controls._widthutil import terminal_cols


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

        cols = terminal_cols()
        fragments: list[tuple[str, str]] = []
        used = 0
        for i, (key, desc) in enumerate(hints):
            sep = "  " if i > 0 else ""
            key_seg = f" {key} "
            desc_seg = f" {desc}"
            seg_w = get_cwidth(sep) + get_cwidth(key_seg) + get_cwidth(desc_seg)
            # Drop trailing hints that do not fit rather than clip mid-word.
            if cols is not None and used + seg_w > cols:
                break
            if sep:
                fragments.append((f"{TEXT_MUTED}", sep))
            fragments.append((f"{PRIMARY} bold", key_seg))
            fragments.append((f"{TEXT_MUTED}", desc_seg))
            used += seg_w
        return FormattedText(fragments)
