"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.keybindings import FOOTER_HINTS


from infinidev.ui.controls.status_bar_control import StatusBarControl
from infinidev.ui.controls.footer_control import FooterControl
