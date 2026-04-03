"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.keybindings import FOOTER_HINTS


class StatusBarControl(FormattedTextControl):
    """Single-line status bar showing model, project, and status info."""

    def __init__(self) -> None:
        self._model = "unknown"
        self._project = ""
        self._status = ""
        super().__init__(self._get_text)

    def _get_text(self) -> FormattedText:
        fragments: list[tuple[str, str]] = []
        fragments.append((f"{PRIMARY} bold", " infinidev "))
        fragments.append((f"{TEXT_MUTED}", " │ "))
        fragments.append((f"{TEXT}", self._model))
        if self._project:
            fragments.append((f"{TEXT_MUTED}", " │ "))
            fragments.append((f"{TEXT}", self._project))
        if self._status:
            fragments.append((f"{TEXT_MUTED}", " │ "))
            fragments.append((f"{ACCENT}", self._status))
        return FormattedText(fragments)

    def set_model(self, model: str) -> None:
        self._model = model

    def set_project(self, project: str) -> None:
        self._project = project

    def set_status(self, status: str) -> None:
        self._status = status


