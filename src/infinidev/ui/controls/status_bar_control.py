"""Status bar control for the bottom of the TUI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.utils import get_cwidth

from infinidev.ui.theme import (
    STYLE_STATUS_BAR, TEXT_MUTED, PRIMARY, TEXT, ACCENT,
)
from infinidev.ui.controls._widthutil import terminal_cols, truncate_cells


class StatusBarControl(FormattedTextControl):
    """Single-line status bar showing model, project, and status info."""

    def __init__(self) -> None:
        self._model = "unknown"
        self._project = ""
        self._status = ""
        super().__init__(self._get_text)

    def _get_text(self) -> FormattedText:
        cols = terminal_cols()
        if cols is None:
            # No running app (e.g. tests) — render raw, unclamped.
            return self._build(self._model, self._project, self._status)

        # Clamp to the terminal width without ever dropping the live status
        # (it carries the cancel/progress indicator). Allocate widths in
        # priority order — status > model > project — so the project path
        # is truncated first, then the model name.
        sep_w = get_cwidth(" │ ")
        remaining = max(0, cols - get_cwidth(" infinidev "))

        def _alloc(text: str) -> str:
            nonlocal remaining
            if not text:
                return ""
            if remaining <= sep_w:
                return ""  # not even room for the separator → drop segment
            avail = remaining - sep_w
            if get_cwidth(text) <= avail:
                remaining -= sep_w + get_cwidth(text)
                return text
            remaining = 0
            return truncate_cells(text, avail)

        status_shown = _alloc(self._status)
        model_shown = _alloc(self._model)
        project_shown = _alloc(self._project)
        return self._build(model_shown, project_shown, status_shown)

    def _build(self, model: str, project: str, status: str) -> FormattedText:
        """Assemble the visual fragments (left→right: model, project, status)."""
        fragments: list[tuple[str, str]] = []
        fragments.append((f"{PRIMARY} bold", " infinidev "))
        if model:
            fragments.append((f"{TEXT_MUTED}", " │ "))
            fragments.append((f"{TEXT}", model))
        if project:
            fragments.append((f"{TEXT_MUTED}", " │ "))
            fragments.append((f"{TEXT}", project))
        if status:
            fragments.append((f"{TEXT_MUTED}", " │ "))
            fragments.append((f"{ACCENT}", status))
        return FormattedText(fragments)

    def set_model(self, model: str) -> None:
        self._model = model

    def set_project(self, project: str) -> None:
        self._project = project

    def set_status(self, status: str) -> None:
        self._status = status


