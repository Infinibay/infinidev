"""Model picker dialog — select an Ollama model."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl, UIControl, UIContent

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED, ACCENT, SURFACE_LIGHT
from infinidev.ui.dialogs.base import dialog_frame

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp

DIALOG_NAME = "model_picker"


class ModelListControl(UIControl):
    """Selectable model list."""

    def __init__(self) -> None:
        self.models: list[dict[str, Any]] = []
        self.cursor: int = 0
        self.current_model: str = ""

    def set_models(self, models: list[dict], current: str) -> None:
        self.models = models
        self.current_model = current
        self.cursor = 0

    def move_cursor(self, delta: int) -> None:
        if self.models:
            self.cursor = max(0, min(len(self.models) - 1, self.cursor + delta))

    def get_selected(self) -> str | None:
        if self.models and 0 <= self.cursor < len(self.models):
            return self.models[self.cursor].get("name", "")
        return None

    def create_content(self, width: int, height: int | None,
                       preview_search: bool = False) -> UIContent:
        lines = []
        for i, model in enumerate(self.models):
            name = model.get("name", "?")
            size = model.get("size", 0)
            size_gb = size / (1024 ** 3) if size else 0
            is_current = name == self.current_model
            marker = " *" if is_current else "  "

            if i == self.cursor:
                style = f"bg:{PRIMARY} #ffffff bold"
            elif is_current:
                style = f"{ACCENT}"
            else:
                style = f"{TEXT}"

            lines.append([(style, f" {marker} {name} ({size_gb:.1f} GB)")])

        if not lines:
            lines = [[(f"{TEXT_MUTED}", " No models found")]]

        def get_line(i):
            return lines[i] if 0 <= i < len(lines) else []

        return UIContent(get_line=get_line, line_count=len(lines))


def create_model_picker():
    """Create the model picker dialog."""
    list_control = ModelListControl()

    body = HSplit([
        Window(content=list_control, height=20),
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT_MUTED}", " Enter = select | Esc = cancel | * = current"),
            ]),
            height=1,
        ),
    ])

    frame = dialog_frame("Select a Model", body, width=60, height=24, border_color=PRIMARY)
    return frame, list_control
