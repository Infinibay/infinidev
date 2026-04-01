"""Single setting value editor dialog."""

from __future__ import annotations

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl

from infinidev.ui.theme import PRIMARY, TEXT, TEXT_MUTED
from infinidev.ui.dialogs.base import dialog_frame

DIALOG_NAME = "setting_value_editor"


class SettingValueEditorState:
    """State for the setting value editor."""

    def __init__(self) -> None:
        self.key: str = ""
        self.description: str = ""
        self.current_value: str = ""
        self.value_type: str = "text"  # text, bool, select
        self.choices: list[str] = []

        self.buffer = Buffer(name="setting-value", multiline=True)

    def open(self, key: str, description: str, current_value: str,
             value_type: str = "text", choices: list[str] | None = None) -> None:
        self.key = key
        self.description = description
        self.current_value = str(current_value)
        self.value_type = value_type
        self.choices = choices or []
        self.buffer.set_document(Document(self.current_value), bypass_readonly=True)

    def get_value(self) -> str:
        return self.buffer.text


def create_setting_editor():
    """Create the setting value editor dialog."""
    state = SettingValueEditorState()

    body = HSplit([
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT} bold", f" {state.key}\n"),
                (f"{TEXT_MUTED}", f" {state.description}\n"),
            ]),
            height=3,
        ),
        Window(
            content=BufferControl(buffer=state.buffer, focusable=True),
            height=5,
        ),
        Window(
            content=FormattedTextControl(lambda: [
                (f"{TEXT_MUTED}", " Enter = save | Esc = cancel"),
            ]),
            height=1,
        ),
    ])

    frame = dialog_frame("Edit Setting", body, width=60, height=12, border_color=PRIMARY)
    return frame, state
