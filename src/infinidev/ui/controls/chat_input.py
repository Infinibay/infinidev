"""Chat input control — Buffer with Enter to submit, history, and autocomplete.

Replaces the Textual ChatInput(TextArea) with a prompt_toolkit Buffer that
has custom key bindings for Enter/Shift+Enter/Up/Down history navigation.
"""

from __future__ import annotations

from typing import Callable

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.filters import Condition


def create_chat_input(
    on_submit: Callable[[str], None],
    on_text_changed: Callable[[str], None] | None = None,
    is_autocomplete_visible: Callable[[], bool] | None = None,
    on_autocomplete_apply: Callable[[], None] | None = None,
    on_autocomplete_next: Callable[[], None] | None = None,
    on_autocomplete_prev: Callable[[], None] | None = None,
    on_autocomplete_dismiss: Callable[[], None] | None = None,
    chat_history_control=None,
) -> tuple[Buffer, BufferControl, KeyBindings]:
    """Create a chat input Buffer with custom keybindings.

    Args:
        on_submit: Called with the message text when Enter is pressed.
        is_autocomplete_visible: Optional callable that returns True when the
            autocomplete menu is visible (Tab should focus it).

    Returns:
        (buffer, control, keybindings) — the buffer for direct text access,
        the control for embedding in a Window, and the keybindings to merge.
    """
    # Message history
    msg_history: list[str] = []
    history_index: list[int] = [0]  # mutable ref
    draft: list[str] = [""]

    def _on_buf_changed(b: Buffer) -> None:
        if on_text_changed:
            on_text_changed(b.text)

    buf = Buffer(
        multiline=True,
        name="chat-input",
        on_text_changed=_on_buf_changed,
    )

    kb = KeyBindings()

    @kb.add("enter", eager=True)
    def submit(event):
        """Submit the message on Enter (not Shift+Enter)."""
        text = buf.text.strip()
        if text:
            msg_history.append(buf.text)
            history_index[0] = len(msg_history)
            draft[0] = ""
            on_submit(buf.text)
            buf.reset(Document(""))

    @kb.add("escape", "enter", eager=True)
    def newline(event):
        """Insert a literal newline on Escape+Enter (Shift+Enter alternative)."""
        buf.insert_text("\n")

    @kb.add("up")
    def history_prev(event):
        """Navigate autocomplete or previous history entry."""
        if is_autocomplete_visible and is_autocomplete_visible():
            if on_autocomplete_prev:
                on_autocomplete_prev()
            return
        # Only intercept when cursor is on the first line
        row = buf.document.cursor_position_row
        if row == 0 and msg_history:
            if history_index[0] > 0:
                if history_index[0] == len(msg_history):
                    draft[0] = buf.text
                history_index[0] -= 1
                buf.set_document(
                    Document(msg_history[history_index[0]]),
                    bypass_readonly=True,
                )
        else:
            # Normal cursor up
            buf.cursor_up()

    @kb.add("down")
    def history_next(event):
        """Navigate autocomplete or next history entry."""
        if is_autocomplete_visible and is_autocomplete_visible():
            if on_autocomplete_next:
                on_autocomplete_next()
            return
        row = buf.document.cursor_position_row
        line_count = buf.document.line_count
        if row == line_count - 1 and msg_history:
            if history_index[0] < len(msg_history) - 1:
                history_index[0] += 1
                buf.set_document(
                    Document(msg_history[history_index[0]]),
                    bypass_readonly=True,
                )
            elif history_index[0] == len(msg_history) - 1:
                history_index[0] = len(msg_history)
                buf.set_document(
                    Document(draft[0]),
                    bypass_readonly=True,
                )
        else:
            buf.cursor_down()

    @kb.add("tab")
    def apply_autocomplete(event):
        """Apply selected autocomplete item."""
        if is_autocomplete_visible and is_autocomplete_visible():
            if on_autocomplete_apply:
                on_autocomplete_apply()
        else:
            buf.insert_text("    ")  # normal tab = 4 spaces

    @kb.add("escape")
    def dismiss_autocomplete(event):
        """Dismiss autocomplete menu."""
        if on_autocomplete_dismiss:
            on_autocomplete_dismiss()

    @kb.add("pageup")
    def page_up(event):
        """Scroll chat history up."""
        if chat_history_control:
            for _ in range(15):
                chat_history_control.move_cursor_up()

    @kb.add("pagedown")
    def page_down(event):
        """Scroll chat history down."""
        if chat_history_control:
            for _ in range(15):
                chat_history_control.move_cursor_down()

    control = BufferControl(
        buffer=buf,
        key_bindings=kb,
        focusable=True,
        focus_on_click=True,
    )

    return buf, control, kb
