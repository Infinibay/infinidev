"""Stdin-prompt modal.

Shown when a running subprocess (from execute_command) asks for
interactive input — sudo/ssh/gpg/passphrase prompts etc. The user
can see the command, the detected prompt, and the stdout/stderr so
far, then either send a reply (password-masked) or kill the process.

The dialog is rendered with a red "WARNING" title bar to make the
prompt visually unmissable — the worker thread is blocked waiting
for the user's decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prompt_toolkit.buffer import Buffer
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.layout.containers import (
    ConditionalContainer, Float, HSplit, VSplit, Window,
)
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.margins import ScrollbarMargin
from prompt_toolkit.layout.processors import PasswordProcessor
from prompt_toolkit.widgets import Frame

from infinidev.ui.theme import (
    ERROR, PRIMARY, SURFACE, SURFACE_DARK, TEXT, TEXT_MUTED,
)

if TYPE_CHECKING:
    from infinidev.ui.app import InfinidevApp


DIALOG_NAME = "stdin_prompt"


def _title_bar(text: str, bg: str = ERROR) -> Window:
    """Red WARNING bar at the top of the modal."""
    return Window(
        content=FormattedTextControl(lambda: [
            (f"#ffffff bg:{bg} bold", f" {text} "),
        ]),
        height=1,
    )


def _kv_line(label: str, value_fn) -> Window:
    """Compact 'label: value' row — value_fn is a callable so the
    text follows the live state (each render pulls fresh)."""
    return Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED} bold", f" {label}: "),
            (TEXT, value_fn()[:200]),
        ]),
        height=1,
    )


def _section_header(text: str) -> Window:
    return Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED} bold", f" ── {text} ── "),
        ]),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )


def _scrollable_readonly(buffer: Buffer, height: int) -> Window:
    """Read-only scrollable Window with a visible scrollbar.

    Mouse wheel + click work because the app is started with
    ``mouse_support=True``. BufferControl handles both natively.
    """
    return Window(
        content=BufferControl(buffer=buffer, focusable=True),
        height=D(preferred=height),
        wrap_lines=True,
        right_margins=[ScrollbarMargin(display_arrows=True)],
        style=f"bg:{SURFACE}",
    )


def _hint_bar() -> Window:
    return Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED}", " Enter = send · Esc = kill · Tab = focus input "),
        ]),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )


def create_stdin_prompt_dialog(app_state: "InfinidevApp") -> Float:
    """Build the stdin-prompt Float. Visible when
    ``app_state.active_dialog == DIALOG_NAME``.

    The content is driven by ``app_state._stdin_prompt_state`` (a
    dict) populated by the app's stdin handler just before it shows
    the modal. This keeps the dialog layout purely presentational.
    """

    def state() -> dict:
        return getattr(app_state, "_stdin_prompt_state", {}) or {}

    stdout_buf = Buffer(read_only=True, name="stdin-prompt-stdout")
    stderr_buf = Buffer(read_only=True, name="stdin-prompt-stderr")
    reply_buf = Buffer(name="stdin-prompt-reply", multiline=False)

    # Expose the buffers on the app so the handler can refresh them
    # when the dialog opens and read the reply when Enter is pressed.
    app_state._stdin_prompt_stdout_buf = stdout_buf
    app_state._stdin_prompt_stderr_buf = stderr_buf
    app_state._stdin_prompt_reply_buf = reply_buf

    command_row = _kv_line("Command", lambda: state().get("command", ""))
    prompt_row = _kv_line("Prompt", lambda: state().get("prompt_text", ""))

    stdout_pane = _scrollable_readonly(stdout_buf, height=8)
    stderr_pane = _scrollable_readonly(stderr_buf, height=8)

    reply_pane = Window(
        content=BufferControl(
            buffer=reply_buf,
            focusable=True,
            input_processors=[PasswordProcessor()],
        ),
        height=1,
        style=f"bg:{SURFACE_DARK}",
    )
    reply_label = Window(
        content=FormattedTextControl(lambda: [
            (f"{TEXT_MUTED} bold", " Input: "),
        ]),
        height=1,
        width=D.exact(9),
    )

    body = HSplit([
        _title_bar("WARNING"),
        Window(height=1, char=" "),
        command_row,
        prompt_row,
        Window(height=1, char=" "),
        _section_header("stdout"),
        stdout_pane,
        _section_header("stderr"),
        stderr_pane,
        Window(height=1, char=" "),
        VSplit([reply_label, reply_pane]),
        _hint_bar(),
    ], style=f"bg:{SURFACE}")

    framed = Frame(
        body=body,
        width=D(preferred=90),
        height=D(preferred=28),
        style=f"{ERROR}",
    )

    return Float(
        content=ConditionalContainer(
            content=framed,
            filter=Condition(
                lambda: app_state.active_dialog == DIALOG_NAME
            ),
        ),
        transparent=False,
    )
