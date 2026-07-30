"""Two ways the transcript stopped responding to what the user did.

Both were intermittent, and both for the same reason: a piece of state was
correct in the common case and silently wrong in the other one. The existing
tests missed them by reaching past the real path — calling the click callback
directly, and rendering a message dict written by hand.
"""

from __future__ import annotations

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

from infinidev.ui.controls.chat_history import ChatHistoryControl
from infinidev.ui.controls.message_widgets import get_widget


# ── clicking the tool group ──────────────────────────────────────────────


def _transcript(n_replies: int, height: int):
    messages = [
        {"type": "agent", "sender": "A", "text": f"reply {i}"}
        for i in range(n_replies)
    ]
    messages += [
        {"type": "tool_call", "tool": "read_file",
         "args": {"file_path": f"a{i}.py"}, "result": "ok"}
        for i in range(4)
    ]
    control = ChatHistoryControl(messages)
    control._line_cache = None
    control._last_rebuild = 0.0
    return control, control.create_content(width=80, height=height)


def _row_of_group_header(content) -> int:
    for i in range(content.line_count):
        if "tools" in "".join(text for _, text in content.get_line(i)):
            return i
    raise AssertionError("no tool-group header rendered")


def _click(control: ChatHistoryControl, row: int) -> None:
    control.mouse_handler(MouseEvent(
        position=Point(x=5, y=row),
        event_type=MouseEventType.MOUSE_UP,
        button=MouseButton.LEFT,
        modifiers=frozenset(),
    ))


@pytest.mark.parametrize(
    "n_replies,height,expect_pad",
    [(1, 40, True), (12, 30, True), (40, 24, False)],
    ids=["short-transcript", "just-under", "taller-than-terminal"],
)
def test_clicking_the_group_header_toggles_it(n_replies, height, expect_pad):
    """The click has to land wherever the header was actually drawn.

    ``create_content`` prepends blank rows to bottom-anchor a transcript
    shorter than the terminal, and prompt_toolkit reports the click in those
    padded content coordinates — while ``_clickable_lines`` is keyed without
    the padding. The group opened only once the conversation grew past the
    viewport and the padding went to zero.
    """
    control, content = _transcript(n_replies, height)
    assert (control._top_pad > 0) is expect_pad

    row = _row_of_group_header(content)
    _click(control, row)
    assert control._tool_group_states, "the click did not reach the callback"


def test_clicking_a_blank_padding_row_does_nothing():
    """The offset correction must not make padding rows clickable."""
    control, _ = _transcript(1, 40)
    _click(control, 0)
    assert control._tool_group_states == {}


# ── the streaming flag that suppresses markdown ──────────────────────────


@pytest.fixture
def markdown_on(monkeypatch):
    """Pin the setting these tests are about.

    ``MARKDOWN_MESSAGES`` is a user setting, so leaving it to the machine
    makes the assertion pass or fail according to whoever's settings.json is
    around — and any test that touches settings changes the answer mid-suite.
    """
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "MARKDOWN_MESSAGES", True)


def _rendered(msg: dict) -> str:
    widget = get_widget(msg.get("type", "agent"))
    return "\n".join(
        "".join(text for _, text in line)
        for line in widget.render(msg, width=60).lines
    )


def test_markdown_source_is_hidden_once_a_stream_closes(markdown_on):
    styled = {"type": "agent", "sender": "A",
              "text": "esto es **negrita**", "streaming": False}
    assert "**" not in _rendered(styled)


def test_a_message_after_the_stream_does_not_strand_the_flag(markdown_on):
    """The reported symptom: replies rendering their own markdown source.

    ``finalize_streaming_message`` only inspected the last message, so a
    tool result or a system notice arriving between the final chunk and the
    stream-end call left ``streaming`` True — and with it, ``**bold``
    visible for the rest of the session.
    """
    from infinidev.ui.app import InfinidevApp

    app = object.__new__(InfinidevApp)
    app.chat_messages = []

    class _Control:
        def invalidate_cache(self): pass

    app._chat_history_control = _Control()
    app.invalidate = lambda: None

    app.append_to_last_message("A", "esto es **negrita**", "agent")
    app.add_message("System", "· ran a tool", "system")     # lands in between
    app.finalize_streaming_message("A", "agent")

    reply = app.chat_messages[0]
    assert reply["streaming"] is False
    assert "**" not in _rendered(reply)


def test_only_the_last_message_can_be_mid_stream():
    """The invariant that makes the fix hold regardless of call order."""
    from infinidev.ui.app import InfinidevApp

    app = object.__new__(InfinidevApp)
    app.chat_messages = []

    class _Control:
        def invalidate_cache(self): pass

    app._chat_history_control = _Control()
    app.invalidate = lambda: None

    app.append_to_last_message("A", "first", "agent")
    app.append_to_last_message("B", "second", "agent")   # a different speaker

    assert app.chat_messages[0]["streaming"] is False
    assert app.chat_messages[1]["streaming"] is True
