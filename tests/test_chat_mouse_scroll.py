"""Tests for deterministic mouse-wheel scrolling on ChatHistoryControl.

Each wheel event moves exactly five transcript lines. Event timing, transcript
size, and render latency must not add acceleration or inertial movement.
"""

from __future__ import annotations

from prompt_toolkit.data_structures import Point
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType

import infinidev.ui.controls.chat_history as ch
from infinidev.ui.controls.chat_history import ChatHistoryControl


def _make(nlines: int = 500) -> ChatHistoryControl:
    msgs = [
        {"type": "agent", "sender": "A", "text": f"m{i}"}
        for i in range(nlines)
    ]
    control = ChatHistoryControl(msgs)
    control._line_cache = None
    control._last_rebuild = 0.0
    control.create_content(width=80, height=24)
    return control


def test_wheel_up_moves_exactly_five_lines_per_event():
    control = _make()

    control.move_cursor_up()
    assert control._scroll_offset == ch._MOUSE_WHEEL_LINES == 5

    control.move_cursor_up()
    assert control._scroll_offset == 10


def test_rapid_wheel_events_do_not_accelerate():
    control = _make()

    for _ in range(10):
        control.move_cursor_up()

    assert control._scroll_offset == 10 * ch._MOUSE_WHEEL_LINES


def test_short_and_long_transcripts_use_the_same_step():
    short = _make(100)
    long = _make(2_000)

    short.move_cursor_up()
    long.move_cursor_up()

    assert short._scroll_offset == long._scroll_offset == ch._MOUSE_WHEEL_LINES


def test_wheel_down_moves_exactly_five_lines_and_follows_tail_at_bottom():
    control = _make()
    control._follow_tail = False
    control._scroll_offset = 12

    control.move_cursor_down()
    assert control._scroll_offset == 7
    assert control._follow_tail is False

    control.move_cursor_down()
    assert control._scroll_offset == 2

    control.move_cursor_down()
    assert control._scroll_offset == 0
    assert control._follow_tail is True


def test_wheel_step_clamps_at_top():
    control = _make()
    max_offset = control._line_count - 1
    control._follow_tail = False
    control._scroll_offset = max_offset - 2

    control.move_cursor_up()

    assert control._scroll_offset == max_offset
    assert control._follow_tail is False


def test_page_navigation_keeps_its_larger_fixed_step():
    control = _make()

    control.page_up()
    assert control._scroll_offset == 15

    control.page_down()
    assert control._scroll_offset == 0
    assert control._follow_tail is True


class _FakeWin:
    def __init__(self, height: int) -> None:
        self.render_info = type("RI", (), {"window_height": height})()


def _vertical_scroll(
    control: ChatHistoryControl,
    height: int,
    *,
    line_count: int,
    follow_tail: bool,
    offset: int,
) -> int:
    control._line_count = line_count
    control._follow_tail = follow_tail
    control._scroll_offset = offset
    return control.get_vertical_scroll(_FakeWin(height))


def test_get_vertical_scroll_follow_tail_shows_bottom():
    control = _make()
    assert _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=True,
        offset=0,
    ) == 180


def test_get_vertical_scroll_maps_offset_one_to_one():
    control = _make()
    base = _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=False,
        offset=0,
    )
    assert base == 180
    assert _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=False,
        offset=10,
    ) == 170
    assert _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=False,
        offset=50,
    ) == 130
    assert _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=False,
        offset=180,
    ) == 0


def test_get_vertical_scroll_clamps_and_handles_short_content():
    control = _make()
    assert _vertical_scroll(
        control,
        50,
        line_count=20,
        follow_tail=False,
        offset=0,
    ) == 0
    assert _vertical_scroll(
        control,
        20,
        line_count=200,
        follow_tail=False,
        offset=9_999,
    ) == 0

    control._line_count = 200
    control._follow_tail = False
    control._scroll_offset = 5
    window = type("W", (), {"render_info": None})()
    assert control.get_vertical_scroll(window) == 0


def test_get_vertical_scroll_keeps_cursor_on_bottom_edge():
    control = _make()
    height, line_count, offset = 20, 200, 30
    vertical_scroll = _vertical_scroll(
        control,
        height,
        line_count=line_count,
        follow_tail=False,
        offset=offset,
    )
    cursor_y = line_count - 1 - offset
    assert cursor_y - vertical_scroll == height - 1


def _wheel_event(up: bool) -> MouseEvent:
    return MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_UP if up else MouseEventType.SCROLL_DOWN,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )


def test_mouse_handler_consumes_scroll_up_and_moves_exactly_five_lines():
    control = _make()

    result = control.mouse_handler(_wheel_event(up=True))

    assert result is None
    assert control._scroll_offset == ch._MOUSE_WHEEL_LINES
    assert control._follow_tail is False


def test_mouse_handler_consumes_scroll_down_and_moves_exactly_five_lines():
    control = _make()
    control._follow_tail = False
    control._scroll_offset = 40

    result = control.mouse_handler(_wheel_event(up=False))

    assert result is None
    assert control._scroll_offset == 35


def _render_window():
    """Build a real Window around ChatHistoryControl and a render callback."""
    import pytest
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.layout.containers import HSplit, Window
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.layout.screen import Screen, WritePosition
    from prompt_toolkit.output import DummyOutput

    pytest.importorskip("prompt_toolkit")
    messages = [
        {"type": "agent", "sender": "A", "text": f"line {i}"}
        for i in range(200)
    ]
    control = ChatHistoryControl(messages)
    window = Window(
        content=control,
        wrap_lines=False,
        get_vertical_scroll=control.get_vertical_scroll,
    )
    app = Application(
        layout=Layout(HSplit([window])),
        output=DummyOutput(),
        input=DummyInput(),
    )
    height, width = 20, 80

    def render() -> int:
        with set_app(app):
            window.write_to_screen(
                Screen(),
                MouseHandlers(),
                WritePosition(0, 0, width, height),
                "",
                False,
                None,
            )
        return window.vertical_scroll

    return app, window, control, render, set_app


def test_real_window_moves_exactly_five_lines_per_wheel_event():
    app, window, control, render, set_app = _render_window()

    with set_app(app):
        start = render()
        assert start > 0

        seen = []
        for _ in range(6):
            control.mouse_handler(_wheel_event(up=True))
            seen.append(render())

        expected = [
            start - ch._MOUSE_WHEEL_LINES * event_count
            for event_count in range(1, 7)
        ]
        assert seen == expected

        for _ in range(6):
            control.mouse_handler(_wheel_event(up=False))

        assert render() == start
        assert control._follow_tail is True
