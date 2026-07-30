"""Tests for the mac-style momentum (inertial) scroll on ChatHistoryControl.

A mouse-wheel notch injects an impulse into a velocity model; an animation
loop applies the velocity and decays it by friction. Rapid notches
accumulate → higher velocity → a longer glide ("the faster you spin, the
more force"). Deterministic jumps (page/home/end) cancel the glide.

These tests drive the model synchronously: there is no prompt_toolkit event
loop under pytest, so _add_impulse just sets velocity and we step frames by
calling _tick() manually.
"""

from __future__ import annotations

import infinidev.ui.controls.chat_history as ch
from infinidev.ui.controls.chat_history import ChatHistoryControl


def _make(nlines: int = 500) -> ChatHistoryControl:
    msgs = [{"type": "agent", "sender": "A", "text": f"m{i}"} for i in range(nlines)]
    c = ChatHistoryControl(msgs)
    c._line_cache = None
    c._last_rebuild = 0.0
    c.create_content(width=80, height=24)
    return c


def _glide(c: ChatHistoryControl, max_frames: int = 500) -> tuple[int, int]:
    """Run animation frames until the velocity is fully spent (0.0).

    _tick() zeroes velocity on the first frame it sees it below
    _MIN_VELOCITY (and stops rescheduling), so looping until it hits 0.0
    reproduces the real terminal state. Returns (lines_moved, frames).
    """
    start = c._scroll_offset
    frames = 0
    while c._velocity != 0.0 and frames < max_frames:
        c._tick()
        frames += 1
    return c._scroll_offset - start, frames


# ── Accumulation: faster = more force ───────────────────────────────────


def test_single_notch_moves_a_little():
    c = _make()
    c._last_wheel_t = -10.0            # huge dt → slow, isolated notch
    c.move_cursor_up()
    assert c._velocity == ch._BASE_IMPULSE  # factor == 1.0 for a slow notch
    moved, _ = _glide(c)
    # A gentle notch glides only a handful of lines.
    assert 1 <= moved <= 8


def test_fast_flick_accumulates_far_more_velocity(monkeypatch):
    c = _make()
    t = [0.0]
    monkeypatch.setattr(ch.time, "monotonic", lambda: t[0])
    for _ in range(5):                 # 5 rapid notches, 20ms apart
        t[0] += 0.02
        c.move_cursor_up()
    # Velocity accumulated well beyond a single slow notch.
    assert c._velocity > ch._BASE_IMPULSE * 5
    moved, _ = _glide(c)
    # A flick travels a long way (an order of magnitude past one notch).
    assert moved > 40


def test_velocity_is_capped():
    c = _make()
    c._last_wheel_t = 0.0
    # Hammer many near-instant notches; velocity must not exceed the cap.
    import infinidev.ui.controls.chat_history as m
    m.time.monotonic  # noqa
    for _ in range(100):
        c._add_impulse(+1)
    assert c._velocity <= ch._MAX_VELOCITY


# ── Friction ────────────────────────────────────────────────────────────


def test_friction_decays_velocity_monotonically():
    c = _make()
    c._velocity = 20.0
    seen = []
    for _ in range(5):
        c._tick()
        seen.append(c._velocity)
    # Strictly decreasing, each ~ *_FRICTION of the last.
    assert all(b < a for a, b in zip(seen, seen[1:]))
    assert abs(seen[0] - 20.0 * ch._FRICTION) < 1e-6


def test_glide_eventually_stops():
    c = _make()
    c._velocity = 25.0
    _, frames = _glide(c)
    assert c._velocity == 0.0
    assert frames < 200


def test_ordinary_scroll_does_not_overshoot_a_viewport(monkeypatch):
    """The complaint that prompted the retune: after a *normal* scroll the
    glide kept travelling for most of a screen, which reads as no friction
    at all. A quarter viewport of overshoot reads as inertia; a full one
    reads as a launch."""
    c = _make()
    t = [0.0]
    monkeypatch.setattr(ch.time, "monotonic", lambda: t[0])
    for _ in range(6):                 # unhurried scrolling, 50 ms apart
        t[0] += 0.05
        c.move_cursor_up()
        for _ in range(3):             # ~3 frames elapse between notches
            c._tick()
    moved, _ = _glide(c)               # the tail once the hand lets go
    assert moved <= 15, f"overshoot of {moved} lines after an ordinary scroll"


# ── Friction is per unit time, not per tick ─────────────────────────────


def test_a_late_frame_decays_as_much_as_the_frames_it_replaced(monkeypatch):
    """``call_later`` only guarantees a lower bound, and a transcript
    rebuild mid-stream can stretch a frame well past 16 ms. Decaying per
    *tick* would make the glide outlive its budget exactly when the app is
    busy — the feel would drift with load."""
    t = [100.0]
    monkeypatch.setattr(ch.time, "monotonic", lambda: t[0])

    on_time = _make()
    on_time._velocity = 10.0
    on_time._last_tick_t = t[0]
    for _ in range(3):                 # three punctual frames
        t[0] += ch._ANIM_INTERVAL
        on_time._tick()

    t[0] = 100.0
    late = _make()
    late._velocity = 10.0
    late._last_tick_t = t[0]
    t[0] += 3 * ch._ANIM_INTERVAL      # one frame, three frames late
    late._tick()

    assert abs(late._velocity - on_time._velocity) < 1e-6
    assert abs(late._scroll_offset - on_time._scroll_offset) <= 1


def test_a_stall_stops_the_glide_instead_of_teleporting(monkeypatch):
    """A suspended terminal or a blocking call must resume to a stopped
    viewport, not fling it the whole remaining distance in one frame."""
    t = [100.0]
    monkeypatch.setattr(ch.time, "monotonic", lambda: t[0])
    c = _make()
    c._velocity = 10.0
    c._last_tick_t = t[0]
    before = c._scroll_offset
    t[0] += 10.0                       # a 10-second stall
    c._tick()
    assert c._velocity == 0.0
    assert c._scroll_offset == before


def test_a_merely_slow_frame_does_not_kill_the_glide(monkeypatch):
    """Rendering a long transcript mid-stream can cost a few hundred
    milliseconds. Treating that as a stall would make the scroll stick
    exactly while the agent is working — lateness is absorbed instead."""
    t = [100.0]
    monkeypatch.setattr(ch.time, "monotonic", lambda: t[0])
    c = _make()
    c._velocity = 10.0
    c._last_tick_t = t[0]
    t[0] += 0.3                        # one very slow frame
    c._tick()
    assert c._velocity > 0.0
    assert c._scroll_offset > 0
    # …and it absorbs at most _MAX_CATCHUP frames, so the hiccup does not
    # land as one visible jump.
    assert c._scroll_offset <= 10.0 * ch._MAX_CATCHUP


# ── Direction / bounds ──────────────────────────────────────────────────


def test_reverse_direction_cancels_residual_glide():
    c = _make()
    c._velocity = 10.0                 # gliding up
    c._last_wheel_t = -10.0
    c.move_cursor_down()               # opposite notch
    # Residual up-velocity was cleared before the down impulse applied.
    assert c._velocity < 0
    assert abs(c._velocity) <= ch._BASE_IMPULSE + 1e-6


def test_hitting_bottom_reengages_tail_follow():
    c = _make()
    c._follow_tail = False
    c._scroll_offset = 2
    c._velocity = -20.0                # flick toward the bottom
    c._tick()
    assert c._scroll_offset == 0
    assert c._follow_tail is True
    assert c._velocity == 0.0          # killed at the boundary


def test_scrolling_up_disengages_tail_follow():
    c = _make()
    assert c._follow_tail is True
    c._last_wheel_t = -10.0
    c.move_cursor_up()
    assert c._follow_tail is False


# ── Deterministic jumps stop momentum ───────────────────────────────────


def test_page_up_cancels_glide():
    c = _make()
    c._velocity = 15.0
    c._frac = 0.5
    c.page_up()
    assert c._velocity == 0.0
    assert c._frac == 0.0


def test_scroll_end_cancels_glide_and_follows_tail():
    c = _make()
    c._velocity = 15.0
    c._follow_tail = False
    c._scroll_offset = 30
    c.scroll_end()
    assert c._velocity == 0.0
    assert c._follow_tail is True
    assert c._scroll_offset == 0


# ── get_vertical_scroll: authoritative 1:1 viewport mapping ─────────────
#
# Regression cover for the desync bug: momentum mutates _scroll_offset, but
# the chat Window derives vertical_scroll from a phantom cursor via
# do_scroll, which only moves at a viewport edge. get_vertical_scroll pins
# vertical_scroll to _scroll_offset so every glide line maps 1:1 to a
# viewport line (and keeps the cursor exactly on the bottom edge so
# prompt_toolkit's do_scroll leaves the value untouched).


class _FakeWin:
    def __init__(self, height):
        self.render_info = type("RI", (), {"window_height": height})()


def _gvs(c, height, *, line_count, follow_tail, offset):
    c._line_count = line_count
    c._follow_tail = follow_tail
    c._scroll_offset = offset
    return c.get_vertical_scroll(_FakeWin(height))


def test_get_vertical_scroll_follow_tail_shows_bottom():
    c = _make()
    # follow_tail → vertical_scroll == max_scroll (content bottom visible).
    assert _gvs(c, 20, line_count=200, follow_tail=True, offset=0) == 180


def test_get_vertical_scroll_maps_offset_one_to_one():
    c = _make()
    base = _gvs(c, 20, line_count=200, follow_tail=False, offset=0)
    assert base == 180                       # offset 0 == bottom
    # Each line of scroll_offset moves vertical_scroll by exactly one line.
    assert _gvs(c, 20, line_count=200, follow_tail=False, offset=10) == 170
    assert _gvs(c, 20, line_count=200, follow_tail=False, offset=50) == 130
    # Scrolled to the very top.
    assert _gvs(c, 20, line_count=200, follow_tail=False, offset=180) == 0


def test_get_vertical_scroll_clamps_and_handles_short_content():
    c = _make()
    # Content fits the viewport → no scroll.
    assert _gvs(c, 50, line_count=20, follow_tail=False, offset=0) == 0
    # Over-scroll clamps at 0.
    assert _gvs(c, 20, line_count=200, follow_tail=False, offset=9999) == 0
    # No render_info yet (first frame) → 0, do_scroll self-corrects.
    c._line_count = 200
    c._follow_tail = False
    c._scroll_offset = 5
    assert c.get_vertical_scroll(type("W", (), {"render_info": None})()) == 0


def test_get_vertical_scroll_keeps_cursor_on_bottom_edge():
    # The whole point: cursor_y - vertical_scroll == height - 1, i.e. the
    # cursor sits exactly on the bottom edge, so prompt_toolkit's do_scroll
    # is a no-op and won't undo our value.
    c = _make()
    H, L, off = 20, 200, 30
    vs = _gvs(c, H, line_count=L, follow_tail=False, offset=off)
    cursor_y = L - 1 - off
    assert cursor_y - vs == H - 1


# ── mouse_handler consumes the wheel in BOTH directions ─────────────────


def _wheel_event(up: bool):
    from prompt_toolkit.data_structures import Point
    from prompt_toolkit.mouse_events import MouseEvent, MouseEventType, MouseButton
    return MouseEvent(
        position=Point(x=0, y=0),
        event_type=MouseEventType.SCROLL_UP if up else MouseEventType.SCROLL_DOWN,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )


def test_mouse_handler_consumes_scroll_up_and_feeds_impulse():
    c = _make()
    ret = c.mouse_handler(_wheel_event(up=True))
    assert ret is None                       # consumed (not delegated to Window)
    assert c._velocity > 0                    # upward impulse
    assert c._follow_tail is False


def test_mouse_handler_consumes_scroll_down():
    c = _make()
    c._follow_tail = False
    c._scroll_offset = 40
    ret = c.mouse_handler(_wheel_event(up=False))
    assert ret is None
    assert c._velocity < 0                    # downward impulse


def test_wheel_first_notch_moves_immediately():
    # The immediate 1-line step means a single notch from the tail scrolls
    # right away, without waiting for the animation timer.
    c = _make()
    assert c._scroll_offset == 0
    c.mouse_handler(_wheel_event(up=True))
    assert c._scroll_offset >= 1              # moved on the notch itself


# ── Real prompt_toolkit Window integration (the definitive regression) ──


def _render_window():
    """Build a real Window wrapping ChatHistoryControl and return (app, win,
    ctrl, render()). Skips cleanly if the pt render API shifts."""
    import pytest
    pt = pytest.importorskip("prompt_toolkit")
    from prompt_toolkit.layout.containers import Window, HSplit
    from prompt_toolkit.layout.layout import Layout
    from prompt_toolkit.layout.screen import Screen, WritePosition
    from prompt_toolkit.layout.mouse_handlers import MouseHandlers
    from prompt_toolkit.output import DummyOutput
    from prompt_toolkit.input import DummyInput
    from prompt_toolkit.application import Application
    from prompt_toolkit.application.current import set_app

    msgs = [{"type": "agent", "sender": "A", "text": f"line {i}"} for i in range(200)]
    ctrl = ChatHistoryControl(msgs)
    win = Window(content=ctrl, wrap_lines=False,
                 get_vertical_scroll=ctrl.get_vertical_scroll)
    app = Application(layout=Layout(HSplit([win])),
                      output=DummyOutput(), input=DummyInput())

    H, W = 20, 80

    def render():
        with set_app(app):
            win.write_to_screen(Screen(), MouseHandlers(),
                                WritePosition(0, 0, W, H), "", False, None)
        return win.vertical_scroll

    return app, win, ctrl, render, set_app


def test_real_window_wheel_scrolls_viewport_no_dead_zone():
    """The bug: with the scrollbar removed, momentum mutated _scroll_offset
    but Window.vertical_scroll (driven by the phantom cursor) stayed pinned,
    so glides floated the cursor across a STATIC viewport (dead zones).
    With get_vertical_scroll authoritative, every wheel notch must visibly
    move the viewport."""
    import pytest
    pytest.importorskip("prompt_toolkit")
    app, win, ctrl, render, set_app = _render_window()

    with set_app(app):
        render()
        start = win.vertical_scroll
        # follow_tail → starts at the bottom (max scroll).
        assert start > 0

        seen = []
        for _ in range(6):
            ctrl.mouse_handler(_wheel_event(up=True))
            for _ in range(40):              # let the glide play out
                if ctrl._velocity == 0.0:
                    break
                ctrl._tick()
            render()
            seen.append(win.vertical_scroll)

        # Every notch produced visible upward movement (strictly decreasing
        # vertical_scroll) — no dead zones, no floating cursor.
        assert seen[0] < start
        assert all(b < a for a, b in zip(seen, seen[1:])), seen

        # Scroll back down re-approaches the bottom and re-engages tail.
        for _ in range(30):
            ctrl.mouse_handler(_wheel_event(up=False))
            for _ in range(40):
                if ctrl._velocity == 0.0:
                    break
                ctrl._tick()
        render()
        assert win.vertical_scroll >= seen[-1]
