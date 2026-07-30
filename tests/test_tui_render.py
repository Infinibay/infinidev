"""Deterministic full-screen renders of the TUI.

prompt_toolkit can draw a layout into an in-memory ``Screen`` with no
terminal attached, which makes the whole UI testable as text. These are
the guard rails for the transcript-first layout: they fail if the
composer stops framing the input, if a side panel starts stealing width
by default, or if the status line drops off the bottom.
"""

from __future__ import annotations

import asyncio

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.data_structures import Size
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput

WIDTH = 96
HEIGHT = 26


class _SizedOutput(DummyOutput):
    def get_size(self) -> Size:
        return Size(rows=HEIGHT, columns=WIDTH)


def _draw(app_state, width: int = WIDTH, height: int = HEIGHT) -> list[str]:
    """Render the app's root container and return the screen as text rows.

    Rows keep the layout's one-column side margin; use ``_body`` when the
    assertion is about content rather than about the margin itself.
    """
    screen = Screen(default_char=None, initial_width=width, initial_height=height)
    app_state._float_container.write_to_screen(
        screen, MouseHandlers(), WritePosition(0, 0, width, height), "", False, None
    )
    screen.draw_all_floats()
    rows = []
    for y in range(height):
        row = screen.data_buffer[y]
        rows.append("".join(row[x].char for x in range(width)).rstrip())
    return rows


def _body(rows: list[str]) -> list[str]:
    """Rows with the layout's left margin removed."""
    return [row[1:] if row.startswith(" ") else row for row in rows]


def _render(messages, *, sidebar=False, explorer=False, width=WIDTH, height=HEIGHT):
    """Build a real app, seed the transcript, and return the drawn screen."""

    async def _run():
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.config.settings import settings

            settings.UI_SIDEBAR_VISIBLE = sidebar
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            app.sidebar_visible = sidebar
            app.explorer_visible = explorer
            app.chat_messages.clear()
            app.chat_messages.extend(messages)
            app._chat_history_control.invalidate_cache()
            return _draw(app, width, height)

    return asyncio.run(_run())


@pytest.fixture
def transcript():
    return [
        {"type": "user", "sender": "You", "text": "add retry logic to the http client"},
        {
            "type": "agent",
            "sender": "Infinidev",
            "text": "Adding exponential backoff to HttpClient.request.",
        },
        {
            "type": "tool_call",
            "sender": "Tool",
            "tool_name": "read_file",
            "args": {"file_path": "src/http/client.py"},
            "result": "120 lines",
        },
        {
            "type": "tool_call",
            "sender": "Tool",
            "tool_name": "replace_lines",
            "args": {"file_path": "src/http/client.py"},
            "result": "applied",
        },
    ]


# ── layout ────────────────────────────────────────────────────────────────


def test_transcript_uses_the_full_width_by_default(transcript):
    rows = _render(transcript)
    screen = "\n".join(rows)
    # No sidebar panel titles anywhere on screen.
    for panel in ("CONTEXT", "STEPS", "ACTIVITY", "LOGS"):
        assert panel not in screen
    # The composer frame spans everything but the one-column side margins.
    top = next(row for row in rows if row.lstrip().startswith("╭"))
    assert len(top) == WIDTH - 1  # trailing margin column is stripped
    assert top.endswith("╮")


def test_composer_frames_the_input_with_a_prompt_mark(transcript):
    rows = _render(transcript)
    framed = [row for row in _body(rows) if row.startswith("│")]
    assert framed, "the composer must draw side borders"
    assert any("›" in row for row in framed), "prompt mark missing"
    assert any("Ask anything" in row for row in framed), "placeholder missing"
    assert any(
        row.startswith("╰") and row.endswith("╯") for row in _body(rows)
    )


def test_status_line_is_the_last_row_and_shows_hints(transcript):
    rows = _render(transcript)
    status = next(row for row in reversed(rows) if row.strip())
    assert "help" in status
    assert "exit" in status


def test_sidebar_returns_on_demand(transcript):
    rows = _render(transcript, sidebar=True)
    screen = "\n".join(rows)
    assert "CONTEXT" in screen and "SESSION" in screen
    # ...and the composer shrinks to match instead of overflowing.
    top = next(row for row in rows if row.lstrip().startswith("╭"))
    assert len(top) < WIDTH - 1


def test_explorer_returns_on_demand(transcript):
    screen = "\n".join(_render(transcript, explorer=True))
    assert "FILES" in screen


# ── message rendering ─────────────────────────────────────────────────────


def test_user_message_is_marked_once_at_its_start(transcript):
    rows = _render(
        [
            {
                "type": "user",
                "sender": "You",
                "text": "line one of the request\nline two of the request",
            }
        ]
    )
    marked = [row for row in _body(rows) if row.startswith(">")]
    assert len(marked) == 1, "the prompt mark opens the turn, it isn't a border"
    assert any(row.startswith("  line two") for row in _body(rows))
    assert "You:" not in "\n".join(rows), "the user's own words need no name tag"


def test_assistant_reply_carries_no_decoration(transcript):
    rows = _render(
        [
            {
                "type": "agent",
                "sender": "Infinidev",
                "text": "first paragraph\n\nsecond paragraph",
            }
        ]
    )
    assert "Infinidev:" not in "\n".join(rows)
    assert any(row.startswith("  first paragraph") for row in _body(rows))
    assert any(row.startswith("  second paragraph") for row in _body(rows))


def test_conversation_is_bottom_anchored(transcript):
    """A short exchange sits just above the composer, not at the top."""
    rows = _render(transcript)
    composer_top = next(
        i for i, row in enumerate(rows) if row.lstrip().startswith("╭")
    )
    last_text = max(
        i for i, row in enumerate(rows[:composer_top]) if row.strip()
    )
    assert composer_top - last_text <= 2, "content must hug the composer"
    assert not rows[0].strip(), "padding goes above the conversation, not below"


def test_named_speakers_still_get_a_header():
    rows = _render([{"type": "agent", "sender": "Reviewer", "text": "needs a test"}])
    assert any("Reviewer:" in row for row in rows)


def test_consecutive_tools_collapse_into_one_group(transcript):
    rows = _render(transcript)
    summary = [row for row in rows if "tools" in row]
    assert summary, "consecutive tool calls must fold into a single summary line"
    assert "Ran 2 tools" in summary[0]
    # Collapsed by default: the individual calls are not listed.
    assert "read_file" not in "\n".join(rows)


def test_copy_affordance_survives_the_headerless_layout(transcript):
    rows = _render([{"type": "agent", "sender": "Infinidev", "text": "hello"}])
    assert any("⧉" in row for row in rows), "copy stays discoverable on replies"


def test_working_indicator_shows_a_spinner_and_a_label():
    """A long phase must look like progress, not like a hang."""

    async def _run():
        with create_pipe_input() as pipe, create_app_session(
            input=pipe, output=_SizedOutput()
        ):
            from infinidev.ui.app import InfinidevApp

            app = InfinidevApp()
            app.sidebar_visible = False
            app.chat_messages.clear()
            app._chat_history_control.show_thinking = True
            app._chat_history_control.work_label = "Planning..."
            app._chat_history_control.invalidate_cache()
            return _draw(app)

    rows = asyncio.run(_run())
    line = next(row for row in rows if "Planning..." in row)
    assert any(frame in line for frame in "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏")


# ── narrow terminals ──────────────────────────────────────────────────────


def test_narrow_terminal_does_not_overflow(transcript):
    narrow = 48
    rows = _render(transcript, width=narrow, height=20)
    assert all(len(row) <= narrow for row in rows)
    top = next(row for row in rows if row.lstrip().startswith("╭"))
    assert len(top) == narrow - 1
