"""Regression tests for the THINKING accumulator lifecycle.

`event_handler.process_event` keeps two thinking buffers on the app:

  * ``_thinking_full`` — the untruncated accumulator for the current LLM
    call, flushed verbatim into the chat as a permanent "Thinking"
    message on ``loop_stream_status`` phase=="done".
  * ``_thinking_text`` — the truncated sidebar VIEW (``"..." + tail``).

The bug these tests pin: because a "done" event fires at the end of
EVERY LLM call (several per step) and the buffers were only cleared on
step transitions, each subsequent "done" re-flushed the same buffer
(duplicate) using the truncated view (cut off). The fix flushes the
FULL text once per call and resets both buffers on every "done".
"""

from __future__ import annotations

from infinidev.ui.event_handler import process_event


class _FakeApp:
    """Minimal stand-in exposing only what the thinking branches touch."""

    def __init__(self) -> None:
        self._thinking_text = ""
        self._thinking_full = ""
        self._streaming_tool_name = None
        self._streaming_token_count = 0
        self._last_thinking_invalidate = 0.0
        self.messages: list[tuple[str, str, str]] = []
        self.logs: list[str] = []

    def add_message(self, sender: str, text: str, msg_type: str = "agent") -> None:
        self.messages.append((sender, text, msg_type))

    def add_log(self, text: str) -> None:
        self.logs.append(text)

    def invalidate(self) -> None:
        pass


def _think_messages(app: _FakeApp) -> list[tuple[str, str, str]]:
    return [m for m in app.messages if m[2] == "think"]


def test_thinking_flush_is_full_text_and_not_duplicated():
    app = _FakeApp()
    # Reasoning longer than the 500-char sidebar truncation threshold.
    long_reasoning = "".join(f"step{i} " for i in range(200))
    assert len(long_reasoning) > 500

    # Stream it in two chunks (simulating native thinking deltas).
    process_event(app, "loop_thinking_chunk", {"text": long_reasoning[:300]})
    process_event(app, "loop_thinking_chunk", {"text": long_reasoning[300:]})

    # Sidebar VIEW is truncated; the full accumulator is not.
    assert app._thinking_text.startswith("...")
    assert len(app._thinking_text) < len(long_reasoning)
    assert app._thinking_full == long_reasoning

    # First stream-done: flush the FULL, untruncated reasoning exactly once
    # (stripped of surrounding whitespace, but otherwise verbatim — no
    # "..." truncation marker, no cut-off tail).
    process_event(app, "loop_stream_status", {"phase": "done"})
    flushed = _think_messages(app)
    assert len(flushed) == 1
    assert flushed[0] == ("Thinking", long_reasoning.strip(), "think")
    # Both buffers reset so the next LLM call starts clean.
    assert app._thinking_full == ""
    assert app._thinking_text == ""

    # Second stream-done with NO chunks in between must NOT re-emit
    # (this was the duplicate bug).
    process_event(app, "loop_stream_status", {"phase": "done"})
    assert len(_think_messages(app)) == 1


def test_two_calls_each_flush_their_own_full_text():
    app = _FakeApp()

    process_event(app, "loop_thinking_chunk", {"text": "first call reasoning"})
    process_event(app, "loop_stream_status", {"phase": "done"})

    process_event(app, "loop_thinking_chunk", {"text": "second call reasoning"})
    process_event(app, "loop_stream_status", {"phase": "done"})

    flushed = _think_messages(app)
    assert [m[1] for m in flushed] == [
        "first call reasoning",
        "second call reasoning",
    ]


def test_runaway_reasoning_stream_is_bounded() -> None:
    from infinidev.ui.event_handler import _MAX_THINKING_TRANSCRIPT_CHARS

    app = _FakeApp()
    process_event(
        app,
        "loop_thinking_chunk",
        {"text": "x" * (_MAX_THINKING_TRANSCRIPT_CHARS + 100)},
    )

    assert len(app._thinking_full) == _MAX_THINKING_TRANSCRIPT_CHARS
    assert app._thinking_full.startswith("[Earlier streamed reasoning truncated")


def test_loop_think_pseudo_tool_not_reflushed_on_next_done():
    app = _FakeApp()
    # An explicit `think` writes straight to chat and updates the sidebar
    # view, but must NOT seed _thinking_full...
    process_event(app, "loop_think", {"reasoning": "explicit reasoning"})
    assert len(_think_messages(app)) == 1
    assert app._thinking_full == ""

    # ...so the next stream-done does not re-emit it as a duplicate.
    process_event(app, "loop_stream_status", {"phase": "done"})
    assert len(_think_messages(app)) == 1
