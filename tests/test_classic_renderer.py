"""Tests for the classic-mode terminal renderer.

Covers:
- SessionStatus token absorption from event payloads
- Stream-chunk buffering of `loop_thinking_chunk`
- Per-event handlers writing to stdout (via capsys)
- PermissionQueue round-trip
- Status table rendering
- Subscribe/unsubscribe lifecycle on the EventBus
"""

from __future__ import annotations

import threading

import pytest

from infinidev.cli import classic_renderer as cr
from infinidev.flows.event_listeners import EventBus


@pytest.fixture()
def fresh_bus(monkeypatch):
    """Replace the module-level event_bus with a fresh instance per test."""
    bus = EventBus()
    monkeypatch.setattr(cr, "event_bus", bus)
    return bus


@pytest.fixture()
def status():
    return cr.SessionStatus(provider="ollama", model="qwen3:32b")


@pytest.fixture()
def renderer(fresh_bus, status):
    r = cr.ClassicRenderer(status)
    r.subscribe()
    yield r
    r.unsubscribe()


# ── Subscription lifecycle ──────────────────────────────────────────────


def test_subscribe_unsubscribe_is_idempotent(fresh_bus, status):
    r = cr.ClassicRenderer(status)
    r.subscribe()
    r.subscribe()  # second call is a no-op
    assert fresh_bus.has_subscribers
    r.unsubscribe()
    r.unsubscribe()
    assert not fresh_bus.has_subscribers


# ── Token absorption ─────────────────────────────────────────────────────


def test_loop_step_update_updates_tokens_and_iteration(renderer, fresh_bus, status, capsys):
    fresh_bus.emit("loop_step_update", 1, "agent-a", {
        "iteration": 3,
        "step_title": "Read project layout",
        "status": "active",
        "prompt_tokens": 1200,
        "completion_tokens": 300,
        "tokens_total": 4500,
    })
    assert status.iteration == 3
    assert status.last_prompt_tokens == 1200
    assert status.last_completion_tokens == 300
    assert status.total_tokens == 4500
    assert status.step_title == "Read project layout"
    out = capsys.readouterr().out
    assert "step 3" in out
    assert "Read project layout" in out


def test_loop_tool_call_updates_total_calls_and_prints(renderer, fresh_bus, status, capsys):
    fresh_bus.emit("loop_tool_call", 1, "agent-a", {
        "tool_name": "read_file",
        "tool_detail": "src/main.py",
        "tool_error": "",
        "call_num": 1,
        "total_calls": 7,
        "tokens_total": 9000,
    })
    assert status.tool_calls_total == 7
    assert status.total_tokens == 9000
    out = capsys.readouterr().out
    assert "read_file" in out
    assert "src/main.py" in out


def test_loop_tool_call_with_error_renders_red_marker(renderer, fresh_bus, capsys):
    fresh_bus.emit("loop_tool_call", 1, "agent-a", {
        "tool_name": "execute_command",
        "tool_detail": "bad cmd",
        "tool_error": "command not found",
        "call_num": 2,
        "total_calls": 2,
    })
    out = capsys.readouterr().out
    assert "command not found" in out
    assert "✗" in out


# ── Streaming reasoning chunks ─────────────────────────────────────────


def test_thinking_chunks_buffer_until_newline(renderer, fresh_bus, capsys):
    fresh_bus.emit("loop_thinking_chunk", 1, "agent-a", {"text": "lets read"})
    fresh_bus.emit("loop_thinking_chunk", 1, "agent-a", {"text": " the file"})
    out_partial = capsys.readouterr().out
    assert out_partial == ""  # nothing flushed yet
    fresh_bus.emit("loop_thinking_chunk", 1, "agent-a", {"text": " first\n"})
    out = capsys.readouterr().out
    assert "lets read the file first" in out
    assert "💭" in out


def test_non_thinking_event_flushes_pending_thoughts(renderer, fresh_bus, capsys):
    fresh_bus.emit("loop_thinking_chunk", 1, "agent-a", {"text": "partial thought"})
    capsys.readouterr()  # nothing yet
    fresh_bus.emit("loop_tool_call", 1, "agent-a", {
        "tool_name": "x", "tool_detail": "", "call_num": 1, "total_calls": 1,
    })
    out = capsys.readouterr().out
    assert "partial thought" in out
    # And the tool call line follows.
    assert "x" in out
    assert "▸" in out


# ── Critic verdicts ─────────────────────────────────────────────────────


def test_assistant_message_records_action_and_prints(renderer, fresh_bus, status, capsys):
    fresh_bus.emit("loop_assistant_message", 1, "agent-a", {
        "action": "warn",
        "message": "you forgot to read the file first",
        "model": "o4-mini",
        "blocked": False,
    })
    assert status.last_verdict_action == "warn"
    out = capsys.readouterr().out
    assert "critic" in out
    assert "o4-mini" in out
    assert "warn" in out


# ── Run lifecycle ──────────────────────────────────────────────────────


def test_loop_start_sets_run_started(renderer, fresh_bus, status):
    assert status.run_started_at is None
    fresh_bus.emit("loop_start", 1, "agent-a", {"prompt": "do the thing"})
    assert status.run_started_at is not None


def test_loop_end_clears_run_started(renderer, fresh_bus, status):
    fresh_bus.emit("loop_start", 1, "agent-a", {"prompt": "x"})
    fresh_bus.emit("loop_end", 1, "agent-a", {"summary": "all good"})
    assert status.run_started_at is None


# ── Renderer never raises on malformed events ────────────────────────────


def test_renderer_swallows_handler_exceptions(renderer, fresh_bus):
    # Missing keys, wrong types — should NOT raise.
    fresh_bus.emit("loop_tool_call", 1, "agent-a", {})
    fresh_bus.emit("loop_step_update", 1, "agent-a", {"iteration": "bogus"})
    fresh_bus.emit("loop_assistant_message", 1, "agent-a", {"action": None})
    # No assertion — getting here without an exception is the test.


# ── PermissionQueue round-trip ────────────────────────────────────────────


def test_permission_queue_blocks_until_resolved():
    pq = cr.PermissionQueue()
    handler = cr.make_permission_handler(pq)

    result_holder = []
    started = threading.Event()

    def worker():
        started.set()
        result_holder.append(handler("read_file", "Read /etc/passwd", "for science"))

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    started.wait(timeout=1.0)

    # Worker is blocked on req.done.wait(); main resolves it.
    req = None
    for _ in range(50):
        req = pq.pending()
        if req is not None:
            break
        threading.Event().wait(0.01)
    assert req is not None
    assert req.tool_name == "read_file"
    req.result = True
    req.done.set()

    t.join(timeout=1.0)
    assert result_holder == [True]


def test_permission_queue_pending_returns_none_when_empty():
    pq = cr.PermissionQueue()
    assert pq.pending() is None


# ── Status table renderer ─────────────────────────────────────────────────


def test_render_status_table_includes_key_fields(status):
    status.iteration = 2
    status.tool_calls_total = 5
    status.total_tokens = 12345
    status.critic_enabled = True
    status.critic_model = "claude-haiku-4-5"
    out = cr.render_status_table(status)
    assert "iteration" in out
    assert "claude-haiku-4-5" in out
    assert "12345" in out


# ── Bottom toolbar callable ───────────────────────────────────────────────


def test_status_renderer_returns_formatted_text(status):
    status.run_started_at = None
    status.iteration = 1
    status.total_tokens = 1500
    render = cr.make_status_renderer(status)
    ft = render()
    # FormattedText is a list of (style, text) tuples.
    flat = "".join(t for _, t in ft)
    assert status.model in flat
    assert "it 1" in flat
    assert "1.5k" in flat


def test_status_renderer_shows_elapsed_when_running(status):
    import time
    status.run_started_at = time.monotonic() - 7
    render = cr.make_status_renderer(status)
    flat = "".join(t for _, t in render())
    assert "s" in flat  # elapsed segment
    # Elapsed should be ~7s.
    assert any(ch.isdigit() for ch in flat)
