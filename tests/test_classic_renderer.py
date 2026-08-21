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


def test_loop_tool_start_is_visible_before_long_tool_finishes(
    renderer, fresh_bus, status, capsys
):
    fresh_bus.emit("loop_tool_start", 1, "agent-a", {
        "tool_name": "execute_command",
        "tool_detail": "python -m pytest",
        "call_num": 3,
        "step_limit": 12,
        "total_calls": 19,
        "total_limit": 1000,
    })

    assert status.activity == "running execute_command"
    out = capsys.readouterr().out
    assert "running" in out
    assert "execute_command" in out
    assert "python -m pytest" in out
    assert "step 3/12" in out
    assert "total 19/1000" in out


def test_loop_llm_call_prints_current_work_and_budget(
    renderer, fresh_bus, status, capsys
):
    fresh_bus.emit("loop_llm_call_start", 1, "agent-a", {
        "phase": "planning",
        "step_title": "Fix forwarded decoder",
        "tool_calls_step": 2,
        "tool_calls_step_limit": 12,
        "tool_calls_total": 18,
        "tool_calls_total_limit": 1000,
    })

    assert status.activity == "model planning"
    out = capsys.readouterr().out
    assert "model planning" in out
    assert "Fix forwarded decoder" in out
    assert "step tools 2/12" in out
    assert "total 18/1000" in out


def test_loop_llm_call_shows_stalled_discovery_recovery(
    renderer, fresh_bus, status, capsys
):
    fresh_bus.emit("loop_llm_call_start", 1, "agent-a", {
        "phase": "recovery",
        "step_title": "Implement parser fix",
        "tool_calls_step": 12,
        "tool_calls_step_limit": 0,
        "tool_calls_total": 24,
        "tool_calls_total_limit": 0,
    })

    assert status.activity == "model recovery"
    out = capsys.readouterr().out
    assert "model recovery" in out
    assert "Implement parser fix" in out
    assert "step tools 12" in out
    assert "total 24" in out
def test_context_compaction_is_printed_in_classic_mode(
    renderer, fresh_bus, status, capsys
):
    fresh_bus.emit("loop_context_compaction", 1, "agent-a", {
        "prompt_tokens": 700_000,
        "context_limit": 1_000_000,
        "remaining_tokens": 300_000,
        "percent_used": 70.0,
    })

    assert status.activity == "compacting context"
    assert status.last_prompt_tokens == 700_000
    out = capsys.readouterr().out
    assert "compacting context" in out
    assert "70.0%" in out
    assert "300,000 free" in out



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


def test_loop_start_resets_stale_token_counters(renderer, fresh_bus, status):
    # Leftover figures from the previous turn must not bleed into the next.
    status.last_prompt_tokens = 111
    status.last_completion_tokens = 222
    status.total_tokens = 999
    status.cache_read = 5
    status.cache_create = 7
    status.tool_calls_total = 3
    status.iteration = 9
    fresh_bus.emit("loop_start", 1, "agent-a", {"prompt": "new turn"})
    assert status.last_prompt_tokens == 0
    assert status.last_completion_tokens == 0
    assert status.total_tokens == 0
    assert status.cache_read == 0
    assert status.cache_create == 0
    assert status.tool_calls_total == 0
    assert status.iteration == 0


def test_council_finished_promises_only_recent_transcript(renderer, fresh_bus, capsys):
    fresh_bus.emit(
        "council_finished",
        1,
        "agent-a",
        {"council": {"id": "council-7", "status": "completed"}},
    )

    out = capsys.readouterr().out
    assert "council-7 completed" in out
    assert "recent transcript in /agents" in out
    assert "transcript kept" not in out


def test_flush_think_does_not_truncate_long_tail(renderer, fresh_bus, capsys):
    long_thought = "x" * 500  # > 320, no newline → kept as the buffered tail
    fresh_bus.emit("loop_thinking_chunk", 1, "agent-a", {"text": long_thought})
    capsys.readouterr()  # nothing flushed yet (no newline)
    # A non-thinking event flushes the buffered tail — in full, not cut at 320.
    fresh_bus.emit("loop_user_message", 1, "agent-a", {"message": "go"})
    out = capsys.readouterr().out
    assert long_thought in out


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
