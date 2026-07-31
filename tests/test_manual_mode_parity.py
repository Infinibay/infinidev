"""Manual tool-calling mode has to produce a valid conversation too.

Manual mode is the documented fallback for models without native function
calling — the local open-weight models this tool exists for. Three
subsystems assumed the function-calling transcript shape and quietly did
the wrong thing when it was absent: the step_complete gates answered a
tool call nobody made, working memory archived nothing, and synthetic
call ids collided.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from infinidev.engine.loop.llm_caller import LLMCaller
from infinidev.engine.loop.user_message_injector import UserMessageInjector
from infinidev.engine.working_memory import WorkingMemory


# ── the gates, in a conversation with no tool channel ─────────────────


def _manual_transcript() -> list[dict]:
    """What manual mode actually builds: prose, and acks as user turns."""
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": 'I will call step_complete\n{"tool": "step_complete"}'},
        {"role": "user", "content": '[Tool: add_note] Result:\n{"status": "noted"}'},
    ]


def test_gate_feedback_in_manual_mode_never_invents_a_tool_result():
    messages = _manual_transcript()
    UserMessageInjector._overwrite_step_complete_tool_result(
        messages, "manual_1", "step_complete REJECTED — verification failed",
    )
    assert not any(m.get("role") == "tool" for m in messages), (
        "a role=tool message answering a call no assistant announced makes "
        "the next request invalid on both OpenAI and Anthropic"
    )
    assert messages[-1]["role"] == "user"
    assert "REJECTED" in messages[-1]["content"]


def test_gate_feedback_in_fc_mode_still_uses_the_tool_channel():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "sc1", "type": "function",
             "function": {"name": "step_complete", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "sc1", "content": '{"status": "acknowledged"}'},
    ]
    UserMessageInjector._overwrite_step_complete_tool_result(
        messages, "sc1", "step_complete REJECTED",
    )
    assert len(messages) == 4, "the existing stub is rewritten, not duplicated"
    assert messages[3]["content"] == "step_complete REJECTED"


def test_gate_feedback_appends_a_tool_result_when_the_stub_is_missing():
    """FC mode with no prior ack — the pre-existing fallback still applies."""
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "sc1", "type": "function",
             "function": {"name": "step_complete", "arguments": "{}"}},
        ]},
    ]
    UserMessageInjector._overwrite_step_complete_tool_result(
        messages, "sc1", "step_complete REJECTED",
    )
    assert messages[-1]["role"] == "tool"
    assert messages[-1]["tool_call_id"] == "sc1"


# ── working memory, independent of transcript shape ──────────────────


def test_archive_calls_stores_what_the_tool_returned(tmp_path):
    memory = WorkingMemory("session-manual", db_path=str(tmp_path / "wm.db"))
    traceback = "AssertionError: expected 3, got 4\n" + ("stack frame\n" * 40)

    stored = memory.archive_calls(1, [
        ("execute_command", json.dumps({"command": "pytest"}), traceback),
    ])

    # The returned label is not decoration: it is the title the row was filed
    # under, which is what the plan block renders so the model can recall it.
    assert stored == ["execute_command(command=pytest)"]
    hits = memory.search("pytest failure assertion", limit=5)
    assert any("expected 3, got 4" in (h.content or "") for h in hits), (
        "in manual mode this is the only path — archive_step sees no "
        "role=tool messages and stores nothing"
    )


def test_archive_calls_skips_results_too_short_to_be_worth_recalling(tmp_path):
    memory = WorkingMemory("session-short", db_path=str(tmp_path / "wm.db"))
    assert memory.archive_calls(1, [("git_status", "{}", "clean")]) == []


def test_archive_calls_survives_unparseable_arguments(tmp_path):
    memory = WorkingMemory("session-bad-args", db_path=str(tmp_path / "wm.db"))
    stored = memory.archive_calls(1, [
        ("read_file", "{not json", "x" * 200),
    ])
    assert len(stored) == 1, "a malformed argument string must not lose the result"


# ── synthetic ids ────────────────────────────────────────────────────


def test_synthetic_ids_do_not_repeat_within_a_step():
    """A pseudo-tool consumed an id without advancing the old seed."""
    caller = LLMCaller()
    caller.reset()

    first = [caller._next_synthetic_id("fc_fallback") for _ in range(2)]  # think + read_file
    # Only read_file executed, so action_tool_calls advanced by 1 — which
    # used to be the seed for the next pass.
    second = [caller._next_synthetic_id("fc_fallback") for _ in range(1)]

    assert not set(first) & set(second), f"id collision: {first} vs {second}"


def test_synthetic_ids_restart_each_step():
    """Ids only need to be unique inside one messages list."""
    caller = LLMCaller()
    caller.reset()
    first_step = [caller._next_synthetic_id("manual") for _ in range(3)]
    caller.reset()
    second_step = [caller._next_synthetic_id("manual") for _ in range(3)]
    assert first_step == second_step
