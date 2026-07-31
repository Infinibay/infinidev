"""The transcript ToolRunner builds has to be one a provider will accept.

These tests assert a shape, not an implementation: after ``run_regular``
returns, every ``role: "tool"`` message must still be reachable from the
assistant turn that announced it, walking back through nothing but other
tool results. A ``user`` turn wedged in between (an image payload, the
budget nudge) splits the block, and OpenAI, Anthropic and Minimax all
reject the next request outright — which turns "the model asked for two
tools" into "the run died mid-step".
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from infinidev.engine.loop.classified_calls import ClassifiedCalls
from infinidev.engine.loop.behavior_tracker import BehaviorTracker
from infinidev.engine.loop.loop_guard import LoopGuard
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.tool_runner import ToolRunner
from infinidev.engine.file_change_tracker import FileChangeTracker


# ── fakes ────────────────────────────────────────────────────────────


def _call(call_id: str, name: str, arguments: str = "{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _engine(nudge_at: int | None = 1):
    """An engine stub with the four attributes ToolRunner reaches for."""
    return SimpleNamespace(
        _nudge_threshold_override=nudge_at,
        _cancel_event=threading.Event(),
        _cr_hooks=SimpleNamespace(on_tool_call=lambda *a, **k: None),
        _hooks=None,
    )


def _ctx(manual_tc: bool = False):
    """A context stub. The tool dispatch is empty on purpose.

    An unresolved tool still returns a JSON error string, which is all
    these tests need: the transcript's *shape* does not depend on whether
    the tool succeeded, and an empty dispatch keeps the test from dragging
    in the real filesystem tools.
    """
    return SimpleNamespace(
        manual_tc=manual_tc,
        is_small=False,
        state=LoopState(),
        tool_dispatch={},
        file_tracker=FileChangeTracker(),
        agent_name="dev",
        verbose=False,
        project_id=1,
        agent_id="agent-1",
        nudge_message_template=None,
        max_per_action=4,
    )


def _run(classified: ClassifiedCalls, *, manual_tc: bool = False, nudge_at: int | None = 1):
    ctx = _ctx(manual_tc=manual_tc)
    runner = ToolRunner(_engine(nudge_at))
    messages: list[dict] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    llm_result = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]),
        raw_content="",
    )
    runner.run_regular(
        ctx, classified, messages, llm_result,
        action_tool_calls=0, iteration=0,
        guard=LoopGuard(is_small=False),
        tracker=BehaviorTracker(set()),
    )
    return messages


def assert_tool_block_contiguous(messages: list[dict]) -> None:
    """Every tool result traces back to an assistant that announced it."""
    for i, msg in enumerate(messages):
        if msg.get("role") != "tool":
            continue
        j = i - 1
        while j >= 0 and messages[j].get("role") == "tool":
            j -= 1
        assert j >= 0, f"tool result at {i} opens the conversation"
        prev = messages[j]
        assert prev.get("role") == "assistant" and prev.get("tool_calls"), (
            f"tool result at index {i} ({msg.get('tool_call_id')}) is preceded "
            f"by a {prev.get('role')!r} turn — the assistant→tool block is split. "
            f"Roles: {[m.get('role') for m in messages]}"
        )


# ── the four ways the block used to split ────────────────────────────


def test_budget_nudge_does_not_split_a_single_batch_with_a_pseudo_tool():
    """The common case: one read plus a think, with the nudge firing.

    The pseudo-tool ack is appended after the batches have run, so a nudge
    written at the end of the batch landed in front of it.
    """
    messages = _run(ClassifiedCalls(
        regular=[_call("c1", "read_file", '{"file_path": "a.py"}')],
        thinks=[_call("c2", "think")],
    ))
    assert_tool_block_contiguous(messages)
    assert messages[-1]["role"] == "user", "the nudge should close the turn"


def test_budget_nudge_does_not_split_a_multi_batch_turn():
    """A write tool gets a batch of its own, so batch 1 ended mid-block."""
    messages = _run(ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "create_file", '{"file_path": "b.py", "content": "x"}'),
    ]))
    assert_tool_block_contiguous(messages)
    assert [m["role"] for m in messages].count("tool") == 2


def test_step_complete_ack_stays_inside_the_block():
    messages = _run(ClassifiedCalls(
        regular=[_call("c1", "read_file", '{"file_path": "a.py"}')],
        step_complete=_call("c2", "step_complete", '{"status": "continue"}'),
    ))
    assert_tool_block_contiguous(messages)


def test_notes_and_writes_together_stay_inside_the_block():
    messages = _run(ClassifiedCalls(
        regular=[
            _call("c1", "read_file", '{"file_path": "a.py"}'),
            _call("c2", "create_file", '{"file_path": "b.py", "content": "x"}'),
        ],
        notes=[_call("c3", "add_note")],
        thinks=[_call("c4", "think")],
    ))
    assert_tool_block_contiguous(messages)


# ── manual mode has no tool channel at all ───────────────────────────


def test_manual_mode_emits_no_tool_role_messages():
    """In manual mode results are prose, so there is no block to split."""
    messages = _run(ClassifiedCalls(
        regular=[_call("c1", "read_file", '{"file_path": "a.py"}')],
        thinks=[_call("c2", "think")],
    ), manual_tc=True)
    assert not any(m.get("role") == "tool" for m in messages)
    assert any("STEP BUDGET" in str(m.get("content", "")) for m in messages), (
        "the nudge has to ride along with the prose results in manual mode"
    )


# ── the nudge itself ─────────────────────────────────────────────────


def test_no_nudge_no_trailing_user_turn():
    messages = _run(ClassifiedCalls(
        regular=[_call("c1", "read_file", '{"file_path": "a.py"}')],
    ), nudge_at=99)
    assert_tool_block_contiguous(messages)
    assert messages[-1]["role"] == "tool"


def test_nudge_fires_once_across_batches():
    """Equality on the counter, so a multi-batch turn still warns once."""
    messages = _run(ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "create_file", '{"file_path": "b.py", "content": "x"}'),
        _call("c3", "read_file", '{"file_path": "c.py"}'),
    ]))
    nudges = [m for m in messages if m.get("role") == "user" and m is not messages[1]]
    assert len(nudges) == 1
