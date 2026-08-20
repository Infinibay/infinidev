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

import json
import threading
from types import SimpleNamespace

import pytest

from infinidev.engine.loop.classified_calls import ClassifiedCalls
from infinidev.engine.loop.behavior_tracker import BehaviorTracker
from infinidev.engine.loop.context_manager import ContextManager
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
    """An engine stub with the cancellation scope ToolRunner collaborates with."""
    tool_cancel_event = threading.Event()
    tool_running_event = threading.Event()
    engine = SimpleNamespace(
        _nudge_threshold_override=nudge_at,
        _cancel_event=threading.Event(),
        _tool_cancel_event=tool_cancel_event,
        _tool_running_event=tool_running_event,
        _cr_hooks=SimpleNamespace(on_tool_call=lambda *a, **k: None),
        _hooks=None,
    )

    def begin_tool_batch() -> None:
        tool_cancel_event.clear()
        tool_running_event.set()

    def finish_tool_batch() -> None:
        tool_running_event.clear()
        tool_cancel_event.clear()

    engine._begin_tool_batch = begin_tool_batch
    engine._finish_tool_batch = finish_tool_batch
    return engine


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
        max_total_calls=40,
        allow_plan_mutation=True,
        skip_plan=False,
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


def test_assistant_history_preserves_provider_reasoning_protocol_fields() -> None:
    messages: list[dict] = []
    message = SimpleNamespace(
        content="",
        reasoning_content="visible summary",
        thinking_blocks=[
            {"type": "thinking", "thinking": "visible summary", "signature": "sig"}
        ],
        provider_specific_fields={"thought_signatures": ["opaque"]},
    )
    result = SimpleNamespace(message=message, raw_content="", reasoning_content="")

    ToolRunner.append_assistant_message(
        _ctx(),
        ClassifiedCalls(regular=[_call("c1", "read_file")]),
        messages,
        result,
    )

    assistant = messages[0]
    assert assistant["reasoning_content"] == "visible summary"
    assert assistant["thinking_blocks"][0]["signature"] == "sig"
    assert assistant["provider_specific_fields"]["thought_signatures"] == ["opaque"]
    assert assistant["tool_calls"][0]["id"] == "c1"


def test_tool_runner_defers_compaction_until_context_pressure(monkeypatch):
    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("tool-round compaction bypassed context pressure")

    monkeypatch.setattr(ContextManager, "compact_for_small", fail_if_called)
    monkeypatch.setattr(ContextManager, "compact_old_tool_results", fail_if_called)

    messages = _run(ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path":"module.py"}'),
    ]))

    assert_tool_block_contiguous(messages)


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
    nudge = messages[-1]["content"]
    assert "If the success criterion is already verified" in nudge
    assert "Call step_complete now" not in nudge


def test_budget_nudge_does_not_split_a_multi_batch_turn():
    """A write tool gets a batch of its own, so batch 1 ended mid-block."""
    messages = _run(ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "create_file", '{"file_path": "b.py", "content": "x"}'),
    ]))
    assert_tool_block_contiguous(messages)
    assert [m["role"] for m in messages].count("tool") == 2


def test_budget_refuses_overflow_without_executing_or_breaking_transcript():
    ctx = _ctx()
    ctx.max_per_action = 1
    ctx.max_total_calls = 1
    runner = ToolRunner(_engine(nudge_at=None))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    classified = ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "read_file", '{"file_path": "b.py"}'),
    ])
    result = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]), raw_content="",
    )

    used = runner.run_regular(
        ctx, classified, messages, result, action_tool_calls=0, iteration=0,
        guard=LoopGuard(is_small=False), tracker=BehaviorTracker(set()),
    )

    assert used == 1
    assert ctx.state.total_tool_calls == 1
    overflow = next(message for message in messages if message.get("tool_call_id") == "c2")
    assert "not_run: tool budget exhausted" in overflow["content"]
    assert_tool_block_contiguous(messages)


def test_unlimited_total_budget_does_not_truncate_a_tool_batch() -> None:
    ctx = _ctx()
    ctx.max_per_action = 2
    ctx.max_total_calls = None
    ctx.state.total_tool_calls = 160
    runner = ToolRunner(_engine(nudge_at=None))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    classified = ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "read_file", '{"file_path": "b.py"}'),
    ])
    result = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]), raw_content="",
    )

    used = runner.run_regular(
        ctx, classified, messages, result, action_tool_calls=0, iteration=0,
        guard=LoopGuard(is_small=False), tracker=BehaviorTracker(set()),
    )

    assert used == 2
    assert ctx.state.total_tool_calls == 162
    assert not any(
        "not_run: tool budget exhausted" in message.get("content", "")
        for message in messages
    )
    assert_tool_block_contiguous(messages)
def test_unlimited_step_executes_the_full_batch_without_budget_nudge() -> None:
    ctx = _ctx()
    ctx.max_per_action = 0
    ctx.step_tool_limit = None
    ctx.max_total_calls = None
    runner = ToolRunner(_engine(nudge_at=1))
    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
    classified = ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "read_file", '{"file_path": "b.py"}'),
        _call("c3", "read_file", '{"file_path": "c.py"}'),
    ])
    result = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]), raw_content="",
    )

    used = runner.run_regular(
        ctx, classified, messages, result, action_tool_calls=0, iteration=0,
        guard=LoopGuard(is_small=False), tracker=BehaviorTracker(set()),
    )

    assert used == 3
    assert ctx.state.total_tool_calls == 3
    tool_bodies = [
        message["content"] for message in messages if message.get("role") == "tool"
    ]
    assert any("[Tool call 3 for this step]" in body for body in tool_bodies)
    assert not any("STEP BUDGET" in body for body in tool_bodies)
    assert not any(
        "not_run: tool budget exhausted" in message.get("content", "")
        for message in messages
    )
    assert_tool_block_contiguous(messages)



def test_step_complete_ack_stays_inside_the_block():
    messages = _run(ClassifiedCalls(
        regular=[_call("c1", "read_file", '{"file_path": "a.py"}')],
        step_complete=_call("c2", "step_complete", '{"status": "continue"}'),
    ))
    assert_tool_block_contiguous(messages)


def test_scheduler_owned_plan_call_is_an_ordered_noop_not_an_error():
    ctx = _ctx()
    ctx.allow_plan_mutation = False
    runner = ToolRunner(_engine(nudge_at=None))
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
    ]
    classified = ClassifiedCalls(regular=[
        _call("c1", "add_step", '{"title":"Duplicate scheduler work"}'),
    ])
    result = SimpleNamespace(
        message=SimpleNamespace(content="", tool_calls=[]), raw_content="",
    )

    used = runner.run_regular(
        ctx, classified, messages, result, action_tool_calls=0, iteration=0,
        guard=LoopGuard(is_small=False), tracker=BehaviorTracker(set()),
    )

    tool_result = next(
        message["content"] for message in messages
        if message.get("tool_call_id") == "c1"
    )
    assert used == 1
    assert ctx.state.total_tool_calls == 1
    assert '"reason": "scheduler_owned"' in tool_result
    assert "Unknown tool" not in tool_result
    assert ctx.state.plan.steps == []
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


def test_mutating_shell_refreshes_and_evicts_cached_files(tmp_path):
    kept = tmp_path / "kept.py"
    removed = tmp_path / "removed.py"
    kept.write_text("old = True\n")
    removed.write_text("gone = False\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    ctx.state.cache_file(str(kept), "stale")
    ctx.state.cache_file(str(removed), "stale")

    kept.write_text("fresh = True\n")
    removed.unlink()
    ToolRunner._refresh_opened_files_after_shell(
        ctx, {"command": "mv source.py destination.py"},
    )

    assert "fresh = True" in ctx.state.opened_files[str(kept)].content
    assert str(removed) not in ctx.state.opened_files


def test_proven_read_only_shell_keeps_cache_without_rereading(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("disk = 'new'\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    ctx.state.cache_file(str(path), "known-current")

    ToolRunner._refresh_opened_files_after_shell(ctx, {"command": "ls -la"})

    assert ctx.state.opened_files[str(path)].content == "known-current"


def test_exact_unchanged_read_is_replaced_by_compact_notice(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    call = _call(
        "read-1",
        "read_file",
        '{"file_path":"module.py","offset":1,"limit":20}',
    )

    cached, executable = runner._partition_repeated_reads(ctx, [call])
    assert cached == []
    assert executable == [call]

    runner._record_read_delivery(ctx, call)
    cached, executable = runner._partition_repeated_reads(ctx, [call])

    assert executable == []
    assert json.loads(cached[0][1]) == {
        "status": "already_delivered",
        "path": str(path),
        "coverage": "exact",
        "message": (
            "This unchanged read is already fully covered by prior results. Use "
            "the existing opened-files/prior evidence; running it again would add "
            "no information."
        ),
    }


def test_minimax_recovery_redelivers_an_exact_cached_read_under_pressure(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    ctx.suppress_discovery_this_step = True
    ctx.unlimited_recovery_reads = True
    ctx.max_context_tokens = 1_000_000
    ctx.state.last_prompt_tokens = 800_000
    runner = ToolRunner(_engine(nudge_at=99))
    call = _call(
        "read-1",
        "read_file",
        '{"file_path":"module.py","offset":1,"limit":20}',
    )
    runner._record_read_delivery(ctx, call)

    cached, executable = runner._partition_repeated_reads(ctx, [call])

    assert cached == []
    assert executable == [call]


def test_minimax_recovery_hides_a_reread_while_target_source_is_live(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    ctx.suppress_discovery_this_step = True
    ctx.unlimited_recovery_reads = True
    runner = ToolRunner(_engine(nudge_at=99))
    call = _call("read-1", "read_file", '{"file_path":"module.py"}')
    runner._record_read_delivery(ctx, call)

    suppressed, executable = runner._partition_suppressed_discovery(ctx, [call])

    assert executable == []
    payload = json.loads(suppressed[0][1])
    assert payload["status"] == "discovery_suppressed"
    assert "already present in the full live transcript" in payload["reason"]


def test_read_delivery_cache_allows_uncovered_range_and_invalidates_on_edit(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("value = 1\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    original = _call("read-1", "read_file", '{"file_path":"module.py","limit":20}')
    uncovered = _call(
        "read-2",
        "read_file",
        '{"file_path":"module.py","offset":21,"limit":5}',
    )
    runner._record_read_delivery(ctx, original)

    cached, executable = runner._partition_repeated_reads(ctx, [uncovered])
    assert cached == []
    assert executable == [uncovered]

    path.write_text("value = 200\n")
    cached, executable = runner._partition_repeated_reads(ctx, [original])
    assert cached == []
    assert executable == [original]


def test_read_delivery_cache_normalizes_aliases_and_covered_subranges(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("".join(f"line_{line}\n" for line in range(1, 81)))
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))

    first = _call(
        "read-1",
        "read_file",
        '{"file_path":"module.py","offset":"10","limit":"21"}',
    )
    alias = _call(
        "read-2",
        "partial_read",
        '{"path":"module.py","start_line":10,"end_line":30}',
    )
    contained = _call(
        "read-3",
        "read_file",
        '{"file_path":"module.py","line_range":"15-20"}',
    )
    runner._record_read_delivery(ctx, first)

    alias_cached, alias_executable = runner._partition_repeated_reads(ctx, [alias])
    contained_cached, contained_executable = runner._partition_repeated_reads(
        ctx, [contained]
    )

    assert alias_executable == []
    assert json.loads(alias_cached[0][1])["coverage"] == "exact"
    assert contained_executable == []
    assert json.loads(contained_cached[0][1])["coverage"] == "contained"


def test_read_delivery_cache_unions_ranges_but_does_not_bridge_gaps(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("".join(f"line_{line}\n" for line in range(1, 81)))
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))

    runner._record_read_delivery(
        ctx,
        _call("read-1", "read_file", '{"file_path":"module.py","offset":1,"limit":10}'),
    )
    runner._record_read_delivery(
        ctx,
        _call("read-2", "read_file", '{"file_path":"module.py","offset":11,"limit":10}'),
    )
    covered = _call(
        "read-3", "read_file", '{"file_path":"module.py","offset":5,"limit":15}'
    )
    gap = _call(
        "read-4", "read_file", '{"file_path":"module.py","offset":5,"limit":25}'
    )

    cached, executable = runner._partition_repeated_reads(ctx, [covered, gap])

    assert [call.id for call, _ in cached] == ["read-3"]
    assert executable == [gap]


def test_shell_head_reuses_covered_native_and_shell_ranges(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("".join(f"line_{line}\n" for line in range(1, 121)))
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))

    first_head = _call(
        "shell-1",
        "execute_command",
        '{"command":"head -n 100 module.py","cwd":"."}',
    )
    smaller_head = _call(
        "shell-2",
        "execute_command",
        '{"command":"head --lines=80 module.py","cwd":"."}',
    )
    native_subset = _call(
        "read-3",
        "read_file",
        '{"file_path":"module.py","offset":20,"limit":20}',
    )
    runner._record_read_delivery(ctx, first_head)

    cached, executable = runner._partition_repeated_reads(
        ctx, [smaller_head, native_subset]
    )

    assert [call.id for call, _ in cached] == ["shell-2", "read-3"]
    assert executable == []
    assert all(json.loads(result)["coverage"] == "contained" for _, result in cached)


def test_shell_cat_covers_later_ranges_and_respects_file_revision(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("one\ntwo\nthree\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    cat = _call(
        "shell-1",
        "execute_command",
        '{"command":"cd . && cat module.py"}',
    )
    subset = _call(
        "read-2", "read_file", '{"file_path":"module.py","offset":2,"limit":1}'
    )
    runner._record_read_delivery(ctx, cat)

    cached, executable = runner._partition_repeated_reads(ctx, [subset])
    assert executable == []
    assert json.loads(cached[0][1])["coverage"] == "contained"

    path.write_text("one\nchanged\nthree\n")
    cached, executable = runner._partition_repeated_reads(ctx, [subset])
    assert cached == []
    assert executable == [subset]


def test_shell_read_parser_abstains_on_pipeline_and_unknown_transform(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("one\ntwo\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    calls = [
        _call("shell-1", "execute_command", '{"command":"cat module.py | head"}'),
        _call("shell-2", "execute_command", '{"command":"sort module.py"}'),
    ]

    cached, executable = runner._partition_repeated_reads(ctx, calls)

    assert cached == []
    assert executable == calls


def test_range_coverage_handles_same_start_with_bounded_and_open_end(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("\n".join(str(index) for index in range(100)))
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    bounded = _call(
        "read-1", "read_file",
        '{"file_path":"module.py","offset":10,"limit":5}',
    )
    open_ended = _call(
        "read-2", "read_file",
        '{"file_path":"module.py","offset":10}',
    )
    runner._record_read_delivery(ctx, bounded)
    runner._record_read_delivery(ctx, open_ended)
    contained = _call(
        "read-3", "read_file",
        '{"file_path":"module.py","offset":15,"limit":20}',
    )

    cached, executable = runner._partition_repeated_reads(ctx, [contained])

    assert executable == []
    assert json.loads(cached[0][1])["coverage"] == "contained"


def test_exact_shell_search_is_cached_until_an_input_file_changes(tmp_path):
    path = tmp_path / "module.py"
    path.write_text("needle = 1\nother = 2\n")
    ctx = _ctx()
    ctx.workspace_path = str(tmp_path)
    runner = ToolRunner(_engine(nudge_at=99))
    first = _call(
        "shell-1",
        "execute_command",
        '{"command":"grep -n needle module.py","cwd":"."}',
    )
    repeated = _call(
        "shell-2",
        "execute_command",
        '{"command":"cd . && grep -n needle module.py"}',
    )
    runner._record_read_delivery(ctx, first)

    cached, executable = runner._partition_repeated_reads(ctx, [repeated])
    assert executable == []
    assert json.loads(cached[0][1])["coverage"] == "exact"

    path.write_text("needle = 3\nother = 2\n")
    cached, executable = runner._partition_repeated_reads(ctx, [repeated])
    assert cached == []
    assert executable == [repeated]


def test_semantic_stagnation_step_suppresses_discovery_but_allows_tests(tmp_path):
    ctx = _ctx()
    ctx.suppress_discovery_this_step = True
    runner = ToolRunner(_engine(nudge_at=99))
    calls = [
        _call("read-1", "read_file", '{"file_path":"module.py"}'),
        _call(
            "shell-1",
            "execute_command",
            '{"command":"grep -n needle module.py"}',
        ),
        _call(
            "test-1",
            "execute_command",
            '{"command":"pytest tests/test_module.py -q"}',
        ),
        _call("edit-1", "edit_file", '{}'),
    ]

    suppressed, executable = runner._partition_suppressed_discovery(ctx, calls)

    assert [call.id for call, _ in suppressed] == ["read-1", "shell-1"]
    assert [call.id for call in executable] == ["test-1", "edit-1"]
    assert all(
        json.loads(result)["status"] == "discovery_suppressed"
        for _, result in suppressed
    )


def test_semantic_recovery_spends_two_local_reads_then_suppresses_discovery(tmp_path):
    ctx = _ctx()
    ctx.suppress_discovery_this_step = True
    ctx.semantic_recovery_context_calls = 2
    runner = ToolRunner(_engine(nudge_at=99))
    calls = [
        _call("read-1", "read_file", '{"file_path":"module.py"}'),
        _call(
            "shell-1", "execute_command",
            '{"command":"grep -n needle module.py"}',
        ),
        _call("web-1", "web_search", '{"query":"module implementation"}'),
        _call("edit-1", "edit_file", '{}'),
    ]

    suppressed, executable = runner._partition_suppressed_discovery(ctx, calls)

    assert [call.id for call, _ in suppressed] == ["web-1"]
    assert [call.id for call in executable] == ["read-1", "shell-1", "edit-1"]
    assert ctx.semantic_recovery_context_calls == 0


def test_minimax_recovery_keeps_direct_reads_and_freezes_all_plan_mutation():
    ctx = _ctx()
    ctx.suppress_discovery_this_step = True
    ctx.semantic_recovery_context_calls = 0
    ctx.recovery_direct_reads_only = True
    ctx.unlimited_recovery_reads = True
    runner = ToolRunner(_engine(nudge_at=99))
    ctx.freeze_plan_growth_in_recovery = True
    calls = [
        _call("read-1", "read_file", '{"file_path":"module.py"}'),
        _call("read-2", "read_file", '{"file_path":"other.py"}'),
        _call(
            "shell-1",
            "execute_command",
            '{"command":"grep -n needle module.py"}',
        ),
        _call("plan-1", "add_step", '{"title":"Investigate alternatives"}'),
        _call("plan-2", "modify_step", '{"index":1,"title":"Switch option"}'),
    ]

    suppressed, executable = runner._partition_suppressed_discovery(ctx, calls)

    assert [call.id for call, _ in suppressed] == ["shell-1", "plan-1", "plan-2"]
    assert [call.id for call in executable] == ["read-1", "read-2"]
    assert ctx.semantic_recovery_context_calls == 0


def test_repeated_test_checkpoint_is_minimax_policy_conditional():
    from infinidev.engine.guidance import normalize_test_command

    ctx = _ctx()
    runner = ToolRunner(_engine(nudge_at=99))
    call = _call(
        "test-1",
        "execute_command",
        '{"command":"pytest tests/test_module.py -q"}',
    )
    key = normalize_test_command("pytest tests/test_module.py -q")
    ctx.state.test_workspace_fingerprints[key] = ()
    ctx.state.test_outcome_history[key] = ["passed:1"]

    ctx.reuse_unchanged_test_results = True
    cached, executable = runner._partition_repeated_tests(ctx, [call])
    assert executable == []
    assert json.loads(cached[0][1])["status"] == "test_already_run"

    ctx.reuse_unchanged_test_results = False
    cached, executable = runner._partition_repeated_tests(ctx, [call])
    assert cached == []
    assert executable == [call]

def test_cargo_test_checkpoint_normalizes_wrappers_and_argument_separator():
    from infinidev.engine.guidance import normalize_test_command

    first = (
        "timeout 300 cargo test -p infinigpu-device --lib --no-fail-fast "
        "forwarded_cmdlist_decodes 2>&1 | tail -20"
    )
    repeated = (
        "RUST_BACKTRACE=1 cargo test -p infinigpu-device --lib "
        "--no-fail-fast -- forwarded_cmdlist_decodes"
    )
    assert normalize_test_command(first) == normalize_test_command(repeated)

    ctx = _ctx()
    ctx.reuse_unchanged_test_results = True
    key = normalize_test_command(first)
    ctx.state.test_workspace_fingerprints[key] = ()
    ctx.state.test_outcome_history[key] = ["1 failed"]
    call = _call(
        "test-2", "execute_command", json.dumps({"command": repeated})
    )

    cached, executable = ToolRunner(_engine())._partition_repeated_tests(
        ctx, [call]
    )

    assert executable == []
    assert json.loads(cached[0][1])["status"] == "test_already_run"


def test_no_run_command_does_not_create_test_checkpoint():
    ctx = _ctx()
    ctx.reuse_unchanged_test_results = True
    arguments = json.dumps({"command": "cargo test --lib --no-run"})
    result = json.dumps({
        "exit_code": 0,
        "stdout": "Finished test profile; executable generated",
    })

    ToolRunner.capture_test_output(ctx, arguments, result)

    assert ctx.state.test_outcome_history == {}
    assert ctx.state.test_workspace_fingerprints == {}



def test_workspace_change_releases_minimax_test_checkpoint(tmp_path):
    from infinidev.engine.guidance import normalize_test_command

    ctx = _ctx()
    ctx.reuse_unchanged_test_results = True
    runner = ToolRunner(_engine(nudge_at=99))
    call = _call("test-1", "execute_command", '{"command":"pytest -q"}')
    key = normalize_test_command("pytest -q")
    ctx.state.test_workspace_fingerprints[key] = ()
    path = tmp_path / "module.py"
    ctx.file_tracker.record(str(path), "before\n", "after\n")

    cached, executable = runner._partition_repeated_tests(ctx, [call])

    assert cached == []
    assert executable == [call]


def test_semantic_recovery_does_not_treat_unknown_shell_as_progress():
    ctx = _ctx()
    ctx.suppress_discovery_this_step = True
    runner = ToolRunner(_engine(nudge_at=99))
    calls = [
        _call(
            "python-inspect",
            "execute_command",
            '{"command":"python -c \\"print(open(\'module.py\').read())\\""}',
        ),
        _call(
            "test-1",
            "execute_command",
            '{"command":"python -m pytest tests/test_module.py -q"}',
        ),
    ]

    suppressed, executable = runner._partition_suppressed_discovery(ctx, calls)

    assert [call.id for call, _ in suppressed] == ["python-inspect"]
    assert [call.id for call in executable] == ["test-1"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"path": "a.py"}, {"path": "a.py"}),
        ('{"path":"a.py"}', {"path": "a.py"}),
        ('{"path":"a.py"}</tool_call>', {"path": "a.py"}),
        ("not json", {}),
    ],
)
def test_assistant_tool_arguments_are_provider_safe_json_strings(raw, expected):
    """MiniMax rejects decoded dicts and malformed strings on the next turn."""
    messages = _run(
        ClassifiedCalls(regular=[_call("c1", "read_file", raw)]),
        nudge_at=99,
    )

    arguments = messages[2]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(arguments, str)
    assert json.loads(arguments) == expected


def test_nudge_fires_once_across_batches():
    """Equality on the counter, so a multi-batch turn still warns once."""
    messages = _run(ClassifiedCalls(regular=[
        _call("c1", "read_file", '{"file_path": "a.py"}'),
        _call("c2", "create_file", '{"file_path": "b.py", "content": "x"}'),
        _call("c3", "read_file", '{"file_path": "c.py"}'),
    ]))
    nudges = [m for m in messages if m.get("role") == "user" and m is not messages[1]]
    assert len(nudges) == 1


# ── command-output descriptor propagation ─────────────────────────────


def _append_command_result(result: str):
    ctx = _ctx()
    runner = ToolRunner(_engine(nudge_at=99))
    messages: list[dict] = []
    runner._append_results(
        ctx,
        [(_call("cmd-1", "execute_command", '{"command": "pytest"}'), result)],
        messages,
        action_tool_calls=0,
        iteration=0,
        is_parallel=False,
        guard=LoopGuard(is_small=False),
        tracker=BehaviorTracker(set()),
        tool_results_text=[],
        deferred=[],
    )
    return ctx, messages


def test_no_handle_preserves_result_return_and_archive_view(monkeypatch):
    original = json.dumps(
        {"exit_code": 0, "stdout": "ok", "stderr": "", "success": True}
    )
    monkeypatch.setattr(
        ToolRunner,
        "capture_test_output",
        staticmethod(lambda ctx, args, result: result),
    )

    ctx, messages = _append_command_result(original)

    expected_body = original + "\n[Tool call 1/4 for this step]"
    assert messages == [
        {"role": "tool", "tool_call_id": "cmd-1", "content": expected_body}
    ]
    assert ctx.state.pending_archive == [
        ("execute_command", '{"command": "pytest"}', expected_body)
    ]
    assert not hasattr(ctx, "pending_command_output_handles")


def test_valid_handle_is_propagated_without_changing_model_view(monkeypatch):
    original = json.dumps({
        "exit_code": 0,
        "stdout": "tail",
        "stderr": "",
        "success": True,
        "command_output_handles": {
            "stdout": {
                "artifact_id": 41,
                "type": "command_output",
                "stream": "stdout",
                "char_count": 12001,
                "byte_count": 12001,
            }
        },
    })
    monkeypatch.setattr(
        ToolRunner,
        "capture_test_output",
        staticmethod(lambda ctx, args, result: result),
    )

    ctx, messages = _append_command_result(original)

    assert messages[0]["content"].startswith(original)
    assert ctx.state.pending_archive[0][2] == messages[0]["content"]
    assert ctx.pending_command_output_handles == [{
        "artifact_id": 41,
        "type": "command_output",
        "stream": "stdout",
        "char_count": 12001,
        "byte_count": 12001,
        "tool_call_id": "cmd-1",
    }]


def test_invalid_handle_is_not_propagated(monkeypatch):
    original = json.dumps({
        "exit_code": 0,
        "stdout": "tail",
        "stderr": "",
        "success": True,
        "command_output_handles": {
            "stdout": {
                "artifact_id": 41,
                "type": "command_output",
                "stream": "stderr",
                "char_count": 12001,
                "byte_count": 12001,
            }
        },
    })
    monkeypatch.setattr(
        ToolRunner,
        "capture_test_output",
        staticmethod(lambda ctx, args, result: result),
    )

    ctx, messages = _append_command_result(original)

    assert messages[0]["content"].startswith(original)
    assert not hasattr(ctx, "pending_command_output_handles")


def test_denied_test_command_does_not_replace_verifier_command(monkeypatch):
    def unexpected_capture(*args, **kwargs):
        raise AssertionError("denied commands must not be captured as test runs")

    monkeypatch.setattr(
        ToolRunner,
        "capture_test_output",
        staticmethod(unexpected_capture),
    )

    ctx, messages = _append_command_result(
        json.dumps({"error": "Command denied by user: pytest | tail"})
    )

    assert ctx.state.last_test_command == ""
    assert "Command denied by user" in messages[0]["content"]


def test_failed_diagnostic_does_not_replace_last_passing_test_command():
    ctx = _ctx()

    ToolRunner.capture_test_output(
        ctx,
        json.dumps({"command": "pytest tests/test_focused.py"}),
        json.dumps({"exit_code": 0, "stdout": "1 passed", "stderr": ""}),
    )
    ToolRunner.capture_test_output(
        ctx,
        json.dumps({"command": "pytest"}),
        json.dumps({"exit_code": 1, "stdout": "3 failed", "stderr": ""}),
    )

    assert ctx.state.last_test_command == "pytest"
    assert ctx.state.last_test_exit_code == 1
    assert ctx.state.last_passing_test_command == "pytest tests/test_focused.py"


def test_masked_cargo_failure_overrides_shell_zero_for_feedback_and_state():
    ctx = _ctx()
    raw = json.dumps({
        "exit_code": 0,
        "success": True,
        "stdout": (
            "test result: FAILED. 54 passed; 3 failed\n"
            "error: test failed\nEXIT=101\n"
        ),
        "stderr": "",
    })

    result = ToolRunner.capture_test_output(
        ctx,
        json.dumps({
            "command": (
                "cargo test -p infinigpu-device --lib | tail -15; "
                "echo EXIT=$?"
            ),
        }),
        raw,
    )
    payload = json.loads(result)

    assert payload["shell_exit_code"] == 0
    assert payload["exit_code"] == 101
    assert payload["success"] is False
    assert "shell's final zero" in payload["status_note"]
    assert ctx.state.last_test_exit_code == 101
