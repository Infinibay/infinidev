"""A correctable tool-name miss cannot immediately end an autonomous Task."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep


def _ctx() -> SimpleNamespace:
    state = LoopState()
    state.plan.steps = [PlanStep(index=1, title="Run pwd", status="active")]
    return SimpleNamespace(
        state=state,
        project_id=1,
        agent_id="m3-live-recovery",
        workspace_path=".",
    )


def _blocked_call() -> SimpleNamespace:
    return SimpleNamespace(
        id="complete-1",
        function=SimpleNamespace(
            arguments='{"summary":"no shell","status":"blocked"}',
        ),
    )


def _messages(previous_result: str) -> list[dict[str, str]]:
    return [
        {"role": "tool", "tool_call_id": "attempt-1", "content": previous_result},
        {"role": "tool", "tool_call_id": "complete-1", "content": "ok"},
    ]


def test_unknown_tool_suggestion_requires_one_corrected_attempt() -> None:
    engine = LoopEngine()
    call = _blocked_call()
    messages = _messages(
        '{"error":"Unknown tool: shell_command. '
        'Did you mean one of: execute_command?"}',
    )

    assert engine._step_gate._recoverable_tool_error(_ctx(), call, messages) is True
    assert "Retry the intended operation once" in messages[-1]["content"]
    assert "status=\"blocked\"" in messages[-1]["content"]


def test_recovery_gate_is_bounded_to_one_correction_turn_per_step() -> None:
    engine = LoopEngine()
    ctx = _ctx()
    call = _blocked_call()
    messages = _messages(
        '{"error":"Unknown tool: shell_command. '
        'Did you mean one of: execute_command?"}',
    )

    assert engine._step_gate._recoverable_tool_error(ctx, call, messages) is True
    assert engine._step_gate._recoverable_tool_error(ctx, call, messages) is False


def test_model_reported_tool_surface_miss_requires_a_corrected_attempt() -> None:
    engine = LoopEngine()
    call = SimpleNamespace(
        id="complete-1",
        function=SimpleNamespace(arguments=(
            '{"summary":"The shell tool advertised to this turn is not callable",'
            '"status":"blocked"}'
        )),
    )
    messages = _messages('{"status":"ok"}')

    assert engine._step_gate._recoverable_tool_error(_ctx(), call, messages) is True
    assert "execute_command" in messages[-1]["content"]


def test_unrelated_tool_failure_can_still_report_blocked() -> None:
    engine = LoopEngine()
    messages = _messages('{"error":"Permission denied by user"}')

    assert engine._step_gate._recoverable_tool_error(
        _ctx(), _blocked_call(), messages,
    ) is False
