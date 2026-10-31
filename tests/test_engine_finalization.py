"""Task completion must respect validation, fresh instructions and cancellation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop.context_builder import build_iteration_messages
from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.loop_plan import LoopPlan, PlanStep
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.step_result import StepResult


class _Finishes:
    def __init__(self):
        self.calls = []

    def finish(self, ctx, status, iteration, result=None):
        self.calls.append((status, result))
        return result if result is not None else status


@pytest.fixture
def run():
    engine = LoopEngine()
    engine._cr_hooks = SimpleNamespace(_enabled=False)
    state = LoopState(plan=LoopPlan(steps=[PlanStep(index=1, title="Work", status="done")]))
    ctx = SimpleNamespace(
        state=state, file_tracker=FileChangeTracker(), project_id=1, agent_id="developer",
        start_iteration=0, guardrail=None, guardrail_max_retries=0, llm_params={},
        system_prompt="Developer", desc="Implement rollback", expected="Rollback verified",
        tool_schemas=[], tool_dispatch={}, max_per_action=0, max_prompt_tokens=None,
        max_total_calls=None, manual_tc=False, task=None, history_window=0,
        max_context_tokens=1_000_000, skip_plan=False, is_small=False, resumed=False,
    )
    return engine, ctx, _Finishes()


def _close(run, *, status="done", count=0):
    engine, ctx, finishes = run
    step_result = StepResult(
        summary="Original answer", final_answer="Original answer", status=status,
    )
    result = engine._check_termination(ctx, step_result, finishes, 2, count)
    return result, step_result


@pytest.mark.parametrize(("validation", "expected"), [
    (False, "blocked"), ((False, "rollback is broken"), "blocked"),
])
def test_rejected_final_result_cannot_be_done(run, validation, expected):
    engine, ctx, finishes = run
    ctx.guardrail = lambda text: validation

    result, _ = _close(run)

    assert finishes.calls[0][0] == expected
    assert "Partial result" in result


def test_validator_failure_cannot_be_done(run):
    engine, ctx, finishes = run

    def broken(text):
        raise RuntimeError("validator unavailable")

    ctx.guardrail = broken
    result, _ = _close(run)

    assert finishes.calls[0][0] == "failed"
    assert "validator unavailable" in result


def test_cancellation_during_positive_validation_wins(run):
    engine, ctx, finishes = run

    def validate(text):
        engine.cancel()
        return True

    ctx.guardrail = validate
    _close(run)

    assert finishes.calls[0][0] == "cancelled"


@pytest.mark.parametrize("arrival", ["before_validation", "during_validation"])
def test_new_user_input_defers_completion_without_losing_or_duplicating_it(run, arrival):
    engine, ctx, finishes = run
    instruction = "Also verify rollback of externally assigned identifiers."
    if arrival == "before_validation":
        engine.inject_message(instruction)

    def validate(text):
        if arrival == "during_validation":
            engine.inject_message(instruction)
        return True

    ctx.guardrail = validate
    result, step_result = _close(run)

    assert result is None
    assert finishes.calls == []
    assert step_result.interrupted
    assert [entry.text for entry in ctx.state.user_guidance] == [instruction]
    prompt = build_iteration_messages(engine, ctx, 3)[1]["content"]
    assert instruction in prompt
    assert "urgent-user-message" in prompt
    assert [entry.text for entry in ctx.state.user_guidance] == [instruction]


def test_late_input_is_checkpointed_even_if_no_further_model_budget_remains(run, temp_db):
    from infinidev.db.service import get_session_runtime_state, register_session
    from infinidev.tools.base.context import set_context

    engine, ctx, finishes = run
    register_session("late-completion", "/work")
    set_context(agent_id="developer", session_id="late-completion")
    ctx.max_total_calls = 1
    ctx.state.total_tool_calls = 1
    engine.inject_message("Keep the original identifier mapping.")

    result, _ = _close(run)

    assert result is None
    raw = get_session_runtime_state("late-completion")["ui_state"]["loop_resume"]["state"]
    assert raw["user_guidance"][0]["text"] == "Keep the original identifier mapping."
    assert finishes.calls == []


def test_finished_plan_requests_an_explicit_task_decision_instead_of_claiming_success(run):
    engine, ctx, finishes = run
    result, _ = _close(run, status="continue", count=2)

    assert result is None
    assert finishes.calls == []
    assert ctx.state.pending_engine_notice
    build_iteration_messages(engine, ctx, 3)
    _close(run, status="continue", count=3)
    assert ctx.state.pending_engine_notice == ""


def test_pending_retention_and_drain_are_idempotent_but_equal_texts_remain_distinct():
    from infinidev.engine.loop.user_message_injector import UserMessageInjector

    injector = UserMessageInjector()
    state = LoopState()
    injector.inject("Keep IDs")
    injector.inject("Keep IDs")

    assert injector.has_pending
    injector.retain_pending(state)
    injector.retain_pending(state)
    injector.drain_for_state(state)

    assert not injector.has_pending
    assert [entry.text for entry in state.user_guidance] == ["Keep IDs", "Keep IDs"]
    assert len({entry.message_id for entry in state.user_guidance}) == 2


@pytest.mark.parametrize("stop_task", [False, True])
def test_corrective_tools_use_foreground_and_cooperative_cancellation_scope(
    run, monkeypatch, stop_task,
):
    from infinidev.engine.loop import guardrail_runner
    from infinidev.engine.tool_progress import is_tool_cancelled

    engine, ctx, finishes = run
    ctx.state.iteration_count = 3
    ctx.guardrail_max_retries = 1
    ctx.guardrail = lambda text: (text == "CORRECTED", "fix it")
    observations = []
    model_calls = []

    class CooperativeTool:
        name = "cooperative"

        def _run(self):
            observations.append(engine.has_active_tool)
            if stop_task:
                engine.cancel()
            else:
                assert engine.cancel_active_tool()
            observations.append(is_tool_cancelled())
            return '{"partial_output":"work before interruption"}'

    def model(*args):
        model_calls.append(1)
        calls = [SimpleNamespace(
            id="tool", function=SimpleNamespace(name="cooperative", arguments="{}"),
        )] if len(model_calls) == 1 else None
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(
            message=SimpleNamespace(content="CORRECTED" if not calls else "", tool_calls=calls),
        )])

    ctx.tool_dispatch = {"cooperative": CooperativeTool()}
    monkeypatch.setattr(guardrail_runner, "_call_llm", model)

    outcome = engine._validate_final_result(ctx, "ORIGINAL")

    assert observations == [True, True]
    assert not engine.has_active_tool
    assert not engine._tool_cancel_event.is_set()
    assert outcome.status == ("cancelled" if stop_task else "done")
    assert len(model_calls) == (1 if stop_task else 2)


def test_guidance_delivered_inside_correction_still_requires_task_reassessment(run, monkeypatch):
    from infinidev.engine.loop import guardrail_runner

    engine, ctx, finishes = run
    ctx.state.iteration_count = 3
    ctx.guardrail_max_retries = 2
    ctx.guardrail = lambda text: (text == "CORRECTED", "fix it")
    calls = []

    def model(*args):
        calls.append(1)
        if len(calls) == 1:
            engine.inject_message("Also preserve externally assigned identifiers.")
        return SimpleNamespace(usage=None, choices=[SimpleNamespace(message=SimpleNamespace(
            content="DRAFT" if len(calls) == 1 else "CORRECTED", tool_calls=None,
        ))])

    monkeypatch.setattr(guardrail_runner, "_call_llm", model)
    result, step_result = _close(run)

    assert result is None
    assert finishes.calls == []
    assert step_result.interrupted
    assert not engine._user_message_injector.has_pending
    assert ctx.state.user_guidance[0].text == "Also preserve externally assigned identifiers."
    assert ctx.state.pending_engine_notice


@pytest.mark.parametrize("arrival", ["before_validation", "during_validation"])
def test_forwarded_worker_guidance_blocks_stale_completion_without_consuming_delivery(run, arrival):
    engine, ctx, finishes = run
    engine._team_actor = "worker-1"
    events = []
    instruction = "Also verify external identifiers."
    event = {"id": 7, "kind": "user_guidance", "author": "user", "content": instruction}

    def poll(actor):
        result = json.dumps(events)
        events.clear()
        return result

    engine._team_runtime = SimpleNamespace(
        pending_user_guidance=lambda actor: list(events),
        guidance_attachments=lambda event_id: [],
        poll=poll,
    )
    if arrival == "before_validation":
        events.append(event)
    else:
        def validate(text):
            events.append(event)
            return True

        ctx.guardrail = validate

    result, step_result = _close(run)

    assert result is None
    assert finishes.calls == []
    assert step_result.interrupted
    assert [entry.text for entry in ctx.state.user_guidance] == [instruction]
    assert events == [event]
    messages = []
    engine._inject_mid_step_user_messages(ctx, messages)
    assert events == []
    assert instruction in str(messages)
    assert len(ctx.state.user_guidance) == 1


def test_team_guidance_snapshot_preserves_user_and_peer_delivery(temp_db, workspace_dir):
    from infinidev.engine.team.runtime import ROOT, TeamRuntime

    runtime = TeamRuntime(
        session_id="finalization-inbox", project_id=1, workspace_path=str(workspace_dir),
        root_agent_id="root", catalog=[],
    )
    try:
        runtime.poll(ROOT)
        runtime.store.update(lambda state, emit: emit(
            "message", "peer", "Check the remaining evidence", recipient=ROOT,
        ))
        runtime.forward_user_guidance("Keep external IDs")

        pending = runtime.pending_user_guidance(ROOT)
        assert [event["content"] for event in pending] == ["Keep external IDs"]
        assert runtime.pending_user_guidance(ROOT) == pending
        delivered = json.loads(runtime.poll(ROOT))
        assert {event["content"] for event in delivered} == {
            "Keep external IDs", "Check the remaining evidence",
        }
        assert runtime.pending_user_guidance(ROOT) == []
    finally:
        runtime.close()
