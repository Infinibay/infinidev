"""How a run ends, and what it says about how it ended.

Three ways the loop used to reach a wrong ending: it reported a user
cancellation as an exhausted iteration budget, it aborted a healthy run
for "failing to produce function calls" when the model had been calling
them all along, and it had one gate with no bound at all.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.loop.critic_liaison import CriticLiaison, _MAX_REJECTS_PER_STEP
from infinidev.engine.loop.loop_guard import LoopGuard, _MAX_PSEUDO_ONLY_ROUNDS
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.step_result import StepResult
from infinidev.engine.loop.step_summarizer import _synthesize_final
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.engine import (
    _apply_exploration_policy,
    _enforce_edit_requirement,
    _enforce_step_effect,
    _has_substantive_done_evidence,
    _reconcile_step_result,
    _resource_exhaustion_reason,
    _should_advance_plan,
    _task_requires_edits,
)


def _ctx() -> SimpleNamespace:
    return SimpleNamespace(
        state=LoopState(), project_id=1, agent_id="a1",
        manual_tc=False, is_small=False,
    )


# ── cancellation is a status, not a flag ─────────────────────────────


def test_cancelled_run_does_not_claim_the_iteration_limit():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="edited auth.py")]

    text = _synthesize_final(state, "cancelled")

    assert "iteration limit" not in text.lower(), (
        "a cancelled run reported itself as exhausted to the reviewer, the "
        "task_end hooks and the next turn's work summary"
    )
    assert "stopped by the user" in text.lower()
    assert "edited auth.py" in text, "what did happen is still reported"


def test_exhausted_run_still_says_iteration_limit():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="edited auth.py")]
    assert "iteration limit" in _synthesize_final(state, "exhausted").lower()


def test_prompt_token_budget_is_an_independent_resource_fuse():
    ctx = _ctx()
    ctx.max_total_calls = 40
    ctx.max_prompt_tokens = 300_000
    ctx.state.total_tool_calls = 12
    ctx.state.total_prompt_tokens = 300_001

    reason = _resource_exhaustion_reason(ctx)

    assert reason is not None
    assert "prompt token limit reached" in reason
    assert "300001/300000" in reason


def test_disabled_prompt_budget_leaves_tool_fuse_authoritative():
    ctx = _ctx()
    ctx.max_total_calls = 40
    ctx.max_prompt_tokens = None
    ctx.state.total_prompt_tokens = 900_000

    assert _resource_exhaustion_reason(ctx) is None

    ctx.state.total_tool_calls = 40
    assert "tool call limit reached" in _resource_exhaustion_reason(ctx)


def test_synthesize_final_defaults_to_exhausted():
    state = LoopState()
    state.history = [ActionRecord(step_index=1, summary="x")]
    assert "iteration limit" in _synthesize_final(state).lower()


def test_cancelled_with_no_history_says_so():
    assert "stopped by the user" in _synthesize_final(LoopState(), "cancelled").lower()


def test_cancel_survives_until_the_turn_ends():
    """execute() used to clear the flag, so the next phase resurrected the run."""
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    engine.cancel()
    assert engine.is_cancelled is True

    engine.begin_turn()
    assert engine.is_cancelled is False


# ── liveness is not the budget counter ───────────────────────────────


def test_a_step_closed_with_only_pseudo_tools_is_not_a_stall():
    """think + step_complete executes no regular tool and is not a stall."""
    result = StepResult(summary="done thinking", status="continue")
    result.action_tool_calls = 0
    result.saw_tool_calls = True

    assert result.saw_tool_calls, (
        "the abort reads this; action_tool_calls is a budget counter and "
        "reports 0 for a perfectly well-behaved step"
    )


def test_a_step_with_no_function_calls_at_all_is_a_stall():
    result = StepResult(summary="", status="continue")
    assert result.saw_tool_calls is False


def _tool_call(name: str, arguments: str = "{}", call_id: str = "tc"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _BudgetBoundaryCaller:
    def __init__(self) -> None:
        self.completion_modes: list[bool] = []

    def reset(self) -> None:
        pass

    def call(
        self, ctx, messages, is_planning, action_tool_calls=0, *,
        completion_only=False,
    ):
        from infinidev.engine.loop.llm_call_result import LLMCallResult

        self.completion_modes.append(completion_only)
        if not completion_only:
            return LLMCallResult(tool_calls=[_tool_call("read_file", call_id="read")])
        return LLMCallResult(tool_calls=[_tool_call(
            "step_complete",
            '{"summary":"verified","status":"done","final_answer":"done"}',
            call_id="complete",
        )])


def _run_budget_boundary(*, max_total_calls: int = 10):
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.loop.loop_guard import LoopGuard
    from infinidev.engine.loop.tool_processor import ToolProcessor

    engine = LoopEngine()
    caller = _BudgetBoundaryCaller()
    ctx = _ctx()
    ctx.max_per_action = 1
    ctx.max_total_calls = max_total_calls
    ctx.max_prompt_tokens = None
    ctx.allow_plan_mutation = True
    ctx.skip_plan = False
    engine._inject_mid_step_user_messages = lambda _ctx, _messages: None
    engine._guidance = SimpleNamespace(try_queue=lambda *args, **kwargs: None)
    engine._critic = SimpleNamespace(
        review_alongside=lambda _ctx, _messages, _calls, _reasoning, run: run(),
    )
    engine._step_gate = SimpleNamespace(blocks=lambda *args, **kwargs: False)
    engine._build_pseudo_only_messages = lambda *args, **kwargs: None

    def execute(
        _ctx, _classified, _messages, _result, action_tool_calls,
        _iteration, _guard, _tracker,
    ):
        _ctx.state.total_tool_calls += 1
        return action_tool_calls + 1

    engine._execute_regular_tools = execute
    result = engine._run_inner_loop(
        ctx, [], 0, caller, ToolProcessor(), LoopGuard(is_small=False),
    )
    return result, caller


def test_per_step_tool_budget_allows_one_completion_only_turn():
    result, caller = _run_budget_boundary()

    assert caller.completion_modes == [False, True]
    assert result.status == "done"
    assert result.interrupted is False
    assert result.action_tool_calls == 1


def test_global_tool_fuse_does_not_allow_an_extra_completion_turn():
    result, caller = _run_budget_boundary(max_total_calls=1)

    assert caller.completion_modes == [False]
    assert result.status == "continue"
    assert result.interrupted is True
    assert "global tool call limit" in result.summary


# ── the pseudo-only spin ─────────────────────────────────────────────


def test_pseudo_only_turns_are_bounded():
    guard = LoopGuard(is_small=False)
    guard.reset()
    ctx, messages = _ctx(), []

    for _ in range(_MAX_PSEUDO_ONLY_ROUNDS):
        assert guard.handle_pseudo_only(ctx, messages) is None

    forced = guard.handle_pseudo_only(ctx, messages)
    assert forced is not None, (
        "nothing else bounds this: the inner while advances on "
        "action_tool_calls, which a pseudo-only turn never moves"
    )
    assert forced.status == "continue"


def test_a_real_tool_call_clears_the_pseudo_only_streak():
    guard = LoopGuard(is_small=False)
    guard.reset()
    ctx, messages = _ctx(), []
    for _ in range(_MAX_PSEUDO_ONLY_ROUNDS):
        guard.handle_pseudo_only(ctx, messages)

    guard.pseudo_only_rounds = 0  # what the engine does on a regular call

    assert guard.handle_pseudo_only(ctx, messages) is None


def test_error_circuit_requires_a_diagnosed_alternative_before_blocking():
    guard = LoopGuard(is_small=False)
    ctx, messages = _ctx(), []
    guard.consecutive_tool_errors = 4

    guard.check_error_circuit_breaker(ctx, messages)

    assert len(messages) == 1
    guidance = messages[0]["content"]
    assert "working directory" in guidance
    assert "known local correction must be tried" in guidance
    assert "only when the changed approach also cannot proceed" in guidance


def test_identical_failed_tool_call_is_nudged_after_second_attempt():
    guard = LoopGuard(is_small=False)
    ctx, messages = _ctx(), []

    guard.on_tool_result("add_step", "{}", had_error=True)
    assert guard.check_repetition(ctx, messages) is None
    assert messages == []

    guard.on_tool_result("add_step", "{}", had_error=True)
    assert guard.check_repetition(ctx, messages) is None
    assert "exact same 'add_step' call" in messages[-1]["content"]


def test_identical_successful_tool_call_keeps_normal_threshold():
    guard = LoopGuard(is_small=False)
    ctx, messages = _ctx(), []

    for _ in range(2):
        guard.on_tool_result("read_file", '{"file_path":"a.py"}', had_error=False)
        assert guard.check_repetition(ctx, messages) is None

    assert messages == []


def test_bounded_rework_demotes_exploration_to_direct_continuation():
    ctx = _ctx()
    ctx.allow_explore = False
    result = StepResult(summary="Need to investigate import resolution", status="explore")

    changed = _apply_exploration_policy(ctx, result)

    assert changed is True
    assert result.status == "continue"
    assert "direct repository tools" in result.summary


def test_first_step_done_with_tools_and_summary_is_substantive():
    ctx = _ctx()
    result = StepResult(summary="Implemented helper and ran focused tests", status="done")
    result.action_tool_calls = 4

    assert _has_substantive_done_evidence(ctx, result) is True


def test_first_step_done_without_actions_still_needs_confirmation():
    ctx = _ctx()
    result = StepResult(summary="Looks done", status="done")

    assert _has_substantive_done_evidence(ctx, result) is False


def test_feature_task_cannot_close_before_any_edit():
    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")
    result = StepResult(summary="Inspected all relevant files", status="done")
    result.action_tool_calls = 9

    assert _has_substantive_done_evidence(ctx, result) is False

    ctx.state.task_has_edits = True
    assert _has_substantive_done_evidence(ctx, result) is True


def test_edit_requirement_applies_beyond_the_first_step():
    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")

    assert _task_requires_edits(ctx) is True
    assert ctx.state.task_has_edits is False


def test_edit_requirement_preserves_the_active_step_before_plan_advance():
    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")
    result = StepResult(
        summary="Inspected the implementation",
        status="done",
        final_answer="Already complete",
    )

    assert _enforce_edit_requirement(ctx, result) is True
    assert result.status == "continue"
    assert result.final_answer is None
    assert result.interrupted is True
    assert _should_advance_plan(result) is False


def test_verified_noop_step_can_close_an_already_satisfied_feature_task():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")
    ctx.state.plan.steps = [PlanStep(
        index=1,
        title="Run pytest to confirm the regression cases pass",
        status="active",
        verify={
            "kind": "command",
            "spec": "python -m pytest -q",
            "observable": "exit code 0",
        },
    )]
    active = ctx.state.plan.active_step
    assert active is not None
    ctx.state.objectively_verified_step_indices.add(active.index)

    result = StepResult(summary="All regression tests pass", status="done")

    assert _enforce_edit_requirement(ctx, result) is False


def test_passing_check_does_not_exempt_an_unedited_change_step():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")
    ctx.state.plan.steps = [PlanStep(
        index=1,
        title="Implement the parser fix",
        status="active",
        verify={
            "kind": "command",
            "spec": "python -m pytest -q",
            "observable": "exit code 0",
        },
    )]
    active = ctx.state.plan.active_step
    assert active is not None
    ctx.state.objectively_verified_step_indices.add(active.index)

    result = StepResult(summary="Existing tests pass", status="done")

    assert _enforce_edit_requirement(ctx, result) is True


def test_implementation_step_cannot_close_without_an_edit():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=1, title="Implement the parser fix", status="active"),
    ]
    result = StepResult(summary="Read the parser and planned tests", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.interrupted is True
    assert _should_advance_plan(result) is False


def test_verified_model_step_accepts_effect_from_an_earlier_task_edit():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.task_has_edits = True
    ctx.state.plan.steps = [
        PlanStep(index=4, title="Implement event dispatch", status="active"),
    ]
    ctx.state.objectively_verified_step_indices.add(4)
    result = StepResult(summary="Focused dispatch checks pass", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is False
    assert _should_advance_plan(result) is True
    assert "earlier task edit" in result.summary


def test_prior_edit_without_step_verification_cannot_skip_implementation():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.task_has_edits = True
    ctx.state.plan.steps = [
        PlanStep(index=4, title="Implement event dispatch", status="active"),
    ]
    result = StepResult(summary="The earlier edit may cover this", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.interrupted is True


def test_prior_edit_and_current_green_test_can_close_model_step():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.task_has_edits = True
    ctx.state.plan.steps = [
        PlanStep(index=4, title="Implement event dispatch", status="active"),
    ]
    result = StepResult(summary="The focused event tests pass", status="continue")
    result.behavior_tracker = SimpleNamespace(
        files_edited=set(),
        successful_test_commands=["python -m pytest tests/test_eventbus.py -q"],
    )

    assert _enforce_step_effect(ctx, result) is False
    assert _should_advance_plan(result) is True
    assert "successful test command" in result.summary


def test_green_test_without_any_task_edit_cannot_skip_implementation():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=4, title="Implement event dispatch", status="active"),
    ]
    result = StepResult(summary="The baseline suite passes", status="continue")
    result.behavior_tracker = SimpleNamespace(
        files_edited=set(),
        successful_test_commands=["python -m pytest -q"],
    )

    assert _enforce_step_effect(ctx, result) is True
    assert result.interrupted is True


def test_tool_runner_records_only_successful_recognized_step_tests():
    import json

    from infinidev.engine.loop.behavior_tracker import BehaviorTracker
    from infinidev.engine.loop.tool_runner import ToolRunner

    ctx = _ctx()
    tracker = BehaviorTracker(set())
    pytest_args = json.dumps({"command": "python -m pytest tests/test_eventbus.py -q"})

    ToolRunner._record_successful_step_test(
        ctx,
        pytest_args,
        json.dumps({"exit_code": 0, "success": True, "stdout": "9 passed"}),
        tracker,
    )
    ToolRunner._record_successful_step_test(
        ctx,
        json.dumps({"command": "python -m compileall ."}),
        json.dumps({"exit_code": 0, "success": True}),
        tracker,
    )
    ToolRunner._record_successful_step_test(
        ctx,
        pytest_args,
        json.dumps({"exit_code": 1, "success": False, "stdout": "1 failed"}),
        tracker,
    )

    assert tracker.successful_test_commands == [
        "python -m pytest tests/test_eventbus.py -q",
    ]


def test_test_capture_replaces_a_failed_exit_with_a_later_green_exit():
    import json

    from infinidev.engine.loop.tool_runner import ToolRunner

    ctx = _ctx()
    arguments = json.dumps({"command": "python -m pytest -q"})

    ToolRunner.capture_test_output(
        ctx,
        arguments,
        json.dumps({"exit_code": 1, "success": False, "stdout": "1 failed"}),
    )
    assert ctx.state.last_test_exit_code == 1

    ToolRunner.capture_test_output(
        ctx,
        arguments,
        json.dumps({"exit_code": 0, "success": True, "stdout": "1 passed"}),
    )
    assert ctx.state.last_test_exit_code == 0


def test_failed_latest_test_blocks_done_until_a_green_run():
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    ctx = _ctx()
    ctx.state.last_test_command = "python tests/runtests.py backend.tests"
    ctx.state.last_test_exit_code = 1
    call = _tool_call(
        "step_complete",
        '{"summary":"looks complete","status":"done"}',
        call_id="complete",
    )
    messages = [
        {"role": "tool", "tool_call_id": "complete", "content": "ok"},
    ]

    assert engine._step_gate._latest_test_failed(ctx, call, messages) is True
    assert "latest recognised test command" in messages[-1]["content"]

    ctx.state.last_test_exit_code = 0
    assert engine._step_gate._latest_test_failed(ctx, call, messages) is False


def test_failed_latest_test_still_allows_an_honest_blocked_outcome():
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    ctx = _ctx()
    ctx.state.last_test_command = "python -m pytest"
    ctx.state.last_test_exit_code = 2
    call = _tool_call(
        "step_complete",
        '{"summary":"missing compiler","status":"blocked"}',
        call_id="complete",
    )

    assert engine._step_gate._latest_test_failed(ctx, call, []) is False


def test_verified_user_approved_step_still_requires_its_own_edit():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.task_has_edits = True
    ctx.state.plan.steps = [
        PlanStep(
            index=4,
            title="Implement event dispatch",
            status="active",
            user_approved=True,
        ),
    ]
    ctx.state.objectively_verified_step_indices.add(4)
    result = StepResult(summary="Focused dispatch checks pass", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.interrupted is True


def test_model_change_container_advances_to_concrete_change_frontier():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=1, title="Implement cache support", status="active"),
        PlanStep(index=2, title="Run cache tests"),
        PlanStep(index=3, title="Add cache state to LoopState"),
    ]
    result = StepResult(summary="Decomposed the implementation", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is False
    assert result.decomposed_phase == "change"
    assert result.interrupted is False
    assert _should_advance_plan(result) is True


def test_model_change_container_recognizes_a_target_named_child():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(
            index=1,
            title="Fix normalize_tags case-insensitive behavior",
            status="active",
        ),
        PlanStep(index=2, title="tags.py: normalize_tags case-insensitively"),
    ]
    result = StepResult(summary="Split out the concrete file edit", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is False
    assert result.decomposed_phase == "change"
    assert _should_advance_plan(result) is True


def test_user_approved_change_cannot_be_superseded_without_an_edit():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(
            index=1,
            title="Implement cache support",
            status="active",
            user_approved=True,
        ),
        PlanStep(index=2, title="Add cache state to LoopState"),
    ]
    result = StepResult(summary="Decomposed the implementation", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.decomposed_phase == ""
    assert result.interrupted is True


def test_user_scope_absorbs_a_concrete_child_for_the_same_target():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(
            index=1,
            title="Fix normalize_tags case-insensitive behavior",
            status="active",
            user_approved=True,
        ),
        PlanStep(
            index=2,
            title="tags.py: normalize_tags case-insensitively",
            explanation="Use casefold keys and retain the original value.",
            expected_output="Focused normalization tests pass.",
        ),
    ]
    result = StepResult(summary="Decomposed the requested change", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.interrupted is True
    assert ctx.state.plan.active_step.title.startswith("tags.py:")
    assert ctx.state.plan.active_step.expected_output == "Focused normalization tests pass."
    assert ctx.state.plan.steps[1].status == "skipped"


def test_user_scope_does_not_absorb_an_unrelated_model_change():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(
            index=1,
            title="Fix normalize_tags case-insensitive behavior",
            status="active",
            user_approved=True,
        ),
        PlanStep(index=2, title="Update database migration rollback"),
    ]
    result = StepResult(summary="Added unrelated work", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert ctx.state.plan.active_step.title.startswith("Fix normalize_tags")
    assert ctx.state.plan.steps[1].status == "pending"


def test_production_change_cannot_delegate_to_a_test_change():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=1, title="Implement cache support", status="active"),
        PlanStep(index=2, title="Add focused regression tests"),
    ]
    result = StepResult(summary="Tests will cover it", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.decomposed_phase == ""
    assert result.interrupted is True


def test_consecutive_change_decomposition_is_bounded():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.consecutive_decompositions = 1
    ctx.state.plan.steps = [
        PlanStep(index=2, title="Implement cache support", status="active"),
        PlanStep(index=3, title="Wire cache into the runner"),
    ]
    result = StepResult(summary="Split it again", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is True
    assert result.decomposed_phase == ""


def test_implementation_step_closes_with_current_edit_evidence():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=1, title="Fix deserialize_db_from_string", status="active"),
    ]
    result = StepResult(summary="Implemented the fix", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited={"creation.py"})

    assert _enforce_step_effect(ctx, result) is False
    assert _should_advance_plan(result) is True


def test_implementation_step_remembers_edit_from_budget_continuation():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=3, title="Implement the parser fix", status="active"),
    ]
    ctx.state.history = [
        ActionRecord(
            step_index=3,
            summary="Budget boundary after edit",
            changes_made="Modified: parser.py",
        ),
    ]
    result = StepResult(summary="Focused test passed", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is False
    assert _should_advance_plan(result) is True


def test_implementation_step_uses_persisted_edit_evidence_after_interruption():
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=3, title="Implement the parser fix", status="active"),
    ]
    ctx.state.edited_step_indices.add(3)
    result = StepResult(summary="Focused test passed", status="continue")
    result.behavior_tracker = SimpleNamespace(files_edited=set())

    assert _enforce_step_effect(ctx, result) is False
    assert _should_advance_plan(result) is True


def test_finalize_inner_loop_persists_current_step_edit_evidence():
    from infinidev.engine.loop.behavior_tracker import BehaviorTracker
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.loop.plan_step import PlanStep

    ctx = _ctx()
    ctx.state.plan.steps = [
        PlanStep(index=4, title="Update tags.py", status="active"),
    ]
    tracker = BehaviorTracker(set())
    tracker.task_has_edits = True
    tracker.files_edited.add("src/tags.py")

    LoopEngine()._finalize_inner_loop(
        ctx,
        StepResult(summary="budget", status="continue", interrupted=True),
        action_tool_calls=10,
        tracker=tracker,
        saw_tool_calls=True,
    )

    assert ctx.state.edited_step_indices == {4}


def test_budget_interruption_resumes_the_same_plan_step():
    interrupted = StepResult(
        summary="tool budget reached",
        status="continue",
        interrupted=True,
    )
    completed = StepResult(
        summary="step complete",
        status="continue",
    )

    assert _should_advance_plan(interrupted) is False
    assert _should_advance_plan(completed) is True


def test_pending_step_reconciliation_precedes_whole_task_edit_gate():
    from infinidev.engine.loop.plan_step import PlanStep
    from infinidev.engine.loop.step_manager import StepManager

    ctx = _ctx()
    ctx.task = SimpleNamespace(kind="feature")
    ctx.state.plan.steps = [
        PlanStep(index=1, title="Inspect src/tags.py", status="active"),
        PlanStep(index=2, title="Edit normalize_tags in src/tags.py"),
    ]
    result, edit_blocked = _reconcile_step_result(
        ctx,
        StepResult(summary="Inspection complete", status="done"),
        StepManager(SimpleNamespace(_hooks=None)),
    )

    assert edit_blocked is False
    assert result.status == "continue"
    assert result.interrupted is False
    assert _should_advance_plan(result) is True


# ── the critic's veto is bounded too ─────────────────────────────────


class _Verdict:
    def __init__(self, action: str) -> None:
        self.action = action
        self.message = "you did not run the tests"
        self.is_silent = False


class _Critic:
    model_short_name = "critic"

    def review(self, *a, **k):
        return _Verdict("reject")


def test_critic_veto_is_demoted_after_the_cap(monkeypatch):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "ASSISTANT_LLM_INCLUDE_STEP_COMPLETE", True, raising=False)

    liaison = CriticLiaison()
    liaison.reset_run()
    monkeypatch.setattr(liaison, "get", lambda ctx: _Critic())

    ctx = _ctx()
    call = SimpleNamespace(id="sc1")
    overwrites: list[str] = []

    def _overwrite(messages, call_id, body):
        overwrites.append(body)

    blocked = [
        liaison.review_step_complete(ctx, [], call, None, _overwrite).blocked
        for _ in range(_MAX_REJECTS_PER_STEP + 1)
    ]

    assert blocked[:_MAX_REJECTS_PER_STEP] == [True] * _MAX_REJECTS_PER_STEP
    assert blocked[-1] is False, (
        "an unbounded veto holds the step open forever at two LLM calls a "
        "turn, and the pseudo-only turn it produces spends no budget"
    )


def test_critic_objection_still_reaches_the_model_after_demotion(monkeypatch):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "ASSISTANT_LLM_INCLUDE_STEP_COMPLETE", True, raising=False)

    liaison = CriticLiaison()
    liaison.reset_run()
    monkeypatch.setattr(liaison, "get", lambda ctx: _Critic())

    ctx = _ctx()
    call = SimpleNamespace(id="sc1")
    review = None
    for _ in range(_MAX_REJECTS_PER_STEP + 1):
        review = liaison.review_step_complete(ctx, [], call, None, lambda *a: None)

    assert review.blocked is False
    assert review.followup is not None, "demoted to advice, not to silence"
    assert "tests" in review.followup["content"]
