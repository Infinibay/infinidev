"""Plan-scope reconciliation at Step and Task boundaries.

``step_complete(status="done")`` is a claim about the whole Task, while
``status="continue"`` closes the active Step and advances the plan.  Models do
not always preserve that distinction.  The engine reconciles an early Task
close locally instead of spending additional model turns repeating an enum
value, and it must then advance the active Step exactly once.
"""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_manager import StepManager
from infinidev.engine.loop.step_result import StepResult


def _state(*steps: PlanStep) -> LoopState:
    state = LoopState()
    state.plan.steps = list(steps)
    return state


def _ctx(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        project_id=None,
        agent_id="test-agent",
    )


def _done(final_answer: str | None = "all set") -> StepResult:
    return StepResult(summary="finished current work", status="done", final_answer=final_answer)


class TestPrematureTaskCompletion:
    def test_done_with_pending_step_becomes_continue(self) -> None:
        state = _state(
            PlanStep(index=1, title="Inspect", status="active"),
            PlanStep(index=2, title="Implement"),
        )

        result = StepManager.reconcile_task_completion(_ctx(state), _done())

        assert result.status == "continue"
        assert result.final_answer is None
        assert "2. Implement" in result.summary

    def test_final_answer_does_not_skip_approved_scope(self) -> None:
        state = _state(
            PlanStep(index=1, title="Inspect", user_approved=True, status="active"),
            PlanStep(index=2, title="Implement", user_approved=True),
            PlanStep(index=3, title="Verify", user_approved=True),
        )

        result = StepManager.reconcile_task_completion(
            _ctx(state), _done("I finished everything."),
        )

        assert result.status == "continue"
        assert result.final_answer is None
        assert "2. Implement" in result.summary
        assert "3. Verify" in result.summary

    def test_developer_authored_horizon_is_also_preserved(self) -> None:
        state = _state(
            PlanStep(index=1, title="Explore", status="active"),
            PlanStep(index=2, title="Write the fix"),
        )

        result = StepManager.reconcile_task_completion(_ctx(state), _done())

        assert result.status == "continue"


class TestValidCompletion:
    def test_current_active_step_is_not_counted_as_future_work(self) -> None:
        state = _state(
            PlanStep(index=1, title="Inspect", status="done"),
            PlanStep(index=2, title="Implement", status="done"),
            PlanStep(index=3, title="Verify", status="active"),
        )
        original = _done()

        assert StepManager.reconcile_task_completion(_ctx(state), original) is original

    def test_blocked_or_skipped_steps_do_not_prevent_completion(self) -> None:
        state = _state(
            PlanStep(index=1, title="Inspect", status="done"),
            PlanStep(index=2, title="Unavailable", status="blocked"),
            PlanStep(index=3, title="Obsolete", status="skipped"),
            PlanStep(index=4, title="Report", status="active"),
        )
        original = _done()

        assert StepManager.reconcile_task_completion(_ctx(state), original) is original

    def test_empty_plan_and_non_done_statuses_are_unchanged(self) -> None:
        context = _ctx(_state())
        continuing = StepResult(summary="more", status="continue")
        blocked = StepResult(summary="cannot proceed", status="blocked")

        assert StepManager.reconcile_task_completion(context, continuing) is continuing
        assert StepManager.reconcile_task_completion(context, blocked) is blocked


class TestStepTransition:
    def test_continue_closes_active_step_and_activates_next(self) -> None:
        state = _state(
            PlanStep(index=1, title="Inspect", status="active"),
            PlanStep(index=2, title="Implement"),
        )
        manager = StepManager(SimpleNamespace(_hooks=None))

        manager.advance_plan(
            _ctx(state), StepResult(summary="inspection complete", status="continue"),
        )

        assert state.plan.steps[0].status == "done"
        assert state.plan.steps[1].status == "active"

    def test_blocked_is_recorded_and_the_next_step_is_visible(self) -> None:
        state = _state(
            PlanStep(index=1, title="Could not edit", status="active"),
            PlanStep(index=2, title="Report the limitation"),
        )
        manager = StepManager(SimpleNamespace(_hooks=None))

        manager.advance_plan(
            _ctx(state), StepResult(summary="Permission denied", status="blocked"),
        )

        assert state.plan.steps[0].status == "blocked"
        assert state.plan.steps[1].status == "active"

    def test_blocked_step_with_planned_recovery_does_not_end_the_task(self) -> None:
        from infinidev.engine.loop.engine import LoopEngine

        state = _state(
            PlanStep(index=1, title="Broken approach", status="active"),
            PlanStep(index=2, title="Try the diagnosed recovery"),
        )
        context = _ctx(state)
        manager = StepManager(SimpleNamespace(_hooks=None))
        result = StepResult(summary="Approach failed", status="blocked")
        manager.advance_plan(context, result)

        terminal = LoopEngine()._check_termination(
            context, result, manager, iteration=0, consecutive_all_done=0,
        )

        assert terminal is None
        assert state.plan.steps[0].status == "blocked"
        assert state.plan.steps[1].status == "active"
