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
from infinidev.engine.loop.behavior_tracker import BehaviorTracker


def _state(*steps: PlanStep) -> LoopState:
    state = LoopState()
    state.plan.steps = list(steps)
    return state


def _ctx(state: LoopState) -> SimpleNamespace:
    return SimpleNamespace(
        state=state,
        project_id=None,
        agent_id="test-agent",
        agent=SimpleNamespace(workspace_path="."),
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

    def test_verified_edits_fold_covered_model_substeps(self, tmp_path) -> None:
        dispatcher = tmp_path / "dispatcher.py"
        registry = tmp_path / "registry.py"
        dispatcher.write_text(
            "class EventBus:\n    def emit(self, snapshot, priority, once): pass\n",
            encoding="utf-8",
        )
        registry.write_text(
            "class Registry:\n    def subscribe(self, priority, once): pass\n",
            encoding="utf-8",
        )
        state = _state(
            PlanStep(index=1, title="Implement the event bus", status="active"),
            PlanStep(index=2, title="Extend Registry subscribe with priority and once"),
            PlanStep(index=3, title="Implement EventBus emit snapshot semantics"),
        )
        context = _ctx(state)
        context.agent.workspace_path = str(tmp_path)
        tracker = BehaviorTracker(set())
        tracker.files_edited.update({"dispatcher.py", "registry.py"})
        tracker.successful_test_commands.append("pytest -q")
        result = _done()
        result.behavior_tracker = tracker

        reconciled = StepManager.reconcile_task_completion(context, result)

        assert reconciled is result
        assert [step.status for step in state.plan.steps] == [
            "active", "done", "done",
        ]
        assert "verified by pytest -q" in state.plan.steps[1].conclusion

    def test_verified_edits_do_not_fold_unrelated_model_substep(self, tmp_path) -> None:
        changed = tmp_path / "registry.py"
        changed.write_text("def subscribe(priority, once): pass\n", encoding="utf-8")
        state = _state(
            PlanStep(index=1, title="Implement subscriptions", status="active"),
            PlanStep(index=2, title="Rewrite authentication token refresh"),
        )
        context = _ctx(state)
        context.agent.workspace_path = str(tmp_path)
        tracker = BehaviorTracker(set())
        tracker.files_edited.add("registry.py")
        tracker.successful_test_commands.append("pytest -q")
        result = _done()
        result.behavior_tracker = tracker

        reconciled = StepManager.reconcile_task_completion(context, result)

        assert reconciled.status == "continue"
        assert state.plan.steps[1].status == "pending"


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
    def test_continue_without_follow_up_keeps_the_same_step_active(self) -> None:
        from infinidev.engine.loop.engine import _reconcile_step_result

        state = _state(
            PlanStep(index=1, title="Implement and verify", status="active"),
        )
        context = _ctx(state)
        manager = StepManager(SimpleNamespace(_hooks=None))

        result, _ = _reconcile_step_result(
            context,
            StepResult(summary="Implementation done; tests remain", status="continue"),
            manager,
        )

        assert result.interrupted is True
        assert "No follow-up Step was scheduled" in result.summary
        assert state.plan.active_step is state.plan.steps[0]

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

    def test_decomposed_change_drains_changes_before_verification(self) -> None:
        state = _state(
            PlanStep(index=1, title="Implement cache", status="active"),
            PlanStep(index=2, title="Run cache tests"),
            PlanStep(index=3, title="Add cache state"),
            PlanStep(index=4, title="Update tool runner"),
        )
        manager = StepManager(SimpleNamespace(_hooks=None))

        manager.advance_plan(
            _ctx(state),
            StepResult(
                summary="Split into concrete edits",
                status="continue",
                decomposed_phase="change",
            ),
        )

        assert state.plan.steps[0].status == "skipped"
        assert state.plan.steps[1].status == "pending"
        assert state.plan.steps[2].status == "active"
        assert state.plan.execution_phase == "change"
        assert state.plan.consecutive_decompositions == 1

        manager.advance_plan(
            _ctx(state), StepResult(summary="Added state", status="continue"),
        )
        assert state.plan.steps[3].status == "active"
        assert state.plan.steps[1].status == "pending"

        manager.advance_plan(
            _ctx(state), StepResult(summary="Updated runner", status="continue"),
        )
        assert state.plan.steps[1].status == "active"
        assert state.plan.execution_phase == ""

    def test_successful_concrete_edit_resets_decomposition_budget(self) -> None:
        state = _state(
            PlanStep(index=2, title="Add cache state", status="active"),
            PlanStep(index=3, title="Run cache tests"),
        )
        state.plan.consecutive_decompositions = 1
        state.edited_step_indices.add(2)
        manager = StepManager(SimpleNamespace(_hooks=None))

        manager.advance_plan(
            _ctx(state), StepResult(summary="Added cache state", status="continue"),
        )

        assert state.plan.consecutive_decompositions == 0
        assert state.plan.steps[1].status == "active"

    def test_net_change_retires_older_discovery_for_the_edited_file(
        self, monkeypatch,
    ) -> None:
        state = _state(
            PlanStep(
                index=2,
                title="Implement all/any handling in assertion/rewrite.py",
                status="active",
            ),
            PlanStep(index=3, title="Explore assertion/rewrite.py helpers"),
            PlanStep(index=4, title="Run assertion rewrite tests"),
        )
        state.step_entry_change_fingerprints[2] = ()
        context = _ctx(state)
        context.file_tracker = SimpleNamespace(
            change_fingerprint=lambda **kwargs: (
                ("/workspace/assertion/rewrite.py", "new-digest"),
            )
        )
        monkeypatch.setattr(
            "infinidev.engine.loop.step_manager._emit_log", lambda *a, **k: None,
        )

        StepManager(SimpleNamespace(_hooks=None)).advance_plan(
            context,
            StepResult(summary="Implemented the change", status="continue"),
        )

        assert state.plan.steps[0].status == "done"
        assert state.plan.steps[1].status == "skipped"
        assert state.plan.steps[2].status == "active"
        assert 2 not in state.step_entry_change_fingerprints
