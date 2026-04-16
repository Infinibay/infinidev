"""Tests for <plan-overview> block and per-step detail rendering.

These tests lock in the Commit 1 invariants from the pipeline redesign:
  * <plan-overview> is emitted whenever LoopPlan.overview is non-empty,
    regardless of whether any steps exist.
  * The overview appears BEFORE <plan> in the iteration prompt — agents
    should read the big picture before the step list.
  * PlanStep.detail is rendered only inside <current-action> for the
    active step (pending/done steps show only their title) to keep the
    iteration prompt compact.
"""

from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.context import build_iteration_prompt


def _state_with_steps(*steps: PlanStep, overview: str = "") -> LoopState:
    state = LoopState()
    state.plan.overview = overview
    state.plan.steps = list(steps)
    return state


class TestPlanOverviewBlock:
    def test_empty_overview_no_block(self):
        state = LoopState()
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<plan-overview>" not in prompt

    def test_overview_block_appears(self):
        state = _state_with_steps(overview="Fix the JWT validation bug in auth.")
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<plan-overview>" in prompt
        assert "Fix the JWT validation bug in auth." in prompt
        assert "</plan-overview>" in prompt

    def test_overview_appears_before_plan(self):
        state = _state_with_steps(
            PlanStep(index=1, title="Read auth.py", status="active"),
            overview="Big picture narrative here.",
        )
        prompt = build_iteration_prompt("task", "expected", state)
        ov_idx = prompt.index("<plan-overview>")
        plan_idx = prompt.index("<plan>")
        assert ov_idx < plan_idx

    def test_overview_without_steps_still_renders(self):
        state = _state_with_steps(overview="Overview only — no steps yet.")
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<plan-overview>" in prompt

    def test_skip_plan_hides_overview(self):
        state = _state_with_steps(overview="Would normally show.")
        prompt = build_iteration_prompt("task", "expected", state, skip_plan=True)
        assert "<plan-overview>" not in prompt


class TestPerStepDetailRendering:
    def test_detail_absent_when_empty(self):
        state = _state_with_steps(
            PlanStep(index=1, title="Read auth.py", status="active"),
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<current-action>" in prompt
        assert "Step 1: Read auth.py" in prompt

    def test_detail_renders_for_active_step(self):
        active_detail = "Open src/auth.py, find validate_token, check JWT claims."
        state = _state_with_steps(
            PlanStep(
                index=1,
                title="Read auth.py",
                status="active",
                detail=active_detail,
            ),
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert active_detail in prompt

    def test_detail_not_rendered_for_pending_steps(self):
        pending_detail = "PENDING STEP DETAIL — should not appear in prompt."
        state = _state_with_steps(
            PlanStep(index=1, title="Active step", status="active"),
            PlanStep(
                index=2,
                title="Pending step",
                status="pending",
                detail=pending_detail,
            ),
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert pending_detail not in prompt

    def test_detail_not_rendered_for_done_steps(self):
        done_detail = "DONE STEP DETAIL — should not appear in prompt."
        state = _state_with_steps(
            PlanStep(
                index=1,
                title="Done step",
                status="done",
                detail=done_detail,
            ),
            PlanStep(index=2, title="Active step", status="active"),
        )
        prompt = build_iteration_prompt("task", "expected", state)
        assert done_detail not in prompt

    def test_detail_works_with_small_model(self):
        active_detail = "Small-model detail body."
        state = _state_with_steps(
            PlanStep(
                index=1,
                title="Active",
                status="active",
                detail=active_detail,
            ),
        )
        prompt = build_iteration_prompt(
            "task", "expected", state, small_model=True
        )
        assert active_detail in prompt
