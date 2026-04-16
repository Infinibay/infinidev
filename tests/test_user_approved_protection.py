"""Tests for user_approved step protection in LoopPlan.apply_operations.

Commit 1 of the pipeline redesign introduces `PlanStep.user_approved`. When
set True (by the orchestrator when injecting an analyst-emitted plan into
the developer's LoopState), the LLM can NOT remove, modify, or overwrite
that step mid-execution. The LLM can still add brand-new steps around the
approved ones.
"""

from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_operation import StepOperation


def _approved_plan() -> LoopPlan:
    plan = LoopPlan(overview="User approved this plan in chat.")
    plan.steps = [
        PlanStep(index=1, title="Read auth.py", user_approved=True, status="active"),
        PlanStep(index=2, title="Fix validate_token", user_approved=True),
        PlanStep(index=3, title="Run tests", user_approved=True),
    ]
    return plan


class TestApprovedStepProtection:
    def test_remove_approved_step_is_rejected(self):
        plan = _approved_plan()
        plan.apply_operations([StepOperation(op="remove", index=2)])
        assert plan.steps[1].status != "skipped"
        assert plan.steps[1].title == "Fix validate_token"

    def test_modify_approved_step_is_rejected(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="modify", index=2, title="Different title"),
        ])
        assert plan.steps[1].title == "Fix validate_token"

    def test_add_over_approved_index_is_rejected(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="add", index=2, title="Injected step"),
        ])
        assert plan.steps[1].title == "Fix validate_token"

    def test_add_new_step_at_fresh_index_works(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="add", index=4, title="Extra verification"),
        ])
        titles = [s.title for s in plan.steps]
        assert "Extra verification" in titles
        assert len(plan.steps) == 4


class TestUnapprovedStepsUnaffected:
    def test_llm_steps_remain_modifiable(self):
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Explore repo", status="active"),
            PlanStep(index=2, title="Make plan"),
        ]
        plan.apply_operations([
            StepOperation(op="modify", index=2, title="Write helper"),
        ])
        assert plan.steps[1].title == "Write helper"

    def test_llm_steps_remain_removable(self):
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Keep me", status="active"),
            PlanStep(index=2, title="Remove me"),
            PlanStep(index=3, title="Also keep"),
        ]
        plan.apply_operations([StepOperation(op="remove", index=2)])
        removed = next(s for s in plan.steps if s.index == 2)
        assert removed.status == "skipped"


class TestMixedApprovedAndFree:
    def test_approved_blocked_but_unapproved_applied(self):
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Approved", user_approved=True, status="active"),
            PlanStep(index=2, title="LLM-added"),
        ]
        plan.apply_operations([
            StepOperation(op="modify", index=1, title="should be blocked"),
            StepOperation(op="modify", index=2, title="renamed"),
        ])
        assert plan.steps[0].title == "Approved"
        assert plan.steps[1].title == "renamed"


class TestOverviewImmutability:
    def test_apply_operations_does_not_touch_overview(self):
        plan = _approved_plan()
        original = plan.overview
        plan.apply_operations([
            StepOperation(op="add", index=99, title="Extra step"),
        ])
        assert plan.overview == original
