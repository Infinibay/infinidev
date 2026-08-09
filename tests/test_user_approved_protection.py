"""What ``user_approved`` protects, and what it deliberately stops protecting.

The flag used to gate *operations*: an approved step could not be modified,
removed, or written over. That refused three different acts with equal force —
"reword a title the run just proved misleading", "refine a success criterion
written before anyone had read the code", and "drop the feature the user asked
for" — and only the third one is the model's to be refused.

It now gates *fields* instead. An approved step can be refined through
:data:`APPROVED_MUTABLE_FIELDS`; it cannot be removed, and it cannot be
displaced by an add. What stops the model from simply declaring the run
finished while approved steps sit pending is not this file — it is the scope
gate, covered in ``test_scope_gate.py``.

The fields worth defending are unreachable here by construction:
``StepOperation`` carries no ``detail``, no ``verify`` and no ``status``, so
the planner's researched guidance and its adversarial check cannot be written
through any of these paths. The allowlist is enforced anyway, so that widening
``StepOperation`` later cannot silently widen the freeze.
"""

import json

import pytest

from infinidev.engine.loop import loop_plan as loop_plan_module
from infinidev.engine.loop.loop_plan import APPROVED_MUTABLE_FIELDS, LoopPlan
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_operation import StepOperation
from infinidev.tools.base.context import set_loop_state
from infinidev.tools.meta.plan_tools import AddStepTool, ModifyStepTool, RemoveStepTool


def _approved_plan() -> LoopPlan:
    plan = LoopPlan(overview="User approved this plan in chat.")
    plan.steps = [
        PlanStep(index=1, title="Read auth.py", user_approved=True, status="active"),
        PlanStep(index=2, title="Fix validate_token", user_approved=True),
        PlanStep(index=3, title="Run tests", user_approved=True),
    ]
    return plan


class TestScopeIsProtected:
    def test_remove_approved_step_is_rejected(self):
        plan = _approved_plan()
        plan.apply_operations([StepOperation(op="remove", index=2)])
        assert plan.steps[1].status != "skipped"
        assert plan.steps[1].title == "Fix validate_token"

    def test_add_cannot_displace_an_approved_step(self):
        """An add at an occupied index shifts the occupant, never replaces it."""
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="add", index=2, title="Read the JWT library first"),
        ])
        titles = [s.title for s in plan.steps]
        assert titles == [
            "Read auth.py",
            "Read the JWT library first",
            "Fix validate_token",
            "Run tests",
        ]
        # The approved steps survived with their flag intact — only renumbered.
        assert [s.index for s in plan.steps if s.user_approved] == [1, 3, 4]

    def test_add_cannot_displace_the_active_step(self):
        """The active index keys the gate's once-per-step guarantees."""
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="add", index=1, title="Sneak ahead of the running step"),
        ])
        assert [s.title for s in plan.steps] == [
            "Read auth.py", "Fix validate_token", "Run tests",
        ]

    def test_add_cannot_displace_a_finished_step(self):
        """A done step's index is a foreign key into the archived records."""
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Explored", status="done"),
            PlanStep(index=2, title="Editing", status="active"),
        ]
        plan.apply_operations([
            StepOperation(op="add", index=1, title="Rewrite history"),
        ])
        assert [s.title for s in plan.steps] == ["Explored", "Editing"]


class TestWordingIsNotProtected:
    def test_approved_step_can_be_reworded(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(
                op="modify",
                index=2,
                title="auth.py: add the exp claim check to validate_token()",
            ),
        ])
        assert plan.steps[1].title == (
            "auth.py: add the exp claim check to validate_token()"
        )

    def test_approved_step_success_criterion_can_be_refined(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(
                op="modify",
                index=3,
                expected_output="pytest tests/test_auth.py::test_expired passes",
            ),
        ])
        assert plan.steps[2].expected_output == (
            "pytest tests/test_auth.py::test_expired passes"
        )
        assert plan.steps[2].title == "Run tests", "only the named field changes"

    def test_only_allowlisted_fields_reach_an_approved_step(self, monkeypatch):
        """The allowlist is the mechanism, not a description of today's schema.

        Today every field ``StepOperation`` carries is refinable, so nothing is
        dropped. Narrowing the constant proves a field left out of it is
        refused rather than inherited — which is what protects a ``detail`` or
        ``status`` added to the operation later.
        """
        monkeypatch.setattr(
            loop_plan_module, "APPROVED_MUTABLE_FIELDS", ("explanation",)
        )
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(
                op="modify", index=2,
                title="should not land", explanation="should land",
            ),
        ])
        assert plan.steps[1].title == "Fix validate_token"
        assert plan.steps[1].explanation == "should land"

    def test_a_finished_step_cannot_be_rewritten_through_operations(self):
        plan = _approved_plan()
        plan.steps[1].status = "done"
        plan.apply_operations([
            StepOperation(op="modify", index=2, title="rewrite history"),
        ])
        assert plan.steps[1].title == "Fix validate_token"

    def test_unapproved_steps_take_every_field(self):
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

    def test_a_mixed_batch_applies_to_both(self):
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Approved", user_approved=True, status="active"),
            PlanStep(index=2, title="LLM-added"),
        ]
        plan.apply_operations([
            StepOperation(op="modify", index=1, title="Approved, reworded"),
            StepOperation(op="modify", index=2, title="renamed"),
        ])
        assert plan.steps[0].title == "Approved, reworded"
        assert plan.steps[1].title == "renamed"


class TestInsertion:
    def test_a_prerequisite_lands_before_the_work_that_needs_it(self):
        """The case the old destructive add made impossible."""
        plan = LoopPlan()
        plan.steps = [
            PlanStep(index=1, title="Reading", status="active"),
            PlanStep(index=2, title="Edit api.py to call verify()"),
        ]
        plan.apply_operations([
            StepOperation(op="add", index=2, title="Add verify() to auth.py"),
        ])
        assert [s.title for s in plan.steps] == [
            "Reading", "Add verify() to auth.py", "Edit api.py to call verify()",
        ]
        assert [s.index for s in plan.steps] == [1, 2, 3]

    def test_appending_past_the_end_still_works(self):
        plan = _approved_plan()
        plan.apply_operations([
            StepOperation(op="add", index=4, title="Extra verification"),
        ])
        assert [s.title for s in plan.steps][-1] == "Extra verification"
        assert len(plan.steps) == 4


class TestUndischarged:
    def test_it_reports_what_the_run_has_not_dealt_with(self):
        plan = _approved_plan()
        plan.steps[0].status = "done"
        plan.steps[1].status = "active"
        assert [s.index for s in plan.undischarged()] == [2, 3]

    def test_the_closing_step_is_excluded(self):
        """At gate time the step being closed is still ``active``."""
        plan = _approved_plan()
        plan.steps[0].status = "done"
        plan.steps[1].status = "done"
        plan.steps[2].status = "active"
        assert plan.undischarged(exclude_index=3) == []

    def test_a_blocked_step_is_discharged(self):
        """Blocked was attempted and reported. That is not a silent drop."""
        plan = _approved_plan()
        plan.steps[0].status = "done"
        plan.steps[1].status = "blocked"
        plan.steps[2].status = "active"
        assert [s.index for s in plan.undischarged(exclude_index=3)] == []


class TestPublicPlanTools:
    @staticmethod
    def _bind(state: LoopState) -> None:
        set_loop_state("test-agent", state)

    def test_add_step_before_inserts_and_reports_the_shift(self, bound_tool):
        state = LoopState()
        state.plan.steps = [
            PlanStep(index=1, title="Reading", status="active"),
            PlanStep(index=2, title="Implement api.py"),
        ]
        self._bind(state)
        tool = bound_tool(AddStepTool)

        result = json.loads(tool._run(title="Add auth.py helper", before=2))

        assert result["status"] == "added"
        assert result["index"] == 2
        assert "moved down by one" in result["shifted"]
        assert [step.title for step in state.plan.steps] == [
            "Reading", "Add auth.py helper", "Implement api.py",
        ]

    def test_first_public_step_becomes_active(self, bound_tool):
        state = LoopState()
        self._bind(state)

        result = json.loads(bound_tool(AddStepTool)._run(
            title="Inspect src/widget.py",
        ))

        assert result["status"] == "added"
        assert state.plan.active_step is not None
        assert state.plan.active_step.title == "Inspect src/widget.py"

    def test_later_public_step_stays_pending(self, bound_tool):
        state = LoopState()
        state.plan.steps = [PlanStep(index=1, title="Current", status="active")]
        self._bind(state)

        bound_tool(AddStepTool)._run(title="Inspect src/later.py")

        assert state.plan.steps[1].status == "pending"

    def test_modify_step_reports_fields_applied_to_approved_scope(self, bound_tool):
        state = LoopState()
        state.plan = _approved_plan()
        self._bind(state)
        tool = bound_tool(ModifyStepTool)

        result = json.loads(tool._run(
            index=2,
            title="auth.py: fix validate_token()",
            expected_output="the expired-token test passes",
        ))

        assert result["applied"] == ["title", "expected_output"]
        assert state.plan.steps[1].title == "auth.py: fix validate_token()"
        assert state.plan.steps[1].expected_output == "the expired-token test passes"

    def test_remove_step_still_refuses_approved_scope(self, bound_tool):
        state = LoopState()
        state.plan = _approved_plan()
        self._bind(state)
        tool = bound_tool(RemoveStepTool)

        result = json.loads(tool._run(index=2))

        assert "cannot be removed" in result["error"]
        assert state.plan.steps[1].status == "pending"

    def test_rolling_horizon_refuses_a_fourth_open_step(self, bound_tool):
        state = LoopState()
        state.plan = LoopPlan(rolling_horizon_limit=3)
        state.plan.steps = [
            PlanStep(index=1, title="Read auth.py", status="active"),
            PlanStep(index=2, title="Edit auth.py"),
            PlanStep(index=3, title="Test auth.py"),
        ]
        self._bind(state)
        tool = bound_tool(AddStepTool)

        result = json.loads(tool._run(title="Document auth.py"))

        assert "Rolling horizon" in result["error"]
        assert len(state.plan.steps) == 3

    def test_add_step_reuses_a_near_duplicate_open_step(self, bound_tool):
        state = LoopState()
        state.plan.steps = [
            PlanStep(
                index=1,
                title="Implement click.utils.normalize_prog_name in src/click/utils.py",
                status="active",
            ),
        ]
        self._bind(state)

        result = json.loads(bound_tool(AddStepTool)._run(
            title="Implement normalize_prog_name function in src/click/utils.py",
        ))

        assert result["status"] == "duplicate"
        assert result["existing_index"] == 1
        assert len(state.plan.steps) == 1

    def test_add_step_keeps_distinct_phases_for_the_same_target(self, bound_tool):
        state = LoopState()
        state.plan.steps = [
            PlanStep(
                index=1,
                title="Implement normalize_prog_name in src/click/utils.py",
                status="active",
            ),
        ]
        self._bind(state)

        result = json.loads(bound_tool(AddStepTool)._run(
            title="Test normalize_prog_name in tests/test_utils.py",
        ))

        assert result["status"] == "added"
        assert len(state.plan.steps) == 2


def test_step_result_operations_cannot_add_duplicate_open_work():
    plan = LoopPlan()
    plan.steps = [
        PlanStep(index=1, title="Inspect src/auth.py", status="active"),
    ]

    plan.apply_operations([
        StepOperation(op="add", index=2, title="Inspect the src/auth.py file"),
    ])

    assert len(plan.steps) == 1


def test_step_result_operations_respect_the_rolling_horizon():
    plan = LoopPlan(rolling_horizon_limit=1)
    plan.steps = [
        PlanStep(index=1, title="Implement src/auth.py", status="active"),
    ]

    plan.apply_operations([
        StepOperation(op="add", index=2, title="Test tests/test_auth.py"),
    ])

    assert len(plan.steps) == 1


class TestOverviewImmutability:
    def test_apply_operations_does_not_touch_overview(self):
        plan = _approved_plan()
        original = plan.overview
        plan.apply_operations([
            StepOperation(op="add", index=99, title="Extra step"),
        ])
        assert plan.overview == original


def test_the_allowlist_covers_every_field_the_operation_can_carry():
    """A guard on the guard: a new StepOperation field must be a decision.

    If this fails, someone added a field to ``StepOperation``. Decide whether
    an approved step should accept it and add it to the constant, or leave it
    out so the freeze holds — but decide, rather than inheriting whichever
    happened by default.
    """
    writable = set(StepOperation.model_fields) - {"op", "index"}
    assert writable == set(APPROVED_MUTABLE_FIELDS), (
        "StepOperation gained or lost a field. Review "
        "loop_plan.APPROVED_MUTABLE_FIELDS before updating this test."
    )
