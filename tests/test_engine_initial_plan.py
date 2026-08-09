"""Tests for LoopEngine's initial_plan parameter.

Commit 5 of the pipeline redesign. The developer's LoopState starts
pre-seeded with the analyst's plan: overview renders every iteration,
steps preserve their authority provenance, and the first step is active
so execution has a starting point. Planner-inferred steps are deliberately
not projected as user-approved.

The full loop is not exercised here (it makes LLM calls). We verify
the seeding helper and that the seeded state produces the right
iteration prompt — that's the entire contract of the feature.
"""

from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from types import SimpleNamespace

from infinidev.engine.loop.engine import (
    _is_planning_mode,
    _seed_initial_plan_if_fresh,
    _seed_state_from_plan,
)
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.context import build_iteration_prompt


def _sample_plan() -> Plan:
    return Plan(
        overview="Fix the JWT validation bug in src/auth.py and add a regression test.",
        steps=[
            PlanStepSpec(
                title="Read auth.py and find validate_token",
                detail="Open src/auth.py, locate validate_token, note exp-claim handling.",
                expected_output="File read, function located, hypothesis formed.",
            ),
            PlanStepSpec(
                title="Patch the exp check",
                detail="Update validate_token to reject tokens past their exp timestamp.",
                expected_output="Code edited, unit test passes.",
            ),
            PlanStepSpec(
                title="Run the auth test suite",
                detail="Execute pytest tests/test_auth.py -q; verify green.",
                expected_output="All auth tests pass.",
            ),
        ],
    )


class TestSeedStateFromPlan:
    def test_overview_copied(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        assert "JWT validation" in state.plan.overview

    def test_steps_count_matches(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        assert len(state.plan.steps) == 3

    def test_planner_steps_are_model_inferred(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        for step in state.plan.steps:
            assert step.authority == "model_inferred"
            assert step.user_approved is False

    def test_explicit_user_step_receives_scope_protection(self):
        state = LoopState()
        plan = Plan(
            overview="User explicitly requested this exact operation.",
            steps=[
                PlanStepSpec(
                    title="Update the requested file",
                    authority="user_explicit",
                )
            ],
        )

        _seed_state_from_plan(state, plan)

        assert state.plan.steps[0].authority == "user_explicit"
        assert state.plan.steps[0].user_approved is True

    def test_first_step_active_rest_pending(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        assert state.plan.steps[0].status == "active"
        for step in state.plan.steps[1:]:
            assert step.status == "pending"

    def test_step_detail_and_expected_copied(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        first = state.plan.steps[0]
        assert first.title == "Read auth.py and find validate_token"
        assert "validate_token" in first.detail
        assert "File read" in first.expected_output

    def test_step_indices_are_1_based_and_ordered(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        assert [s.index for s in state.plan.steps] == [1, 2, 3]

    def test_empty_plan_yields_empty_state(self):
        state = LoopState()
        _seed_state_from_plan(state, Plan(overview="", steps=[]))
        assert state.plan.overview == ""
        assert state.plan.steps == []

    def test_resumed_state_is_not_replaced_by_the_task_bootstrap_plan(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        state.plan.steps[0].status = "done"

        _seed_initial_plan_if_fresh(
            SimpleNamespace(state=state, resumed=True),
            Plan(overview="new bootstrap", steps=[], rolling_horizon_limit=3),
        )

        assert state.plan.overview.startswith("Fix the JWT")
        assert len(state.plan.steps) == 3
        assert state.plan.steps[0].status == "done"


class TestPlanningMode:
    def test_switches_to_execution_as_soon_as_a_step_is_added(self):
        state = LoopState()
        ctx = SimpleNamespace(state=state, skip_plan=False)

        assert _is_planning_mode(ctx) is True
        _seed_state_from_plan(state, _sample_plan())
        assert _is_planning_mode(ctx) is False

    def test_reenters_planning_when_the_horizon_has_no_active_step(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        for step in state.plan.steps:
            step.status = "done"
        ctx = SimpleNamespace(state=state, skip_plan=False)

        assert _is_planning_mode(ctx) is True

    def test_plan_free_adapter_never_enters_planning(self):
        ctx = SimpleNamespace(state=LoopState(), skip_plan=True)
        assert _is_planning_mode(ctx) is False


class TestSeededStateRendersCorrectly:
    def test_plan_overview_in_prompt(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        prompt = build_iteration_prompt("task", "expected", state)
        assert "<plan-overview>" in prompt
        assert "JWT validation" in prompt

    def test_bootstrap_branch_is_suppressed(self):
        """When initial_plan populates steps, the 'No plan yet' branch
        should NOT appear. The bootstrap prompt is only emitted when
        state.plan.steps is empty."""
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        prompt = build_iteration_prompt("task", "expected", state)
        assert "No plan yet" not in prompt
        assert "Your FIRST action must be to call add_step" not in prompt

    def test_active_step_detail_renders(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        prompt = build_iteration_prompt("task", "expected", state)
        assert "validate_token" in prompt  # detail of the active step

    def test_pending_step_detail_stays_hidden(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        prompt = build_iteration_prompt("task", "expected", state)
        # step 2's detail mentions "exp timestamp" — should not render yet
        assert "exp timestamp" not in prompt

    def test_plan_block_lists_all_step_titles(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        prompt = build_iteration_prompt("task", "expected", state)
        assert "Read auth.py and find validate_token" in prompt
        assert "Patch the exp check" in prompt
        assert "Run the auth test suite" in prompt
