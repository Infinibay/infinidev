"""Tests for LoopEngine's initial_plan parameter.

Commit 5 of the pipeline redesign. The developer's LoopState starts
pre-seeded with the analyst's plan: overview renders every iteration
as <plan-overview>, steps are marked user_approved so the LLM can't
remove them, and the first step is active so execution has a
starting point.

The full loop is not exercised here (it makes LLM calls). We verify
the seeding helper and that the seeded state produces the right
iteration prompt — that's the entire contract of the feature.
"""

from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.loop.engine import _seed_state_from_plan
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

    def test_all_steps_user_approved(self):
        state = LoopState()
        _seed_state_from_plan(state, _sample_plan())
        for step in state.plan.steps:
            assert step.user_approved is True

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
