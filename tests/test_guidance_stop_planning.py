"""Unit tests for the ``stop_planning_start_coding`` detector.

Covers the rate-based heuristic introduced after session 2's
empirical feedback: "está bien agregar steps en la etapa de
planning, el tema es no agregar y agregar y agregar en el mismo
step". The detector should NEVER fire in the initial planning
iteration, and should only fire when the count of ``add_step``
tool calls WITHIN a single step's message slice crosses the
threshold AND the task has produced no edits yet.
"""

from __future__ import annotations

import pytest

from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.guidance.detectors import (
    _has_stop_planning_start_coding,
    _STOP_PLANNING_ADD_STEP_THRESHOLD,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def make_state(
    *,
    iteration: int = 2,
    has_edits: bool = False,
    plan_steps: int = 3,
) -> LoopState:
    """Build a LoopState in a specific execution shape.

    Defaults to iter=2 (past planning), no edits yet, 3 plan steps —
    the most common execute-mode shape. Tests override specific
    fields to cover boundary conditions.
    """
    s = LoopState()
    s.iteration_count = iteration
    s.task_has_edits = has_edits
    s.plan = LoopPlan(steps=[
        PlanStep(index=i, title=f"step {i}", status="pending")
        for i in range(plan_steps)
    ])
    return s


def add_step_messages(n: int) -> list[dict]:
    """Build a message list containing *n* ``add_step`` tool calls in
    a single assistant turn — the canonical "plan-bombing" pattern."""
    return [{
        "role": "assistant",
        "tool_calls": [
            {
                "function": {
                    "name": "add_step",
                    "arguments": f'{{"title":"step {i}"}}',
                }
            }
            for i in range(n)
        ],
    }]


def mixed_tool_messages(add_steps: int, other_tools: int) -> list[dict]:
    """Simulate a step with some add_step AND some real work tool calls.

    Used to verify that the detector counts add_step specifically, not
    total tool activity — a step with 3 add_steps + 10 read_files is
    not procrastination, it's exploration.
    """
    calls = []
    for i in range(add_steps):
        calls.append({
            "function": {
                "name": "add_step",
                "arguments": f'{{"title":"s{i}"}}',
            }
        })
    for i in range(other_tools):
        calls.append({
            "function": {
                "name": "read_file",
                "arguments": f'{{"file_path":"f{i}.py"}}',
            }
        })
    return [{"role": "assistant", "tool_calls": calls}]


# ─────────────────────────────────────────────────────────────────────────────
# Cases where the detector MUST fire
# ─────────────────────────────────────────────────────────────────────────────


class TestFires:
    """Scenarios where the detector should signal procrastination."""

    def test_threshold_add_steps_in_one_turn(self):
        """N add_step calls in a single assistant message → fires."""
        state = make_state(iteration=2, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD)
        assert _has_stop_planning_start_coding(messages, state) is True

    def test_above_threshold_in_one_turn(self):
        """10 add_steps in one turn is the pattern the user described."""
        state = make_state(iteration=2, has_edits=False)
        messages = add_step_messages(10)
        assert _has_stop_planning_start_coding(messages, state) is True

    def test_spread_across_multiple_turns_same_step(self):
        """Add_steps spread across multiple assistant messages in the
        same step slice still count. The slice is pre-cut by the
        dispatcher to a single step, so all assistant messages in it
        belong to one outer-loop iteration."""
        state = make_state(iteration=3, has_edits=False)
        messages = []
        for batch in range(3):
            messages.extend(add_step_messages(3))  # 9 total
        assert _has_stop_planning_start_coding(messages, state) is True

    def test_fires_in_later_iterations(self):
        """Later iterations (not just iter 2) still fire when the
        pattern appears."""
        state = make_state(iteration=10, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD)
        assert _has_stop_planning_start_coding(messages, state) is True


# ─────────────────────────────────────────────────────────────────────────────
# Cases where the detector MUST NOT fire
# ─────────────────────────────────────────────────────────────────────────────


class TestDoesNotFire:
    """Scenarios that would be false positives if the detector triggered."""

    def test_planning_iteration_even_with_many_add_steps(self):
        """THE critical case: in iter 1, the model is building the
        initial plan and will legitimately emit many add_steps. This
        is the user's core concern — "castigar porque lo haga en
        planning no estaría bien"."""
        state = make_state(iteration=1, has_edits=False)
        messages = add_step_messages(20)  # way above threshold
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_iteration_zero_is_also_planning(self):
        """iter 0 is technically the very first inner loop before
        any step_complete — treat as planning phase."""
        state = make_state(iteration=0, has_edits=False)
        messages = add_step_messages(20)
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_below_threshold_in_execute_mode(self):
        """3 add_steps in an execute iteration is legitimate replanning
        — e.g. the model discovered two new files it needs to read
        and added two steps for them."""
        state = make_state(iteration=3, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD - 1)
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_task_has_edits_immunises(self):
        """Once any edit lands, the procrastination pattern is
        broken by definition. The model is actively writing code
        AND planning — planning more is fine because there IS
        progress to plan around."""
        state = make_state(iteration=5, has_edits=True)
        messages = add_step_messages(20)  # way above threshold
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_mixed_tool_usage_does_not_fire(self):
        """A step with 3 add_steps + 10 read_files is exploration,
        not procrastination — the model is reading files to
        understand the code before writing. Count add_step
        specifically, not total tool activity."""
        state = make_state(iteration=3, has_edits=False)
        messages = mixed_tool_messages(
            add_steps=_STOP_PLANNING_ADD_STEP_THRESHOLD - 1,
            other_tools=10,
        )
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_empty_messages(self):
        """Empty message slice can't trigger the detector."""
        state = make_state(iteration=5, has_edits=False)
        assert _has_stop_planning_start_coding([], state) is False

    def test_no_add_steps_many_other_tools(self):
        """A busy step with 20 read_file calls and zero add_steps is
        genuine work, not planning-procrastination."""
        state = make_state(iteration=3, has_edits=False)
        messages = mixed_tool_messages(add_steps=0, other_tools=20)
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_state_none(self):
        """Defensive: state=None never crashes and never fires."""
        messages = add_step_messages(100)
        assert _has_stop_planning_start_coding(messages, None) is False


# ─────────────────────────────────────────────────────────────────────────────
# Boundary conditions around the threshold
# ─────────────────────────────────────────────────────────────────────────────


class TestBoundary:
    """Exact-threshold behaviour — catching off-by-one mistakes."""

    def test_exactly_threshold(self):
        """Exactly _STOP_PLANNING_ADD_STEP_THRESHOLD add_steps → fires.
        The comparison uses >= so threshold is inclusive."""
        state = make_state(iteration=2, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD)
        assert _has_stop_planning_start_coding(messages, state) is True

    def test_one_below_threshold(self):
        """threshold - 1 → does not fire."""
        state = make_state(iteration=2, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD - 1)
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_iteration_boundary_one(self):
        """iter=1 is planning, must not fire regardless of add_step
        count."""
        state = make_state(iteration=1, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD)
        assert _has_stop_planning_start_coding(messages, state) is False

    def test_iteration_boundary_two(self):
        """iter=2 is the first execute iteration — eligible to fire."""
        state = make_state(iteration=2, has_edits=False)
        messages = add_step_messages(_STOP_PLANNING_ADD_STEP_THRESHOLD)
        assert _has_stop_planning_start_coding(messages, state) is True
