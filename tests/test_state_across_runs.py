"""State that has to survive — or must not survive — a second execute().

A user turn can enter execute() more than once: the review's rework loop
re-enters the same engine up to three times. Two pieces of state got the
scope wrong in opposite directions.
"""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.models import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.step_result import StepResult


# ── the file tracker across a rework pass ────────────────────────────


def test_merge_keeps_files_the_rework_pass_did_not_touch():
    first = FileChangeTracker()
    first.record("/w/src/foo.py", None, "def foo(): pass")
    first.record("/w/tests/test_foo.py", None, "def test_foo(): pass")

    second = FileChangeTracker()
    second.record("/w/src/foo.py", "def foo(): pass", "def foo(): return 1")
    second.merge_from(first)

    changed = set(second.get_all_paths())
    assert any("test_foo.py" in p for p in changed), (
        "the reviewer stopped seeing files the rework pass happened not to "
        "touch, and a rework that wrote nothing skipped review entirely"
    )
    assert any("foo.py" in p for p in changed)


def test_merge_keeps_a_created_file_reported_as_created():
    first = FileChangeTracker()
    first.record("/w/src/foo.py", None, "v1")

    second = FileChangeTracker()
    second.record("/w/src/foo.py", "v1", "v2")
    second.merge_from(first)

    assert second.get_action("/w/src/foo.py") == "created", (
        "the diff was being computed against the file's post-first-pass "
        "state, so a file created this turn read as merely modified"
    )


def test_merge_keeps_the_newest_content():
    first = FileChangeTracker()
    first.record("/w/src/foo.py", None, "v1")
    second = FileChangeTracker()
    second.record("/w/src/foo.py", "v1", "v2")
    second.merge_from(first)
    assert "v2" in second.get_diff("/w/src/foo.py")


def test_merge_sums_change_counts_and_keeps_reasons_in_order():
    first = FileChangeTracker()
    first.record("/w/a.py", None, "1")
    first.record_reason("/w/a.py", "created it")

    second = FileChangeTracker()
    second.record("/w/a.py", "1", "2")
    second.record_reason("/w/a.py", "fixed the review comment")
    second.merge_from(first)

    assert second.get_change_count("/w/a.py") == 2
    assert second.get_reasons("/w/a.py") == ["created it", "fixed the review comment"]


# ── explore must not close the step it asked to break down ───────────


def _state_with_plan() -> LoopState:
    state = LoopState()
    state.plan = LoopPlan(steps=[
        PlanStep(index=1, title="one", status="done", user_approved=True),
        PlanStep(index=2, title="two", status="active", user_approved=True),
        PlanStep(index=3, title="three", status="pending", user_approved=True),
    ])
    return state


def test_explore_leaves_the_step_active():
    """Nothing reactivates a step, and apply_operations will not re-add an
    approved one — so marking it done loses it for good."""
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    state = _state_with_plan()
    ctx = SimpleNamespace(
        state=state, project_id=1, agent_id="a1", verbose=False,
        agent_name="dev", event_id=None,
    )
    step_result = StepResult(summary="this needs breaking down", status="explore")

    step_mgr = SimpleNamespace(
        auto_split=lambda c, r: r,
        advance_plan=lambda c, r: (_ for _ in ()).throw(
            AssertionError("advance_plan must not run for an explore step")
        ),
        summarize_and_record=lambda *a, **k: None,
    )
    engine._run_post_step(ctx, step_result, step_mgr, [], 0, 0)

    assert state.plan.steps[1].status == "active"
    assert state.plan.steps[2].status == "pending"


def test_continue_does_not_advance_the_plan():
    from infinidev.engine.loop.engine import LoopEngine

    engine = LoopEngine()
    state = _state_with_plan()
    ctx = SimpleNamespace(
        state=state, project_id=1, agent_id="a1", verbose=False,
        agent_name="dev", event_id=None,
    )
    advanced: list[bool] = []
    step_mgr = SimpleNamespace(
        auto_split=lambda c, r: r,
        advance_plan=lambda c, r: advanced.append(True),
        summarize_and_record=lambda *a, **k: None,
    )
    engine._run_post_step(
        ctx, StepResult(summary="done", status="continue"), step_mgr, [], 0, 0,
    )
    assert advanced == []
