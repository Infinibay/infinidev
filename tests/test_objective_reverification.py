"""Tests for Phase 2: post-loop objective re-verification.

At task end the review-rework loop re-runs every planner-authored step
verification together (the cross-objective regression backstop the
per-step gate cannot see), feeding failures back to the developer.

Covers:
  - LoopEngine.get_objective_checks() / revived get_plan_steps()
  - _collect_objective_checks / _format_objective_failures helpers
  - run_review_rework_loop objective re-verification: all-pass is a no-op,
    a regressed check drives a bounded developer re-execution
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.plan import Plan, PlanStepSpec
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis import review_engine as re_mod
from infinidev.engine.analysis.review_engine import (
    run_review_rework_loop,
    _collect_objective_checks,
    _format_objective_failures,
)
from infinidev.engine.loop.engine import LoopEngine, _seed_state_from_plan
from infinidev.engine.loop.models import LoopState


def _seed_engine_with(verifies) -> LoopEngine:
    """A LoopEngine whose _last_state has steps carrying the given verifies."""
    eng = LoopEngine()
    state = LoopState()
    plan = Plan(
        overview="x" * 10,
        steps=[
            PlanStepSpec(title=f"step {i}", detail="d", verify=v)
            for i, v in enumerate(verifies, start=1)
        ],
    )
    _seed_state_from_plan(state, plan)
    eng._last_state = state
    return eng


# ── engine accessors ─────────────────────────────────────────────────────

class TestEngineAccessors:
    def test_get_objective_checks_filters_executable(self):
        eng = _seed_engine_with([
            StepVerification(kind="command", spec="true"),
            None,
            StepVerification(kind="none"),
            StepVerification(kind="test_id", spec="t.py::t"),
        ])
        checks = eng.get_objective_checks()
        assert [c[0] for c in checks] == [1, 4]  # only executable steps
        assert all(isinstance(c[2], StepVerification) for c in checks)

    def test_get_objective_checks_empty_without_state(self):
        assert LoopEngine().get_objective_checks() == []

    def test_get_plan_steps_revived(self):
        eng = _seed_engine_with([StepVerification(kind="command", spec="true")])
        steps = eng.get_plan_steps()
        assert len(steps) == 1
        assert steps[0]["step"] == 1 and steps[0]["title"] == "step 1"

    def test_get_plan_steps_empty_without_state(self):
        assert LoopEngine().get_plan_steps() == []


# ── helpers ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_collect_handles_missing_getter(self):
        assert _collect_objective_checks(SimpleNamespace()) == []

    def test_collect_handles_raising_getter(self):
        def boom():
            raise RuntimeError("nope")
        assert _collect_objective_checks(SimpleNamespace(get_objective_checks=boom)) == []

    def test_format_objective_failures(self):
        check = StepVerification(kind="command", spec="false", observable="OK")
        vres = SimpleNamespace(format_for_developer=lambda: "exit 1 output", summary="failed")
        out = _format_objective_failures([(2, "patch auth", check, vres)])
        assert "Step 2: patch auth" in out
        assert "command): false" in out
        assert "must show: OK" in out
        assert "exit 1 output" in out


# ── integration: run_review_rework_loop objective re-verification ────────

class _FakeAgent:
    def activate_context(self, **kw): pass
    def deactivate(self): pass


class _FakeReviewer:
    """Approves immediately so the LLM review loop exits at once, letting the
    test focus on the objective-reverification stage that runs before it."""
    def reset(self): pass
    def _should_multi_pass(self, *a, **k): return False
    @property
    def can_review_again(self): return True
    def review(self, **kw):
        return ReviewResult(verdict="APPROVED", summary="ok")


class _FakeEngine:
    """Minimal engine surface used by run_review_rework_loop."""
    def __init__(self, workspace, checks, execute_results=None):
        self._workspace = workspace
        self._checks = checks
        self.execute_calls = []
        self._execute_results = list(execute_results or [])
    # objective checks captured up front
    def get_objective_checks(self): return self._checks
    def get_plan_steps(self): return []
    def get_file_contents(self): return {}
    def get_file_tracker(self): return None
    def get_changed_files_summary(self): return ""
    def get_file_change_reasons(self): return {}
    def execute(self, *, agent, task_prompt, verbose=True):
        self.execute_calls.append(task_prompt)
        # The act of "fixing" is simulated by the test mutating the workspace
        # between rounds; here we just return the next canned result.
        return self._execute_results.pop(0) if self._execute_results else "fixed"


def _run(engine, monkeypatch):
    # Neutralise the whole-suite VerificationEngine + automated checks so the
    # test isolates the objective-reverification stage.
    monkeypatch.setattr(
        re_mod, "collect_automated_checks",
        lambda **kw: {"verification_passed": True},
    )
    return run_review_rework_loop(
        engine=engine,
        agent=_FakeAgent(),
        session_id="s1",
        task_prompt=("do the thing", "expected"),
        initial_result="initial",
        reviewer=_FakeReviewer(),
        on_status=None,
    )


class TestReworkLoopObjectiveReverify:
    def test_all_pass_no_reexecution(self, tmp_path, monkeypatch):
        eng = _FakeEngine(str(tmp_path), checks=[(1, "s1", StepVerification(kind="command", spec="true"))])
        result, review = _run(eng, monkeypatch)
        assert eng.execute_calls == []          # nothing to fix
        assert review.is_approved

    def test_regression_triggers_reexecution(self, tmp_path, monkeypatch):
        # A marker file the check requires; absent at first (fail), the
        # simulated developer "fixes" it by creating it before re-verify.
        marker = tmp_path / "MARKER"
        check = StepVerification(kind="file_contains", spec="MARKER", observable="ready")

        class _FixingEngine(_FakeEngine):
            def execute(self, *, agent, task_prompt, verbose=True):
                self.execute_calls.append(task_prompt)
                marker.write_text("ready")   # the fix
                return "fixed"

        eng = _FixingEngine(str(tmp_path), checks=[(1, "s1", check)])
        result, review = _run(eng, monkeypatch)
        assert len(eng.execute_calls) == 1                     # fixed in one round
        assert "objective verification FAILED" in eng.execute_calls[0][0]
        assert review.is_approved

    def test_persistent_failure_bounded_by_max_rounds(self, tmp_path, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 2)
        check = StepVerification(kind="command", spec="false")  # never passes
        eng = _FakeEngine(str(tmp_path), checks=[(1, "s1", check)])
        result, review = _run(eng, monkeypatch)
        # Re-executes exactly max_rounds times, then gives up (does not hang).
        assert len(eng.execute_calls) == 2

    def test_disabled_skips_reverify(self, tmp_path, monkeypatch):
        from infinidev.config import settings as settings_mod
        monkeypatch.setattr(settings_mod.settings, "REVIEW_OBJECTIVE_REVERIFY_ENABLED", False)
        check = StepVerification(kind="command", spec="false")
        eng = _FakeEngine(str(tmp_path), checks=[(1, "s1", check)])
        result, review = _run(eng, monkeypatch)
        assert eng.execute_calls == []   # feature off → no re-execution
