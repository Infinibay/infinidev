"""Tests for planner-derived verification versus user acceptance criteria.

The planner emits falsifiable checks, but those checks are not user-authored
requirements. They remain derived verification criteria throughout the
Task and reviewer handoff.

Covers:
  - is_falsifiable filter
  - task_from_free_text with real criteria (is_synthesised flips False)
  - _build_plan_from_args parses + filters acceptance_criteria
  - emit_plan schema advertises acceptance_criteria
  - run_review_rework_loop enriches the reviewer's task_description
"""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.orchestration.task_schema import (
    is_falsifiable, task_from_free_text, is_synthesised,
)
from infinidev.engine.analysis.planner import _build_plan_from_args
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis import review_engine as re_mod
from infinidev.engine.analysis.review_engine import run_review_rework_loop


# ── is_falsifiable ───────────────────────────────────────────────────────

class TestIsFalsifiable:
    def test_concrete_is_falsifiable(self):
        assert is_falsifiable("expired JWTs are rejected by validate_token") is True

    def test_vague_quality_words_rejected(self):
        assert is_falsifiable("the code looks good") is False
        assert is_falsifiable("the API is clean") is False

    def test_too_short_rejected(self):
        assert is_falsifiable("ok") is False
        assert is_falsifiable("") is False


# ── task_from_free_text with real criteria ───────────────────────────────

class TestTaskFromFreeText:
    def test_placeholder_when_no_criteria(self):
        t = task_from_free_text("Fix the JWT validation bug in the auth module please")
        assert is_synthesised(t) is True

    def test_user_criteria_flip_synthesised_false(self):
        t = task_from_free_text(
            "Fix the JWT validation bug in the auth module please",
            acceptance_criteria=["expired tokens are rejected by validate_token"],
        )
        assert is_synthesised(t) is False
        assert t.acceptance_criteria == ["expired tokens are rejected by validate_token"]

    def test_planner_criteria_remain_derived(self):
        t = task_from_free_text(
            "Fix the JWT validation bug in the auth module please",
            derived_verification_criteria=[
                "expired tokens are rejected by validate_token"
            ],
        )
        assert is_synthesised(t) is True
        assert t.derived_verification_criteria == [
            "expired tokens are rejected by validate_token"
        ]

    def test_empty_criteria_falls_back_to_placeholder(self):
        t = task_from_free_text(
            "Fix the JWT validation bug in the auth module please",
            acceptance_criteria=[],
        )
        assert is_synthesised(t) is True


# ── planner parses + filters criteria ────────────────────────────────────

class TestBuildPlanCriteria:
    def _args(self, criteria):
        return {
            "overview": "Fix the exp check and prove it with a test, end to end.",
            "steps": [{"title": "Patch validate_token", "detail": "reject expired"}],
            "acceptance_criteria": criteria,
        }

    def test_falsifiable_criteria_kept(self):
        plan = _build_plan_from_args(self._args([
            "expired tokens are rejected by validate_token",
            "tests/test_auth.py::test_expired passes",
        ]))
        assert plan is not None
        assert len(plan.acceptance_criteria) == 2

    def test_non_falsifiable_dropped(self):
        plan = _build_plan_from_args(self._args([
            "the auth code looks good",                       # dropped
            "expired tokens are rejected by validate_token",  # kept
        ]))
        assert plan.acceptance_criteria == ["expired tokens are rejected by validate_token"]

    def test_missing_criteria_yields_empty(self):
        plan = _build_plan_from_args({
            "overview": "Do the thing thoroughly and carefully now.",
            "steps": [{"title": "step one"}],
        })
        assert plan.acceptance_criteria == []


# ── emit_plan schema advertises the field ────────────────────────────────

class TestEmitPlanSchema:
    def test_acceptance_criteria_in_schema(self):
        from infinidev.engine.schema_sanitizer import tool_to_openai_schema
        from infinidev.tools.planner.emit_plan_tool import EmitPlanTool
        import json
        blob = json.dumps(tool_to_openai_schema(EmitPlanTool()))
        assert "acceptance_criteria" in blob


# ── run_review_rework_loop enriches the reviewer's task_description ───────

class _FakeAgent:
    def activate_context(self, **kw): pass
    def deactivate(self): pass


class _CapturingReviewer:
    """Approves immediately, but records the task_description it was judged with."""
    def __init__(self):
        self.seen_description = None
    def reset(self): pass
    def _should_multi_pass(self, *a, **k): return False
    @property
    def can_review_again(self): return True
    def review(self, *, task_description, **kw):
        self.seen_description = task_description
        return ReviewResult(verdict="APPROVED", summary="ok")


class _FakeEngine:
    def __init__(self, workspace):
        self._workspace = workspace
    def get_objective_checks(self): return []
    def get_plan_steps(self): return []
    def get_file_contents(self): return {"a.py": "x = 1"}
    def get_file_tracker(self): return None
    def get_changed_files_summary(self): return "diff --git a/a.py b/a.py"
    def get_file_change_reasons(self): return {}
    def execute(self, **kw): return "done"


class TestReviewEnrichment:
    def _run(self, monkeypatch, reviewer, criteria, derived=None):
        monkeypatch.setattr(re_mod, "collect_automated_checks", lambda **kw: {"verification_passed": True})
        import tempfile
        with tempfile.TemporaryDirectory() as ws:
            return run_review_rework_loop(
                engine=_FakeEngine(ws), agent=_FakeAgent(), session_id="s1",
                task_prompt=("Fix the bug", "expected"), initial_result="r",
                reviewer=reviewer, on_status=None, acceptance_criteria=criteria,
                derived_verification_criteria=derived,
            )

    def test_criteria_injected_into_review(self, monkeypatch):
        rv = _CapturingReviewer()
        self._run(monkeypatch, rv, ["expired tokens are rejected by validate_token"])
        assert "Acceptance criteria" in rv.seen_description
        assert "expired tokens are rejected" in rv.seen_description

    def test_no_criteria_leaves_description_plain(self, monkeypatch):
        rv = _CapturingReviewer()
        self._run(monkeypatch, rv, None)
        assert rv.seen_description == "Fix the bug"

    def test_derived_criteria_are_labeled_non_authoritative(self, monkeypatch):
        rv = _CapturingReviewer()
        self._run(
            monkeypatch,
            rv,
            None,
            ["expired tokens are rejected by validate_token"],
        )
        assert "Derived verification criteria" in rv.seen_description
        assert "NOT user-authored requirements" in rv.seen_description
        assert "expired tokens are rejected" in rv.seen_description
