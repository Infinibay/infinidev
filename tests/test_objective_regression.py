"""Tests for Phase 6: resume-aware objective regression detection.

The task-end re-verify reads this session's prior verdicts from the
ledger; an objective that PASSED before but FAILs now is flagged as a
REGRESSION (a stronger signal than a fresh failure), surfaced via the
objectives_regressed status and counted in the summary.
"""

from __future__ import annotations

from infinidev.config.settings import settings
from infinidev.db.service import record_objective_verdict, get_objective_verdicts
from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis import review_engine as re_mod
from infinidev.engine.analysis.review_engine import run_review_rework_loop

class _FakeAgent:
    def activate_context(self, **kw): pass
    def deactivate(self): pass


class _FakeReviewer:
    def reset(self): pass
    def _should_multi_pass(self, *a, **k): return False
    @property
    def can_review_again(self): return True
    def review(self, **kw): return ReviewResult(verdict="APPROVED", summary="ok")


class _FakeEngine:
    def __init__(self, workspace, checks):
        self._workspace = workspace
        self._checks = checks
    def get_objective_checks(self): return self._checks
    def get_plan_steps(self): return []
    def get_file_contents(self): return {}
    def get_file_tracker(self): return None
    def get_changed_files_summary(self): return ""
    def get_file_change_reasons(self): return {}
    def execute(self, **kw): return "done"


def _run(session_id, checks, tmp_path, monkeypatch):
    monkeypatch.setattr(re_mod, "collect_automated_checks", lambda **kw: {"verification_passed": True})
    monkeypatch.setattr(settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 0)  # no rework
    statuses = []
    run_review_rework_loop(
        engine=_FakeEngine(str(tmp_path), checks), agent=_FakeAgent(),
        session_id=session_id, task_prompt=("do it", "exp"), initial_result="r",
        reviewer=_FakeReviewer(), on_status=lambda l, m: statuses.append((l, m)),
    )
    return statuses


class TestRegressionDetection:
    def test_pass_then_fail_is_regression(self, temp_db, tmp_path, monkeypatch):
        # Prior run recorded this objective (spec 'false') as PASS.
        record_objective_verdict(session_id="s", step_index=1, title="build",
                                 kind="command", spec="false", verdict="PASS")
        # Now the same objective FAILs.
        statuses = _run("s", [(1, "build", StepVerification(kind="command", spec="false"))],
                        tmp_path, monkeypatch)
        assert any(l == "objectives_regressed" for l, m in statuses)
        assert any(l == "objectives_summary" and "1 regressed" in m for l, m in statuses)
        # Recorded with the regression marker in detail.
        rows = get_objective_verdicts("s")
        assert any("REGRESSION" in (r["detail"] or "") for r in rows)

    def test_fresh_fail_not_regression(self, temp_db, tmp_path, monkeypatch):
        # No prior verdict for this spec → a plain failure, not a regression.
        statuses = _run("s2", [(1, "build", StepVerification(kind="command", spec="false"))],
                        tmp_path, monkeypatch)
        assert not any(l == "objectives_regressed" for l, m in statuses)
        assert any(l == "objectives_summary" and "regressed" not in m for l, m in statuses)

    def test_pass_then_pass_not_regression(self, temp_db, tmp_path, monkeypatch):
        record_objective_verdict(session_id="s3", step_index=1, title="build",
                                 kind="command", spec="true", verdict="PASS")
        statuses = _run("s3", [(1, "build", StepVerification(kind="command", spec="true"))],
                        tmp_path, monkeypatch)
        assert not any(l == "objectives_regressed" for l, m in statuses)

    def test_prior_fail_then_fail_not_regression(self, temp_db, tmp_path, monkeypatch):
        # Only PASS→FAIL counts; FAIL→FAIL is not a regression.
        record_objective_verdict(session_id="s4", step_index=1, title="build",
                                 kind="command", spec="false", verdict="FAIL")
        statuses = _run("s4", [(1, "build", StepVerification(kind="command", spec="false"))],
                        tmp_path, monkeypatch)
        assert not any(l == "objectives_regressed" for l, m in statuses)
