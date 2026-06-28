"""Tests for Phase 5: durable persistence of objective verdicts.

The task-end re-verification now writes each objective's final verdict
(PASS / FAIL / UNVERIFIABLE) to the objective_verdicts table and emits a
one-line summary. Covers:
  - the table exists after init_db (schema.sql is the source of truth)
  - record_objective_verdict / get_objective_verdicts round-trip
  - run_review_rework_loop persists verdicts + emits objectives_summary
"""

from __future__ import annotations

import os
import tempfile

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import (
    init_db, execute_with_retry,
    record_objective_verdict, get_objective_verdicts,
)
from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis import review_engine as re_mod
from infinidev.engine.analysis.review_engine import run_review_rework_loop


@pytest.fixture
def temp_db():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    original = settings.DB_PATH
    settings.DB_PATH = path
    init_db()
    try:
        yield
    finally:
        settings.DB_PATH = original
        os.unlink(path)


# ── schema + round-trip ──────────────────────────────────────────────────

class TestLedgerSchema:
    def test_table_created(self, temp_db):
        tables = execute_with_retry(
            lambda c: c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        )
        assert "objective_verdicts" in [t[0] for t in tables]

    def test_record_and_get_roundtrip(self, temp_db):
        record_objective_verdict(
            session_id="sess-1", step_index=2, title="patch auth",
            kind="test_id", spec="tests/test_auth.py::test_expired",
            verdict="PASS", detail="1 passed",
        )
        rows = get_objective_verdicts("sess-1")
        assert len(rows) == 1
        assert rows[0]["verdict"] == "PASS"
        assert rows[0]["title"] == "patch auth"
        assert rows[0]["step_index"] == 2

    def test_scoped_by_session(self, temp_db):
        record_objective_verdict(session_id="a", step_index=1, title="x",
                                 kind="command", spec="true", verdict="PASS")
        record_objective_verdict(session_id="b", step_index=1, title="y",
                                 kind="command", spec="false", verdict="FAIL")
        assert len(get_objective_verdicts("a")) == 1
        assert get_objective_verdicts("b")[0]["verdict"] == "FAIL"

    def test_missing_session_empty(self, temp_db):
        assert get_objective_verdicts("nobody") == []


# ── integration: re-verify loop persists + summarises ────────────────────

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


class TestReworkLoopPersistence:
    def test_passing_objective_persisted_and_summarised(self, temp_db, tmp_path, monkeypatch):
        monkeypatch.setattr(re_mod, "collect_automated_checks", lambda **kw: {"verification_passed": True})
        statuses = []
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "build passes", StepVerification(kind="command", spec="true"))])
        run_review_rework_loop(
            engine=eng, agent=_FakeAgent(), session_id="sess-int",
            task_prompt=("do it", "expected"), initial_result="r",
            reviewer=_FakeReviewer(), on_status=lambda l, m: statuses.append((l, m)),
        )
        # Persisted to the ledger
        rows = get_objective_verdicts("sess-int")
        assert len(rows) == 1 and rows[0]["verdict"] == "PASS"
        # And summarised to the user
        assert any(l == "objectives_summary" and "1 passed" in m for l, m in statuses)

    def test_failing_objective_recorded_as_fail(self, temp_db, tmp_path, monkeypatch):
        monkeypatch.setattr(re_mod, "collect_automated_checks", lambda **kw: {"verification_passed": True})
        monkeypatch.setattr(settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 0)  # no rework
        eng = _FakeEngine(str(tmp_path),
                          checks=[(1, "build passes", StepVerification(kind="command", spec="false"))])
        run_review_rework_loop(
            engine=eng, agent=_FakeAgent(), session_id="sess-fail",
            task_prompt=("do it", "expected"), initial_result="r",
            reviewer=_FakeReviewer(), on_status=None,
        )
        rows = get_objective_verdicts("sess-fail")
        assert len(rows) == 1 and rows[0]["verdict"] == "FAIL"
