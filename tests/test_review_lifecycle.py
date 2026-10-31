"""Review operations must stop with the developer's terminal outcome."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine.analysis.evidence_review import (
    EvidenceReviewResult,
    run_evidence_review_rework_loop,
)
from infinidev.engine.analysis.review_engine import run_review_rework_loop
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.verification_result import VerificationResult


class _Agent:
    active = False

    def activate_context(self, **kwargs):
        self.active = True

    def deactivate(self):
        self.active = False


class _Engine:
    def __init__(self, workspace):
        self._workspace = str(workspace)
        self._last_status = "done"
        self.is_cancelled = False
        self.stop_status = "done"
        self.cancel_on_execute = False
        self.execute_calls = 0
        self.execute_result = "Partial work preserved."
        self.checks = []
        self.changed = False

    def execute(self, **kwargs):
        self.execute_calls += 1
        self._last_status = self.stop_status
        self.is_cancelled = self.cancel_on_execute
        return self.execute_result

    def get_objective_checks(self):
        return self.checks

    def has_file_changes(self):
        return self.changed

    def get_file_contents(self):
        return {"app.py": "pass\n"}

    def get_file_tracker(self):
        return None

    def get_plan_steps(self):
        return []

    def get_changed_files_summary(self):
        return "app.py changed"

    def get_file_change_reasons(self):
        return {}


class _Reviewer:
    def __init__(self):
        self.calls = 0
        self.on_review = lambda: None

    def reset(self):
        self.calls = 0

    @property
    def can_review_again(self):
        return self.calls < 2

    def _should_multi_pass(self, *args):
        return False

    def review(self, **kwargs):
        self.calls += 1
        self.on_review()
        return ReviewResult("REJECTED", "The change is incomplete", issues=[{
            "severity": "blocking", "description": "Fix the changed behavior",
        }])


class _EvidenceReviewer(_Reviewer):
    def review(self, **kwargs):
        self.calls += 1
        self.on_review()
        return EvidenceReviewResult("REJECTED", "The claim needs evidence", issues=[{
            "severity": "blocking", "claim_excerpt": "certain",
            "problem": "The evidence does not establish certainty",
        }])


@pytest.fixture
def review_env(tmp_path, monkeypatch):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "EVIDENCE_REVIEW_MAX_ROUNDS", 2)
    monkeypatch.setattr(settings, "REVIEW_OBJECTIVE_REVERIFY_ENABLED", True)
    monkeypatch.setattr(settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 1)
    engine = _Engine(tmp_path)
    env = SimpleNamespace(
        engine=engine, agent=_Agent(), reviewer=_Reviewer(),
        verification_calls=0, verification_passed=True,
        on_verification=lambda: None,
    )

    class Verifier:
        def __init__(self, **kwargs):
            pass

        def verify(self, **kwargs):
            env.verification_calls += 1
            env.on_verification()
            return VerificationResult(
                passed=env.verification_passed, summary="test result",
                commands_run=[{
                    "command": "pytest", "exit_code": 0 if env.verification_passed else 1,
                    "output": "passed" if env.verification_passed else "failed",
                }],
            )

    monkeypatch.setattr(
        "infinidev.engine.analysis.verification_engine.VerificationEngine", Verifier,
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.review_engine.collect_automated_checks",
        lambda **kwargs: {"verification_passed": True},
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review._recent_tool_evidence",
        lambda *args: "archived evidence",
    )
    return env


def _run_code_review(env, **kwargs):
    return run_review_rework_loop(
        engine=env.engine, agent=env.agent, session_id="review-lifecycle",
        task_prompt=("Fix the behavior", "Working behavior"),
        initial_result="Original result", reviewer=env.reviewer, **kwargs,
    )


@pytest.mark.parametrize("status", ["cancelled", "failed", "blocked", "exhausted"])
def test_stopped_evidence_rework_does_not_start_another_review(review_env, status):
    env = review_env
    reviewer = _EvidenceReviewer()
    env.engine.stop_status = status

    result, review = run_evidence_review_rework_loop(
        engine=env.engine, agent=env.agent, session_id="evidence-stop",
        task_prompt=("Research", "A supported answer"), initial_result="certain",
        evidence_reviewer=reviewer,
    )

    assert result == "Partial work preserved."
    assert review.is_rejected
    assert reviewer.calls == 1
    assert env.engine.execute_calls == 1
    assert env.engine._last_status == status
    assert env.agent.active is False


def test_cancelled_verification_rework_starts_no_more_checks_or_review(review_env):
    env = review_env
    env.verification_passed = False
    env.engine.stop_status = "cancelled"

    result, _ = _run_code_review(env)

    assert result == "Partial work preserved."
    assert env.verification_calls == 1
    assert env.engine.execute_calls == 1
    assert env.reviewer.calls == 0
    assert env.agent.active is False


def test_failed_objective_rework_starts_no_more_checks_or_review(review_env, monkeypatch):
    env = review_env
    env.engine.stop_status = "failed"
    env.engine.checks = [(1, "Check behavior", StepVerification(kind="command", spec="false"))]
    checks = []

    def verify(_verifier, check):
        checks.append(check.spec)
        return VerificationResult(passed=False, summary="objective failed")

    monkeypatch.setattr("infinidev.engine.analysis.objective_verifier.ObjectiveVerifier.verify", verify)

    result, _ = _run_code_review(env, run_verification=False)

    assert result == "Partial work preserved."
    assert checks == ["false"]
    assert env.engine.execute_calls == 1
    assert env.reviewer.calls == 0


def test_blocked_code_rework_starts_no_more_verification_or_review(review_env):
    env = review_env
    env.engine.stop_status = "blocked"

    result, review = _run_code_review(env)

    assert result == "Partial work preserved."
    assert review.is_rejected
    assert env.verification_calls == 1
    assert env.reviewer.calls == 1
    assert env.engine.execute_calls == 1


@pytest.mark.parametrize("kind", ["code", "evidence"])
def test_existing_cancellation_starts_no_review_work(review_env, kind):
    env = review_env
    env.engine.is_cancelled = True
    reviewer = _EvidenceReviewer()

    if kind == "code":
        result, review = _run_code_review(env)
    else:
        result, review = run_evidence_review_rework_loop(
            engine=env.engine, agent=env.agent, session_id="cancelled-evidence",
            task_prompt=("Research", "Report"), initial_result="Original result",
            evidence_reviewer=reviewer,
        )

    assert result == "Original result"
    assert review is None
    assert env.engine.execute_calls == 0
    assert env.verification_calls == 0
    assert env.reviewer.calls == reviewer.calls == 0


def test_cancellation_during_objective_check_stops_the_next_check(review_env, monkeypatch):
    env = review_env
    env.engine.checks = [
        (index, f"check {index}", StepVerification(kind="command", spec=f"check-{index}"))
        for index in (1, 2)
    ]
    checks = []

    def verify(_verifier, check):
        checks.append(check.spec)
        env.engine.is_cancelled = True
        return VerificationResult(passed=True, summary="check passed")

    monkeypatch.setattr("infinidev.engine.analysis.objective_verifier.ObjectiveVerifier.verify", verify)

    _run_code_review(env, run_verification=False)

    assert checks == ["check-1"]
    assert env.reviewer.calls == 0
    assert env.engine.execute_calls == 0


@pytest.mark.parametrize("kind", ["verification", "code_review", "evidence_review"])
def test_cancellation_during_operation_prevents_developer_rework(review_env, kind):
    env = review_env

    def cancel():
        env.engine.is_cancelled = True

    if kind == "verification":
        env.verification_passed = False
        env.on_verification = cancel
        _run_code_review(env)
    elif kind == "code_review":
        env.reviewer.on_review = cancel
        _run_code_review(env)
    else:
        reviewer = _EvidenceReviewer()
        reviewer.on_review = cancel
        run_evidence_review_rework_loop(
            engine=env.engine, agent=env.agent, session_id="cancelled-evidence",
            task_prompt=("Research", "Report"), initial_result="certain",
            evidence_reviewer=reviewer,
        )

    assert env.engine.execute_calls == 0
    assert env.agent.active is False


@pytest.mark.parametrize("stop_status", ["done", "cancelled"])
@pytest.mark.parametrize("empty_result", [None, "", "  "])
def test_empty_code_rework_preserves_the_previous_partial_answer(
    review_env, stop_status, empty_result,
):
    env = review_env
    env.engine.stop_status = stop_status
    env.engine.execute_result = empty_result

    result, _ = _run_code_review(env)

    assert result == "Original result"
    assert env.engine._last_status == stop_status


@pytest.mark.parametrize("multi_pass", [False, True])
def test_cancellation_during_review_preparation_prevents_the_judge(
    review_env, monkeypatch, multi_pass,
):
    env = review_env

    def prepare(**kwargs):
        env.engine.is_cancelled = True
        return {}

    if multi_pass:
        env.reviewer._should_multi_pass = lambda *args: True
        env.reviewer._run_extraction_pass = prepare
    else:
        monkeypatch.setattr(
            "infinidev.engine.analysis.review_engine.collect_automated_checks", prepare,
        )

    _run_code_review(env)

    assert env.reviewer.calls == 0
    assert env.engine.execute_calls == 0


def test_cancellation_during_evidence_loading_prevents_the_reviewer(review_env, monkeypatch):
    env = review_env
    reviewer = _EvidenceReviewer()

    def load_evidence(*args):
        env.engine.is_cancelled = True
        return "archived evidence"

    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review._recent_tool_evidence", load_evidence,
    )

    result, review = run_evidence_review_rework_loop(
        engine=env.engine, agent=env.agent, session_id="cancelled-evidence",
        task_prompt=("Research", "Report"), initial_result="Original result",
        evidence_reviewer=reviewer,
    )

    assert result == "Original result"
    assert review is None
    assert reviewer.calls == 0
    assert env.engine.execute_calls == 0


class _Hooks:
    def on_phase(self, phase):
        pass

    def on_status(self, level, message):
        pass

    def notify(self, *args):
        pass


def _run_pipeline_review(env):
    from infinidev.engine.orchestration.pipeline import _run_review_phase

    return _run_review_phase(
        engine=env.engine, agent=env.agent, session_id="pipeline-review",
        task_prompt=("Complete the task", "Supported result"), result="Partial result",
        reviewer=env.reviewer, hooks=_Hooks(),
    )


@pytest.mark.parametrize("new_file", [False, True])
def test_evidence_rejection_blocks_pipeline_without_code_approval(
    review_env, monkeypatch, temp_db, new_file,
):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "REVIEW_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCE_REVIEW_ENABLED", True)
    env = review_env
    code_reviews = []

    def evidence_review(**kwargs):
        env.engine.changed = new_file
        return "Unresolved claims", EvidenceReviewResult("REJECTED", "Unsupported")

    def code_review(**kwargs):
        code_reviews.append(True)
        return "Code approved", ReviewResult("APPROVED", "Looks fine")

    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review.run_evidence_review_rework_loop",
        evidence_review,
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.review_engine.run_review_rework_loop", code_review,
    )

    result = _run_pipeline_review(env)

    assert env.engine._last_status == "blocked"
    assert result == "Unresolved claims"
    assert code_reviews == []


def test_evidence_review_error_after_report_creation_does_not_start_code_review(
    review_env, monkeypatch, temp_db,
):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "REVIEW_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCE_REVIEW_ENABLED", True)
    env = review_env
    code_reviews = []

    def evidence_review(**kwargs):
        env.engine.changed = True
        raise RuntimeError("evidence reviewer failed")

    def code_review(**kwargs):
        code_reviews.append(True)
        return "Code approved", ReviewResult("APPROVED", "Looks fine")

    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review.run_evidence_review_rework_loop",
        evidence_review,
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.review_engine.run_review_rework_loop", code_review,
    )

    result = _run_pipeline_review(env)

    assert env.engine._last_status == "failed"
    assert result == "Partial result"
    assert code_reviews == []


@pytest.mark.parametrize("stop_status", ["cancelled", "failed", "blocked", "exhausted"])
@pytest.mark.parametrize("review_outcome", ["error", "rejected"])
def test_code_review_preserves_stopped_rework_outcome(
    review_env, monkeypatch, temp_db, stop_status, review_outcome,
):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "REVIEW_ENABLED", True)
    env = review_env
    env.engine.changed = True

    def code_review(**kwargs):
        env.engine._last_status = stop_status
        if review_outcome == "error":
            raise RuntimeError("review operation failed")
        return "Partial work", ReviewResult("REJECTED", "Unresolved")

    monkeypatch.setattr(
        "infinidev.engine.analysis.review_engine.run_review_rework_loop", code_review,
    )

    _run_pipeline_review(env)

    assert env.engine._last_status == stop_status


@pytest.mark.parametrize("changed", [False, True])
def test_pipeline_skips_all_review_work_after_cancellation(
    review_env, monkeypatch, temp_db, changed,
):
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "REVIEW_ENABLED", True)
    monkeypatch.setattr(settings, "EVIDENCE_REVIEW_ENABLED", True)
    env = review_env
    env.engine.is_cancelled = True
    env.engine.changed = changed
    calls = []

    def review(**kwargs):
        calls.append(True)
        return "Reviewed", None

    monkeypatch.setattr(
        "infinidev.engine.analysis.evidence_review.run_evidence_review_rework_loop", review,
    )
    monkeypatch.setattr(
        "infinidev.engine.analysis.review_engine.run_review_rework_loop", review,
    )

    assert _run_pipeline_review(env) == "Partial result"
    assert calls == []
