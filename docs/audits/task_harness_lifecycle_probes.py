"""Offline characterization probes for the 2026-09-13 Task harness audit.

Run with ``uv run python docs/audits/task_harness_lifecycle_probes.py``.
Assertions describe observed defects, not the desired product contract. These
probes intentionally live outside the regression suite; fixes should reverse
the relevant assertions in focused tests under tests/.
"""

from __future__ import annotations

import json
import logging
from contextlib import ExitStack
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

from infinidev.config.settings import settings
from infinidev.engine.analysis.evidence_review import (
    EvidenceReviewResult,
    run_evidence_review_rework_loop,
)
from infinidev.engine.analysis.review_engine import ReviewEngine, run_review_rework_loop
from infinidev.engine.analysis.review_result import ReviewResult
from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.verification_engine import VerificationEngine
from infinidev.engine.analysis.verification_result import VerificationResult
from infinidev.engine.engines.task import _bootstrap_step
from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.plan_step import PlanStep
from infinidev.engine.loop.resume_checkpoint import (
    build_loop_resume_checkpoint,
    resume_state_for_task,
)
from infinidev.engine.loop.step_complete_gate import StepCompleteGate
from infinidev.engine.loop.step_manager import StepManager
from infinidev.engine.loop.step_operation import StepOperation
from infinidev.engine.loop.step_result import StepResult
from infinidev.engine.loop.user_message_injector import UserMessageInjector
from infinidev.engine.orchestration.pipeline import _run_review_phase
from infinidev.tools.meta.plan_tools import AddStepInput, ModifyStepInput


def evidence_budget() -> dict:
    rejected = EvidenceReviewResult("REJECTED", "Unsupported claim", [{
        "severity": "blocking", "claim_excerpt": "certain", "problem": "unsupported",
    }])
    observations = {}
    for limit in (0, None, 100):
        engine = Mock(_last_state=SimpleNamespace(total_tool_calls=5))
        engine.execute.return_value = "qualified"
        reviewer = Mock()
        reviewer.review.side_effect = [rejected, EvidenceReviewResult("APPROVED", "ok")]
        with patch(
            "infinidev.engine.analysis.evidence_review._recent_tool_evidence",
            return_value="evidence",
        ), patch.object(settings, "EVIDENCE_REVIEW_MAX_ROUNDS", 2):
            result, review = run_evidence_review_rework_loop(
                engine=engine, agent=Mock(), session_id="audit",
                task_prompt=("Investigate", "Report"), initial_result="certain",
                evidence_reviewer=reviewer, max_total_tool_calls=limit,
            )
        observations[str(limit)] = {
            "rework_calls": engine.execute.call_count, "result": result,
            "verdict": review.verdict,
        }
    assert observations["0"]["rework_calls"] == 0
    assert observations["None"]["rework_calls"] == 1
    assert observations["100"]["rework_calls"] == 1
    return observations


def rejected_evidence_status() -> dict:
    engine = Mock(_last_status="done")
    engine.has_file_changes.return_value = False
    with patch.object(settings, "REVIEW_ENABLED", True), patch.object(
        settings, "EVIDENCE_REVIEW_ENABLED", True,
    ), patch(
        "infinidev.engine.analysis.evidence_review.run_evidence_review_rework_loop",
        return_value=("unsupported answer", EvidenceReviewResult("REJECTED", "not grounded")),
    ):
        result = _run_review_phase(
            engine=engine, agent=Mock(), session_id="audit", task_prompt=("Ask", "Result"),
            result="unsupported answer", reviewer=None, hooks=Mock(),
        )
    assert engine._last_status == "done"
    return {"review": "REJECTED", "engine_status": engine._last_status, "result": result}


def checkpoint_identity() -> dict:
    task = SimpleNamespace(title="Implement export", description="Implement CSV export", kind="feature")
    followup = SimpleNamespace(title="Continue please", description="User request: seguí", kind="chore")
    prompt = ("Original task", "Deliver")
    state = LoopState(plan=LoopPlan(steps=[
        PlanStep(index=1, title="Implement export", status="active"),
    ]))
    checkpoint = build_loop_resume_checkpoint(state, prompt, task)
    cancelled = build_loop_resume_checkpoint(state, prompt, task, terminal_status="cancelled")
    result = {
        "same_task_resumes": resume_state_for_task(checkpoint, prompt, task) is not None,
        "followup_resumes": resume_state_for_task(checkpoint, prompt, followup) is not None,
        "cancelled_same_task_resumes": resume_state_for_task(cancelled, prompt, task) is not None,
    }
    assert result == {
        "same_task_resumes": True, "followup_resumes": False,
        "cancelled_same_task_resumes": False,
    }
    return result


def task_check_authoring() -> dict:
    step = _bootstrap_step(SimpleNamespace(title="Investigate queue behavior", kind="investigation"))
    result = {
        "bootstrap_verify": step.verify,
        "add_can_author_verify": "verify" in AddStepInput.model_fields,
        "modify_can_author_verify": "verify" in ModifyStepInput.model_fields,
        "operation_can_author_verify": "verify" in StepOperation.model_fields,
        "investigation_asks_for_change": "concrete change Step" in step.expected_output,
    }
    assert result == {
        "bootstrap_verify": None, "add_can_author_verify": False,
        "modify_can_author_verify": False, "operation_can_author_verify": False,
        "investigation_asks_for_change": True,
    }
    return result


def objectives_can_stale_tests() -> dict:
    events = []
    check = StepVerification(kind="file_contains", spec="result.txt", observable="ok")
    engine = Mock(_workspace="/audit", _last_state=None)
    engine.get_file_contents.return_value = {"result.txt": "ok"}
    engine.get_objective_checks.return_value = [(1, "Verify result", check)]
    engine.get_plan_steps.return_value = []
    engine.get_changed_files_summary.return_value = "diff"
    engine.get_file_change_reasons.return_value = {}
    engine.get_file_tracker.return_value = None
    engine.execute.side_effect = lambda **kwargs: events.append("objective_rework_mutates") or "fixed"
    verifier = Mock()
    verifier.verify.side_effect = lambda **kwargs: events.append("tests_pass") or VerificationResult(
        passed=True, summary="Tests passed",
    )
    objective = Mock()
    objective.verify.side_effect = [
        VerificationResult(passed=False, summary="Objective fails"),
        VerificationResult(passed=True, summary="Objective passes"),
    ]
    reviewer = Mock()
    reviewer._should_multi_pass.return_value = False
    reviewer.review.side_effect = lambda **kwargs: events.append("review_approves") or ReviewResult(
        verdict="APPROVED", summary="Approved",
    )
    with ExitStack() as stack:
        stack.enter_context(patch.object(settings, "REVIEW_OBJECTIVE_REVERIFY_ENABLED", True))
        stack.enter_context(patch.object(settings, "REVIEW_OBJECTIVE_REVERIFY_MAX_ROUNDS", 2))
        stack.enter_context(patch(
            "infinidev.engine.analysis.review_engine._review_workspace", return_value="/audit",
        ))
        stack.enter_context(patch(
            "infinidev.engine.analysis.verification_engine.VerificationEngine", return_value=verifier,
        ))
        stack.enter_context(patch(
            "infinidev.engine.analysis.objective_verifier.ObjectiveVerifier", return_value=objective,
        ))
        stack.enter_context(patch("infinidev.db.service.get_objective_verdicts", return_value=[]))
        stack.enter_context(patch("infinidev.db.service.record_objective_verdict"))
        stack.enter_context(patch(
            "infinidev.engine.analysis.review_engine.collect_automated_checks", return_value={},
        ))
        _, review = run_review_rework_loop(
            engine=engine, agent=Mock(), session_id="audit", task_prompt=("Ask", "Result"),
            initial_result="draft", reviewer=reviewer,
        )
    assert events == ["tests_pass", "objective_rework_mutates", "review_approves"]
    assert review.is_approved
    return {"events": events, "test_runs": verifier.verify.call_count, "verdict": review.verdict}


def attachment_delivery() -> dict:
    injector = UserMessageInjector()
    injector.inject("Use this screenshot", [SimpleNamespace(path="image.png")])
    drained = injector.drain()
    injector.inject("Use this screenshot", [SimpleNamespace(path="image.png")])
    messages = [{"role": "tool", "tool_call_id": "close", "content": "ack"}]
    with patch("infinidev.engine.loop.user_message_injector._emit_log"):
        held = injector.reject_step_complete_on_late_message(
            SimpleNamespace(project_id=1, agent_id="audit"), messages, "close",
        )
    assert drained == ["Use this screenshot"]
    assert held and "image.png" not in json.dumps(messages)
    return {"step_boundary_payload": drained, "late_payload_is_text": True}


def verification_fail_open() -> dict:
    active = PlanStep(index=1, title="Verify artifact", status="active")
    check = StepVerification(kind="file_contains", spec="result.txt", observable="ok")
    active.verify = check
    ctx = SimpleNamespace(
        project_id=1, agent_id="audit", state=LoopState(plan=LoopPlan(steps=[active])),
    )
    gate = StepCompleteGate(Mock())
    call = SimpleNamespace(id="close")
    failure = VerificationResult(passed=False, summary="Missing required output")
    with patch("infinidev.engine.loop.step_complete_gate.emit_log"), patch(
        "infinidev.engine.loop.step_complete_gate.emit_loop_event",
    ):
        results = [gate._block_for_correction(
            ctx, [], call, active, check, failure,
            SimpleNamespace(LOOP_OBJECTIVE_VERIFY_MAX_ATTEMPTS=3),
        ) for _ in range(4)]
    assert results == [True, True, True, False]
    return {"held_for_attempts_1_to_4": results, "warning_notes": len(ctx.state.notes)}


def empty_verification() -> dict:
    with TemporaryDirectory() as workspace:
        result = VerificationEngine(workspace).verify(changed_files=[])
    assert result.passed and not result.commands_run
    return {"passed": result.passed, "commands": result.commands_run, "summary": result.summary}


def implicit_done_with_blocked_step() -> dict:
    engine = Mock()
    engine._apply_guardrail.side_effect = lambda _ctx, result, *a, **kw: result
    ctx = SimpleNamespace(
        state=LoopState(plan=LoopPlan(steps=[
            PlanStep(index=1, title="Required work", status="blocked"),
        ])),
        guardrail=None, guardrail_max_retries=0, llm_params={}, system_prompt="",
        desc="Task", expected="Result", tool_schemas=[], tool_dispatch={}, max_per_action=0,
    )
    manager = Mock()
    LoopEngine._check_termination(
        engine, ctx, StepResult(status="continue", summary="No further action"), manager, 3, 2,
    )
    status = manager.finish.call_args.args[1]
    assert status == "done"
    ctx.state.plan.steps.append(PlanStep(index=2, title="Other work", status="active"))
    with patch("infinidev.engine.loop.step_manager._fold_verified_model_steps"):
        reconciled = StepManager(Mock()).reconcile_task_completion(
            ctx, StepResult(status="done", summary="Other work finished"),
        )
    assert reconciled.status == "done"
    return {
        "plan_step_status": "blocked", "implicit_terminal_status": status,
        "explicit_done_with_prior_blocked": reconciled.status,
    }


def summary_step_identity() -> dict:
    state = LoopState(plan=LoopPlan(steps=[
        PlanStep(index=1, title="Implemented change", status="done"),
        PlanStep(index=2, title="Next work", status="active"),
        PlanStep(index=3, title="Discarded tactic", status="skipped"),
    ]))
    ctx = SimpleNamespace(state=state, is_small=False)
    manager = StepManager(SimpleNamespace(_summarizer_override=False))
    manager._archive_evicted_context = Mock(return_value=[])
    manager._record_outcome = Mock()
    manager._step_end_summary_hook = Mock(return_value="")
    manager._record_command_output_notes = Mock(return_value="")
    manager._arm_semantic_stagnation_control = Mock()
    manager.summarize_and_record(
        ctx, StepResult(status="continue", summary="Implemented change"), [], 1, 0,
    )
    assert state.history[-1].step_index == 3
    return {"actually_completed": 1, "summary_recorded_against": state.history[-1].step_index}


def skipped_review_on_error() -> dict:
    reviewer = ReviewEngine()
    reviewer._completion_with_caching = Mock(side_effect=RuntimeError("provider unavailable"))
    result = reviewer._single_pass_review(
        llm_params={}, task_description="Task", developer_result="Done", file_changes_summary="diff",
        previous_feedback="", file_reasons={}, file_contents={"a.txt": "changed"},
        recent_messages=[], plan_steps=[], automated_checks={},
    )
    assert result.verdict == "SKIPPED"
    return {"verdict": result.verdict, "summary": result.summary}


def main() -> None:
    logging.disable(logging.CRITICAL)
    # No provider requests are permitted, including unexpected future paths.
    with patch("litellm.completion", side_effect=AssertionError("Offline audit")):
        result = {
            probe.__name__: probe() for probe in (
                evidence_budget, rejected_evidence_status, checkpoint_identity,
                task_check_authoring, objectives_can_stale_tests, attachment_delivery,
                verification_fail_open, empty_verification, implicit_done_with_blocked_step,
                summary_step_identity, skipped_review_on_error,
            )
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
