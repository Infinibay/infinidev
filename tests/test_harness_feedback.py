"""Tests for structured, non-authoritative harness feedback."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pytest

from bench.harness_feedback import (
    FeedbackCase,
    FeedbackObservation,
    HarnessFeedback,
    build_feedback_report,
    load_cases,
    render_markdown,
)


def _case() -> FeedbackCase:
    return FeedbackCase.from_dict(
        {
            "id": "c1",
            "category": "prompt_overload",
            "scenario": "history grew",
            "visible_artifact": "history=18000 chars; task=600 chars",
            "question": "What should be tested?",
            "review_status": "approved",
        }
    )


def _feedback() -> HarnessFeedback:
    return HarnessFeedback.from_text(
        '{"no_change_warranted":false,"assessment":"History dominates the static prompt.",'
        '"friction":"Older actions occupy most user context.","evidence":"18k versus 600 chars.",'
        '"suggested_change":"Collapse older action details.","expected_effect":"Lower input size '
        'with unchanged recall.","risk":"A needed prior decision may disappear.","experiment":"Pair '
        'full and collapsed history on held-out tasks; gate on success and evidence recall."}'
    )


def test_feedback_requires_evidence_risk_and_falsifiable_experiment() -> None:
    feedback = _feedback()
    assert feedback.suggested_change == "Collapse older action details."
    with pytest.raises(ValueError, match="needs friction"):
        HarnessFeedback.from_text(
            '{"no_change_warranted":false,"assessment":"bad","friction":"",'
            '"evidence":"x","suggested_change":"x","expected_effect":"x",'
            '"risk":"x","experiment":"x"}'
        )


def test_no_change_response_cannot_smuggle_a_suggestion() -> None:
    with pytest.raises(ValueError, match="must not smuggle"):
        HarnessFeedback.from_text(
            '{"no_change_warranted":true,"assessment":"No issue shown.","friction":"",'
            '"evidence":"","suggested_change":"Still shorten it.","expected_effect":"",'
            '"risk":"","experiment":""}'
        )


def test_report_preserves_raw_feedback_and_marks_it_unverified() -> None:
    case = _case()
    feedback = _feedback()
    row = FeedbackObservation(
        case_id=case.id,
        case_sha256=case.sha256,
        model_identity="provider/model@revision",
        repetition=0,
        response_text="raw model response",
        feedback=feedback,
    )

    report = build_feedback_report([case], [row])

    record = report["categories"]["prompt_overload"][0]
    assert record["raw_response"] == "raw model response"
    assert record["feedback"] == asdict(feedback)
    assert "Unverified model-authored hypothesis" in record["interpretation"]
    assert "Collapse older action details" in render_markdown(report)


def test_checked_in_feedback_cases_cover_every_category() -> None:
    cases = load_cases(Path("bench/harness_feedback_cases.example.jsonl"))
    assert len(cases) == 9
    assert len({case.category for case in cases}) == 9
    assert all(case.review_status == "draft" for case in cases)
