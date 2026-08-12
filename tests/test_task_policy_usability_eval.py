"""Tests for operational task-policy usability evaluation."""

from __future__ import annotations

import pytest

from bench.task_policy_usability_eval import evaluate_report


def _row(
    row_id: str,
    policies: list[str],
    *,
    language: str = "en",
    reason: str | None = None,
) -> dict[str, object]:
    return {
        "id": row_id,
        "policies": policies,
        "language": language,
        "difficulty": "D1_paraphrase",
        "batch": "test",
        "uncategorized_reason": reason,
    }


def test_evaluate_report_reconstructs_exact_predictions_and_error_directions() -> None:
    rows = [
        _row("one", ["bugfix.root_cause"]),
        _row("two", [] , reason="conceptual_question"),
        _row("three", ["feature.contract_first", "research.evidence_first"]),
    ]
    report = {
        "model": "candidate",
        "head": "test_head",
        "examples": 3,
        "errors": [
            {"id": "two", "predicted": ["review.read_only"]},
            {"id": "three", "predicted": ["feature.contract_first"]},
        ],
    }

    result = evaluate_report(report, rows)

    assert result["overall"]["exact_match"] == pytest.approx(1 / 3)
    assert result["overall"]["false_activations"] == 1
    assert result["overall"]["error_direction"] == {
        "over": 1,
        "under": 1,
        "mixed": 0,
    }
    assert result["slices"]["cardinality"]["2"]["exact_match"] == 0.0
    assert not result["deployment"]["automatic_prompt_injection"]["usable"]


def test_evaluate_report_accepts_a_perfect_report_for_automatic_use() -> None:
    rows = [
        _row(str(index), ["bugfix.root_cause"] if index < 10 else [], reason=(
            None if index < 10 else "conceptual_question"
        ))
        for index in range(20)
    ]
    report = {"model": "perfect", "examples": 20, "errors": []}

    result = evaluate_report(report, rows)

    assert result["deployment"]["automatic_prompt_injection"]["usable"]
    assert result["deployment"]["shadow_or_advisory"]["usable"]


def test_evaluate_report_rejects_partial_cross_validation() -> None:
    rows = [_row("one", ["bugfix.root_cause"]), _row("two", [])]
    report = {"examples": 2, "evaluated_examples": 1, "errors": []}

    with pytest.raises(ValueError, match="complete out-of-fold"):
        evaluate_report(report, rows)
