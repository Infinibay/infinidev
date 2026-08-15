"""Tests for the selected-main-model task classifier evaluation harness."""

from __future__ import annotations

from bench.task_policy_main_model_classifier_eval import (
    METHOD_LABELS,
    parse_binary_judges,
    select_stratified_sample,
    summarize,
)


def test_stratified_sample_covers_methods_zero_and_compound() -> None:
    rows = []
    for label in METHOD_LABELS:
        for index, size in enumerate((10, 30, 90)):
            rows.append({
                "candidate_id": f"{label}-{index}",
                "text": "x" * size,
                "expected": (label,),
            })
    rows.extend([
        {"candidate_id": "zero-1", "text": "x" * 20, "expected": ()},
        {"candidate_id": "zero-2", "text": "x" * 60, "expected": ()},
        {
            "candidate_id": "compound",
            "text": "x" * 40,
            "expected": ("research", "bugfix"),
        },
    ])

    sample = select_stratified_sample(
        rows, per_label=2, zero_label=2, compound=1,
    )

    assert len(sample) == 15
    assert {label for row in sample for label in row["expected"]} == set(METHOD_LABELS)
    assert sum(not row["expected"] for row in sample) == 2
    assert sum(len(row["expected"]) > 1 for row in sample) == 1


def test_summary_reports_exact_label_metrics_and_latency() -> None:
    records = [
        {
            "expected": ["bugfix"],
            "predicted": ["bugfix"],
            "latency_ms": 100.0,
        },
        {
            "expected": ["bugfix", "research"],
            "predicted": ["research"],
            "latency_ms": 300.0,
        },
        {"expected": [], "predicted": None, "latency_ms": 500.0},
    ]

    result = summarize(records)

    assert result["calls"] == 3
    assert result["valid"] == 2
    assert result["failures"] == 1
    assert result["exact_match"] == 0.5
    assert result["per_label"]["bugfix"]["precision"] == 1.0
    assert result["per_label"]["bugfix"]["recall"] == 0.5
    assert result["per_label"]["bugfix"]["accuracy"] == 0.5
    assert result["per_label"]["research"]["f1"] == 1.0
    assert result["gate"]["all_labels_pass"] is False
    assert result["latency_ms"] == {"p50": 300.0, "p95": 480.0, "max": 500.0}


def test_parse_binary_judges_requires_and_preserves_all_independent_decisions() -> None:
    decisions = {
        label: {"selected": label in {"bugfix", "review"}, "confidence": 0.9}
        for label in METHOD_LABELS
    }

    result = parse_binary_judges(
        "prefix " + __import__("json").dumps({"decisions": decisions}) + " suffix",
    )

    assert result["operations"] == ["bugfix", "review"]
    assert result["confidences"] == {label: 0.9 for label in METHOD_LABELS}
