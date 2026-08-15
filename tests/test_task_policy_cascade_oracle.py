"""Tests for the offline Qwen and MiniMax cascade ceiling benchmark."""

from __future__ import annotations

import json

import pytest

from bench.task_policy_cascade_oracle import (
    METHOD_LABELS,
    benchmark,
    compare_predictions,
    load_examples,
)


def _jsonl(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_examples_joins_short_and_canonical_policy_names(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    _jsonl(candidates, [
        {"candidate_id": "a", "issue_text": "Fix it"},
        {"candidate_id": "b", "issue_text": "Ignore it"},
    ])
    _jsonl(reviews, [
        {"candidate_id": "a", "include": True, "policies": ["bugfix"]},
        {"candidate_id": "b", "include": False, "policies": []},
    ])
    _jsonl(proposals, [
        {"candidate_id": "a", "policies": ["bugfix.root_cause"]},
    ])

    examples = load_examples(candidates, reviews, proposals)

    assert examples == [{
        "candidate_id": "a",
        "text": "Fix it",
        "expected": frozenset({"bugfix.root_cause"}),
        "llm": frozenset({"bugfix.root_cause"}),
    }]


def test_load_examples_rejects_missing_proposals_and_unknown_labels(tmp_path) -> None:
    candidates = tmp_path / "candidates.jsonl"
    reviews = tmp_path / "reviews.jsonl"
    proposals = tmp_path / "proposals.jsonl"
    _jsonl(candidates, [{"candidate_id": "a", "issue_text": "Fix it"}])
    _jsonl(reviews, [{"candidate_id": "a", "include": True, "policies": ["bugfix"]}])
    _jsonl(proposals, [])
    with pytest.raises(ValueError, match="missing MiniMax proposal"):
        load_examples(candidates, reviews, proposals)

    _jsonl(proposals, [{"candidate_id": "a", "policies": ["maintenance"]}])
    with pytest.raises(ValueError, match="unknown policy"):
        load_examples(candidates, reviews, proposals)


def test_compare_predictions_reports_boolean_rules_and_oracle_ceiling() -> None:
    bugfix, feature = METHOD_LABELS[:2]
    expected = [frozenset({bugfix}), frozenset({feature})]
    qwen = [frozenset({bugfix, feature}), frozenset()]
    llm = [frozenset(), frozenset()]

    report = compare_predictions(expected, qwen, llm)

    assert report["strategies"]["or"]["exact_match"] == 0.0
    assert report["strategies"]["and"]["exact_match"] == 0.0
    assert report["strategies"]["label_wise_oracle"]["exact_match"] == 0.5
    assert report["strategies"]["label_wise_oracle"]["per_label"][bugfix]["recall"] == 1.0
    assert report["strategies"]["label_wise_oracle"]["per_label"][feature]["recall"] == 0.0
    assert report["disagreement"]["examples_with_disagreement"] == 1
    assert report["disagreement"]["example_rate"] == 0.5
    assert report["disagreement"]["label_decisions"] == 2
    assert report["disagreement"]["label_rate"] == pytest.approx(1 / 6)
    assert report["ceiling_95_95_viable"] is False


def test_label_wise_oracle_reports_a_viable_ceiling_when_errors_are_complementary() -> None:
    expected = [frozenset(METHOD_LABELS)]
    qwen = [frozenset(METHOD_LABELS[::2])]
    llm = [frozenset(METHOD_LABELS[1::2])]

    report = compare_predictions(expected, qwen, llm)

    oracle = report["strategies"]["label_wise_oracle"]
    assert oracle["exact_match"] == 1.0
    assert all(metrics["accuracy"] == 1.0 for metrics in oracle["per_label"].values())
    assert all(metrics["recall"] == 1.0 for metrics in oracle["per_label"].values())
    assert report["ceiling_95_95_viable"] is True


def test_benchmark_uses_injected_qwen_predictor_without_model_loading() -> None:
    bugfix, feature = METHOD_LABELS[:2]
    examples = [
        {
            "candidate_id": "a",
            "text": "fix request",
            "expected": frozenset({bugfix}),
            "llm": frozenset(),
        },
        {
            "candidate_id": "b",
            "text": "feature request",
            "expected": frozenset({feature}),
            "llm": frozenset({feature}),
        },
    ]
    predictions = {
        "fix request": (bugfix,),
        "feature request": (),
    }

    report = benchmark(examples, qwen_predictor=lambda text: predictions[text])

    assert report["strategies"]["qwen"]["exact_match"] == 0.5
    assert report["strategies"]["llm"]["exact_match"] == 0.5
    assert report["strategies"]["or"]["exact_match"] == 1.0
    assert report["strategies"]["label_wise_oracle"]["exact_match"] == 1.0
    assert report["ceiling_95_95_viable"] is False
