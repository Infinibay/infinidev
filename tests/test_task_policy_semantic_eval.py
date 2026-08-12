"""Integrity and safety gates for semantic task-policy validation."""

from __future__ import annotations

from bench.task_policy_semantic_eval import (
    build_semantic_validation_corpus,
    evaluate_semantic_profiles,
)


def test_semantic_validation_corpus_is_large_and_unique() -> None:
    examples = build_semantic_validation_corpus()

    assert len(examples) >= 100
    assert len({example.id for example in examples}) == len(examples)
    assert len({example.text for example in examples}) == len(examples)


def test_semantic_validation_reports_selective_safety_metrics() -> None:
    report = evaluate_semantic_profiles(build_semantic_validation_corpus())

    assert report["examples"] >= 100
    assert report["embedding_model"] == "ken/static-qwen3-r512-v2"
    assert report["false_write_authority_rate"] == 0.0
    assert report["false_activation_rate"] == 0.0
    assert report["selective_precision"] >= 0.90
    assert report["space_ids"]
