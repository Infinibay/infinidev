"""Split integrity tests for the experimental static-embedding linear head."""

from __future__ import annotations

from bench.task_policy_hierarchical_head import run_experiment as run_hierarchical_experiment
from bench.task_policy_linear_head import build_linear_head_holdout, run_experiment
from bench.task_policy_semantic_eval import build_semantic_validation_corpus


def test_linear_head_holdout_is_separate_and_unique() -> None:
    validation = build_semantic_validation_corpus()
    holdout = build_linear_head_holdout()

    assert len(holdout) >= 100
    assert not {item.id for item in validation} & {item.id for item in holdout}
    assert not {item.text for item in validation} & {item.text for item in holdout}
    assert len({item.text for item in holdout}) == len(holdout)


def test_linear_head_experiment_records_space_and_splits() -> None:
    report = run_experiment()

    assert report["embedding_space_id"].startswith(
        "ken/static-qwen3-r512-v2:1024:"
    )
    assert report["calibration_examples"] >= 300
    assert report["holdout"]["examples"] >= 100
    assert report["validation_sha256"] != report["holdout_sha256"]


def test_hierarchical_head_improves_selective_safety() -> None:
    report = run_hierarchical_experiment()

    assert report["schema_version"] == 2
    assert report["calibration_examples"] >= 1_500
    assert report["selection"]["selective_precision"] == 1.0
    assert report["ambiguity_development"]["false_activations"] == 0
    assert report["holdout"]["selective_precision"] >= 0.92
    assert report["holdout"]["false_activations"] <= 1
