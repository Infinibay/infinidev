"""Smoke tests for the experimental behavior head."""

from __future__ import annotations

from bench.behavior_linear_head import run_experiment


def test_behavior_linear_head_preserves_split_identity() -> None:
    report = run_experiment()

    assert report["calibration_examples"] >= 250
    assert report["validation_sha256"] != report["holdout_sha256"]
    assert report["embedding_space_id"].startswith("ken/static-qwen3-r512-v2:")
