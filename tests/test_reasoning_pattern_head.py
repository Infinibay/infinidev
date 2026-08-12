"""Dataset and artifact tests for the reasoning mini-head."""

from __future__ import annotations

import json

import numpy as np

from bench.reasoning_pattern_head import LABELS, build_corpus, run_experiment
from infinidev.engine.behavior.reasoning_classifier import (
    ReasoningFeatures,
    classify_reasoning,
)


def test_reasoning_corpus_has_explicit_neutral_and_disjoint_splits() -> None:
    splits = [build_corpus(name) for name in ("calibration", "validation", "holdout")]

    assert all({item.label for item in split} == set(LABELS) for split in splits)
    texts = [{item.text for item in split} for split in splits]
    assert not texts[0] & texts[1]
    assert not texts[0] & texts[2]
    assert not texts[1] & texts[2]


def test_reasoning_head_artifact_has_frozen_space_and_safe_holdout(tmp_path) -> None:
    artifact = tmp_path / "reasoning-head.npz"

    report = run_experiment(artifact)

    assert report["validation"]["unsafe_activation_rate"] == 0.0
    assert report["holdout"]["unsafe_activation_rate"] == 0.0
    assert report["holdout"]["selective_precision"] >= 0.9
    with np.load(artifact, allow_pickle=False) as payload:
        metadata = json.loads(payload["metadata"].tobytes())
        assert payload["weights"].shape == (1034, len(LABELS))
    assert metadata["embedding_space_id"].startswith(
        "ken/static-qwen3-r512-v2:1024:"
    )


def test_packaged_head_classifies_visible_reasoning_and_abstains_safely() -> None:
    excessive = classify_reasoning(
        "The target and test are already loaded, but I will keep browsing unrelated files.",
        ReasoningFeatures(
            modifying_task=1.0,
            discovery_pressure=1.0,
            required_work_pending=1.0,
            evidence_seen=1.0,
        ),
    )
    hard_negative = classify_reasoning(
        "This might be a cache bug, but I need source or test evidence before claiming it.",
        ReasoningFeatures(),
    )

    assert excessive.label == "excessive_exploration"
    assert hard_negative.label in {None, "uncategorized"}
