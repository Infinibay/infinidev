from __future__ import annotations

import numpy as np

from bench.task_policy_external_review import ExternalReview
from bench.task_policy_pairwise_finetune import (
    PairwiseParameters,
    _pair_weight,
    _select_threshold,
)


def _review(*policies: str, annotation_kind: str = "human") -> ExternalReview:
    return ExternalReview(
        candidate_id="row-1",
        repo="owner/repo",
        language="Python",
        text="Please fix the existing API contract.",
        policies=policies,
        notes="test",
        annotation_kind=annotation_kind,
    )


def test_pair_weight_emphasizes_positive_and_known_hard_negative() -> None:
    review = _review("bugfix.root_cause")

    assert _pair_weight(
        review,
        "bugfix.root_cause",
        example_weight=0.5,
        positive_weight=3.0,
        hard_negative_weight=2.0,
    ) == 1.5
    assert _pair_weight(
        review,
        "feature.contract_first",
        example_weight=0.5,
        positive_weight=3.0,
        hard_negative_weight=2.0,
    ) == 1.0
    assert _pair_weight(
        review,
        "research.evidence_first",
        example_weight=0.5,
        positive_weight=3.0,
        hard_negative_weight=2.0,
    ) == 0.5


def test_threshold_selection_prioritizes_actual_accuracy_and_recall_gate() -> None:
    scores = np.asarray([0.95] * 95 + [0.40] * 5 + [0.30] * 95 + [0.90] * 5)
    expected = np.asarray([True] * 100 + [False] * 100)

    threshold = _select_threshold(
        scores,
        expected,
        accuracy_target=0.95,
        recall_target=0.95,
    )
    predicted = scores >= threshold

    assert np.mean(predicted == expected) >= 0.95
    assert np.mean(predicted[:100]) >= 0.95


def test_pairwise_defaults_preserve_the_pretrained_nli_head() -> None:
    parameters = PairwiseParameters(model_name="example/model")

    assert parameters.head == "nli"
