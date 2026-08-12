"""Quality gates for the large synthetic task-policy augmentation."""

from __future__ import annotations

from collections import Counter

from bench.task_policy_large_calibration import (
    audit_large_calibration_corpus,
    build_large_calibration_corpus,
)
from bench.task_policy_linear_head import (
    build_ambiguity_challenge,
    build_linear_head_holdout,
)
from bench.task_policy_semantic_eval import build_semantic_validation_corpus


def test_large_calibration_is_balanced_and_family_diverse() -> None:
    report = audit_large_calibration_corpus()

    assert report["examples"] == 736
    assert report["duplicate_ids"] == 0
    assert report["duplicate_texts"] == 0
    assert report["by_policy"]["uncategorized"] == 160
    assert all(
        count == 96
        for policy, count in report["by_policy"].items()
        if policy != "uncategorized"
    )
    assert all(count >= 12 for count in report["families_by_policy"].values())
    assert report["neutral_families"] >= 20


def test_large_calibration_rows_have_consistent_authority_and_text() -> None:
    examples = build_large_calibration_corpus()
    modifying = {
        "bugfix.root_cause",
        "feature.contract_first",
        "refactor.preserve_behavior",
        "performance.measure_first",
    }
    counts = Counter(example.policy or "uncategorized" for example in examples)

    assert len(counts) == 7
    for example in examples:
        assert 25 <= len(example.text) <= 260
        assert "{object}" not in example.text
        assert example.write_authority is (example.policy in modifying)


def test_large_calibration_does_not_copy_evaluation_text() -> None:
    calibration = {example.text for example in build_large_calibration_corpus()}
    evaluation = {
        example.text
        for example in (
            build_semantic_validation_corpus()
            + build_ambiguity_challenge()
            + build_linear_head_holdout()
        )
    }

    assert calibration.isdisjoint(evaluation)
