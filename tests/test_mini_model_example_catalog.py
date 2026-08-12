"""Coverage and quality gates for conditional-prompt example categories."""

from __future__ import annotations

from collections import Counter

from bench.authority_example_corpus import (
    audit_authority_corpus,
    build_authority_corpus,
)
from bench.message_pattern_corpus import audit_message_pattern_corpus
from bench.mini_model_example_catalog import audit_example_catalog
from bench.reasoning_synthetic_corpus import audit_reasoning_augmentation
from bench.task_policy_compound_corpus import audit_compound_corpus
from bench.task_policy_discourse_corpus import audit_discourse_corpus
from infinidev.engine.task_policies.router import resolve_task_profile


def test_every_declared_example_category_exceeds_its_target() -> None:
    report = audit_example_catalog()

    assert report["categories"] >= 40
    assert report["below_minimum"] == []
    assert report["below_target"] == []
    assert all(row["positives"] >= 48 for row in report["rows"])
    assert all(row["negatives"] >= 64 for row in report["rows"])


def test_open_ended_categories_are_intentionally_large() -> None:
    rows = {row["id"]: row for row in audit_example_catalog()["rows"]}

    assert rows["authority.answer_only"]["positives"] >= 256
    assert rows["reasoning.uncategorized"]["positives"] >= 256
    assert rows["message.uncategorized"]["positives"] >= 256


def test_compound_and_discourse_splits_are_balanced_and_disjoint() -> None:
    compound = audit_compound_corpus()
    discourse = audit_discourse_corpus()

    assert compound["pairs"] == 7
    assert compound["duplicate_ids"] == 0
    assert compound["duplicate_texts"] == 0
    assert not any(compound["cross_split_text_overlap"].values())
    assert all(
        count >= 48
        for key, count in compound["pair_counts"].items()
        if key.startswith("calibration:")
    )
    assert discourse["duplicate_ids"] == 0
    assert discourse["duplicate_texts"] == 0
    assert discourse["cross_split_overlap"] == 0
    assert set(discourse["calibration_by_category"].values()) == {48}


def test_reasoning_message_and_authority_augmentations_are_unique() -> None:
    reports = (
        audit_reasoning_augmentation(),
        audit_message_pattern_corpus(),
        audit_authority_corpus(),
    )

    assert all(report["duplicate_ids"] == 0 for report in reports)
    assert all(report["duplicate_texts"] == 0 for report in reports)


def test_literal_authority_corpus_exceeds_ninety_five_percent_accuracy() -> None:
    errors = Counter()
    examples = build_authority_corpus()
    for example in examples:
        profile = resolve_task_profile(example.text, enable_embeddings=False)
        actual = set(profile.authority)
        if not set(example.required) <= actual or set(example.forbidden) & actual:
            errors[example.category] += 1

    accuracy = (len(examples) - sum(errors.values())) / len(examples)
    assert accuracy > 0.95, errors
