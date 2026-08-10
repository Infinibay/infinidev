"""Tests for the offline semantic phase-detector benchmark."""

from __future__ import annotations

from bench.embedding_phase_detection import LABELS, OBJECTS, TEMPLATES, build_examples


def test_phase_detection_splits_do_not_share_objects_or_template_families() -> None:
    examples = build_examples()

    for language in ("en", "es"):
        object_sets = [set(OBJECTS[split][language]) for split in OBJECTS]
        assert all(
            not left & right
            for index, left in enumerate(object_sets)
            for right in object_sets[index + 1:]
        )

    families = {
        split: {row.family for row in examples if row.split == split}
        for split in OBJECTS
    }
    assert not families["train"] & families["calibration"]
    assert not families["train"] & families["test"]
    assert not families["calibration"] & families["test"]


def test_phase_detection_dataset_is_large_balanced_and_bilingual() -> None:
    examples = build_examples()
    test = [row for row in examples if row.split == "test"]

    assert len(examples) >= 800
    assert len(test) >= 500
    assert {row.language for row in examples} == {"en", "es"}
    assert {row.label for row in examples} == set(LABELS)
    for language in ("en", "es"):
        counts = {
            label: sum(row.language == language and row.label == label for row in test)
            for label in LABELS
        }
        assert min(counts.values()) >= 32


def test_phase_detection_templates_are_split_by_construction() -> None:
    for split, languages in TEMPLATES.items():
        for language, phases in languages.items():
            assert set(phases) == set(LABELS), (split, language)
            assert all("{obj}" in template for rows in phases.values() for template in rows)
