"""Dataset integrity tests for semantic runtime-behavior evaluation."""

from __future__ import annotations

from bench.behavior_semantic_eval import LABELS, build_behavior_corpus


def test_behavior_semantic_splits_are_large_unique_and_disjoint() -> None:
    splits = {
        name: build_behavior_corpus(name)
        for name in ("calibration", "validation", "holdout")
    }

    assert len(splits["calibration"]) >= 250
    assert len(splits["validation"]) >= 100
    assert len(splits["holdout"]) >= 100
    for examples in splits.values():
        assert len({item.id for item in examples}) == len(examples)
        assert len({item.text for item in examples}) == len(examples)
        assert {item.label for item in examples} == {*LABELS, None}
    texts = {name: {item.text for item in examples} for name, examples in splits.items()}
    assert texts["calibration"].isdisjoint(texts["validation"])
    assert texts["calibration"].isdisjoint(texts["holdout"])
    assert texts["validation"].isdisjoint(texts["holdout"])
