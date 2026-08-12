"""Natural behavior head split and metric tests."""

from __future__ import annotations

from bench.behavior_natural_head import NaturalExample, behavior_metrics, split_examples


def _example(identifier: str, family: str, label: str | None = None) -> NaturalExample:
    return NaturalExample(identifier, family, f"observable {identifier}", label)


def test_split_examples_keeps_project_families_disjoint() -> None:
    historical = [
        _example("cal", "requests", "healthy_progress"),
        _example("val", "ripgrep-15.1.0", "healthy_progress"),
    ]
    holdout = [_example("held", "jsmn")]

    calibration, validation, final_holdout = split_examples(historical, holdout)

    assert [item.id for item in calibration] == ["cal"]
    assert [item.id for item in validation] == ["val"]
    assert [item.id for item in final_holdout] == ["held"]


def test_behavior_metrics_counts_neutral_false_activations() -> None:
    examples = [
        _example("neutral", "jsmn"),
        _example("positive", "jsmn", "healthy_progress"),
    ]

    metrics = behavior_metrics(examples, ["excessive_exploration", "healthy_progress"])

    assert metrics["selective_precision"] == 0.5
    assert metrics["neutral_false_activation_rate"] == 1.0
    assert metrics["positive_macro_recall"] == 1.0
