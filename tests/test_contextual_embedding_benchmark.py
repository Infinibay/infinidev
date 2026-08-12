"""Unit tests for the optional contextual embedding benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from bench.contextual_embedding_benchmark import (
    DEFAULT_CONTEXTUAL_MODEL,
    TrainingParameters,
    _apply_cpu_attention_config,
    _cardinality_class_weights,
    _cardinality_targets,
    _format_model_inputs,
    _prediction_metrics,
    _train_frozen_head,
)
from bench.contextual_task_policy_finetune import _set_trainable_layers
from bench.task_policy_multilabel_head import MultiLabelExample


def test_contextual_benchmark_defaults_to_multilingual_e5_small() -> None:
    assert DEFAULT_CONTEXTUAL_MODEL == "intfloat/multilingual-e5-small"


def test_model_input_prefix_can_be_disabled_for_bge() -> None:
    assert _format_model_inputs(["Fix the race"], "query: ") == [
        "query: Fix the race"
    ]
    assert _format_model_inputs(["Fix the race"], "") == ["Fix the race"]


def test_portable_cpu_attention_disables_optional_xformers_paths() -> None:
    config = type("Config", (), {})()

    result = _apply_cpu_attention_config(config)

    assert result is config
    assert config.use_memory_efficient_attention is False
    assert config.unpad_inputs is False
    assert config._attn_implementation == "eager"


def test_cardinality_targets_preserve_three_policy_examples() -> None:
    examples = [
        MultiLabelExample("zero", "x", (), "calibration"),
        MultiLabelExample("one", "x", ("bugfix.root_cause",), "calibration"),
        MultiLabelExample(
            "three",
            "x",
            (
                "bugfix.root_cause",
                "refactor.preserve_behavior",
                "performance.measure_first",
            ),
            "calibration",
        ),
    ]

    assert _cardinality_targets(examples).tolist() == [0, 1, 3]


def test_cardinality_class_weights_balance_total_mass() -> None:
    targets = np.asarray([0, 1, 1, 1, 2, 2, 3])

    weights = _cardinality_class_weights(targets)
    weighted_mass = np.bincount(targets) * weights

    assert weighted_mass.tolist() == pytest.approx([1.75] * 4)


def test_cardinality_class_weights_require_every_supported_class() -> None:
    with pytest.raises(ValueError, match="every supported cardinality"):
        _cardinality_class_weights(np.asarray([0, 1, 1, 2]))


def test_cardinality_class_weights_support_partial_balancing() -> None:
    targets = np.asarray([0, 1, 1, 1, 2, 2, 3])

    unweighted = _cardinality_class_weights(targets, power=0.0)
    partial = _cardinality_class_weights(targets, power=0.5)
    full = _cardinality_class_weights(targets, power=1.0)

    assert unweighted.tolist() == pytest.approx([1.0] * 4)
    assert partial.tolist() == pytest.approx(np.sqrt(full).tolist())


def test_cardinality_class_weights_reject_invalid_power() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _cardinality_class_weights(np.asarray([0, 1, 2, 3]), power=1.1)


def test_frozen_head_uses_configured_cardinality_balance_power() -> None:
    pytest.importorskip("torch")
    examples = [
        MultiLabelExample("zero", "x", (), "calibration"),
        MultiLabelExample("one", "x", ("bugfix.root_cause",), "calibration"),
        MultiLabelExample(
            "two",
            "x",
            ("bugfix.root_cause", "refactor.preserve_behavior"),
            "calibration",
        ),
        MultiLabelExample(
            "three",
            "x",
            (
                "bugfix.root_cause",
                "refactor.preserve_behavior",
                "performance.measure_first",
            ),
            "calibration",
        ),
    ]
    vectors = np.eye(4, dtype=np.float32)

    _, selection = _train_frozen_head(
        vectors,
        examples,
        vectors,
        examples,
        TrainingParameters(
            hidden_size=4,
            batch_size=4,
            max_epochs=0,
            evaluate_every=1,
            patience_evaluations=1,
            cardinality_balance_power=0.5,
        ),
    )

    assert selection["best_epoch"] == 0


def test_prediction_metrics_require_exact_multilabel_sets() -> None:
    examples = [
        MultiLabelExample("neutral", "thanks", (), "holdout"),
        MultiLabelExample(
            "single", "repair it", ("bugfix.root_cause",), "holdout"
        ),
        MultiLabelExample(
            "compound",
            "repair and simplify",
            ("bugfix.root_cause", "refactor.preserve_behavior"),
            "holdout",
        ),
    ]

    perfect = _prediction_metrics(examples, [(), ("bugfix.root_cause",), (
        "refactor.preserve_behavior", "bugfix.root_cause",
    )])
    missing_label = _prediction_metrics(
        examples, [(), ("bugfix.root_cause",), ("bugfix.root_cause",)]
    )

    assert perfect["exact_match"] == 1.0
    assert perfect["micro_precision"] == 1.0
    assert perfect["micro_recall"] == 1.0
    assert missing_label["exact_match"] == 2 / 3
    assert missing_label["micro_precision"] == 1.0
    assert missing_label["micro_recall"] < 1.0


def test_finetune_unfreezes_only_requested_final_encoder_layers() -> None:
    class Parameter:
        def __init__(self, size: int) -> None:
            self.size = size
            self.requires_grad = True

        def numel(self) -> int:
            return self.size

    class Layer:
        def __init__(self, size: int) -> None:
            self.parameter = Parameter(size)

        def parameters(self) -> list[Parameter]:
            return [self.parameter]

    class Encoder:
        def __init__(self) -> None:
            self.encoder = type("Stack", (), {})()
            self.encoder.layer = [Layer(10), Layer(20), Layer(30)]
            self.embedding = Parameter(40)

        def parameters(self) -> list[Parameter]:
            return [
                self.embedding,
                *(layer.parameter for layer in self.encoder.layer),
            ]

    encoder = Encoder()

    trainable, total = _set_trainable_layers(encoder, unfrozen_layers=2)

    assert (trainable, total) == (50, 100)
    assert not encoder.embedding.requires_grad
    assert not encoder.encoder.layer[0].parameter.requires_grad
    assert encoder.encoder.layer[1].parameter.requires_grad
    assert encoder.encoder.layer[2].parameter.requires_grad


def test_trainable_layers_can_freeze_the_complete_encoder() -> None:
    class Parameter:
        requires_grad = True

        def numel(self) -> int:
            return 1

    class Layer:
        def __init__(self) -> None:
            self.parameter = Parameter()

        def parameters(self) -> list[Parameter]:
            return [self.parameter]

    class Encoder:
        def __init__(self) -> None:
            self.encoder = type("Stack", (), {"layer": [Layer(), Layer()]})()

        def parameters(self) -> list[Parameter]:
            return [layer.parameter for layer in self.encoder.layer]

    encoder = Encoder()

    trainable, total = _set_trainable_layers(encoder, unfrozen_layers=0)

    assert (trainable, total) == (0, 2)
    assert not any(parameter.requires_grad for parameter in encoder.parameters())
