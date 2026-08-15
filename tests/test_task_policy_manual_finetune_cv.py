"""Unit tests for manual-corpus E5 fine-tune cross-validation helpers."""

from __future__ import annotations

import numpy as np
import pytest

from bench.task_policy_external_review import ExternalReview
from bench.task_policy_manual_finetune_cv import (
    ManualFinetuneParameters,
    _consensus_predictions,
    _domain_checkpoint_key,
    _format_model_input,
    _hashed_lexical_features,
    _model_scores,
    _individual_accuracy_checkpoint_key,
    _lora_target_modules,
    _predict_cardinality,
    _positive_weights,
    _pool_hidden_state,
    _resolve_device,
    _select_accuracy_thresholds,
    _select_domain_accuracy_thresholds,
    _select_natural_accuracy_threshold,
    _validate_external_partition,
)


def test_finetune_parameters_default_to_shared_mean_architecture() -> None:
    parameters = ManualFinetuneParameters()

    assert parameters.architecture == "shared_mean"
    assert parameters.threshold_calibration == "independent"
    assert parameters.lora_rank == 0
    assert parameters.pooling == "mean"
    assert parameters.query_instruction is None
    assert parameters.load_in_4bit is False
    assert parameters.exclusive_loss_weight == pytest.approx(0.7)


def test_model_input_supports_plain_and_instruct_embedding_formats() -> None:
    assert _format_model_input("fix it", None) == "query: fix it"
    assert _format_model_input("fix it", "Classify this request") == (
        "Instruct: Classify this request\nQuery:fix it"
    )
    with pytest.raises(ValueError, match="must not be blank"):
        _format_model_input("fix it", "   ")


def test_hidden_state_pooling_supports_mean_and_last_token() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0], [50.0, 60.0]],
        [[70.0, 80.0], [5.0, 6.0], [7.0, 8.0]],
    ])
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]])

    mean = _pool_hidden_state(hidden, mask, "mean")
    last = _pool_hidden_state(hidden, mask, "last")

    assert torch.equal(mean, torch.tensor([[2.0, 3.0], [6.0, 7.0]]))
    assert torch.equal(last, torch.tensor([[3.0, 4.0], [7.0, 8.0]]))
    with pytest.raises(ValueError, match="pooling must be"):
        _pool_hidden_state(hidden, mask, "unknown")


def test_positive_weights_support_tempered_inverse_frequency() -> None:
    targets = np.asarray([
        [1, 1],
        [0, 1],
        [0, 1],
        [0, 0],
    ])

    unweighted = _positive_weights(targets, 0.0)
    tempered = _positive_weights(targets, 0.5)
    full = _positive_weights(targets, 1.0)

    assert unweighted.tolist() == pytest.approx([1.0, 1.0])
    assert tempered.tolist() == pytest.approx(np.sqrt(full).tolist())
    assert full.tolist() == pytest.approx([3.0, 1 / 3])


def test_positive_weights_reject_invalid_power() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        _positive_weights(np.asarray([[1]]), 1.1)


def test_checkpoint_key_prioritizes_the_worst_per_label_accuracy() -> None:
    def report(accuracies: list[float], exact_match: float) -> dict[str, object]:
        return {
            "per_label": {
                label: {"precision": 0.9, "recall": 0.8, "accuracy": accuracy}
                for label, accuracy in zip(
                    (
                        "bugfix.root_cause",
                        "feature.contract_first",
                        "refactor.preserve_behavior",
                        "research.evidence_first",
                        "review.read_only",
                        "performance.measure_first",
                    ),
                    accuracies,
                    strict=True,
                )
            },
            "exact_match": exact_match,
            "false_activations": 0,
        }

    individually_accurate = report([0.96] * 6, 0.7)
    aggregate_winner = report([1.0, 1.0, 1.0, 1.0, 1.0, 0.94], 0.99)

    assert _individual_accuracy_checkpoint_key(
        individually_accurate, accuracy_target=0.95
    ) > (
        _individual_accuracy_checkpoint_key(aggregate_winner, accuracy_target=0.95)
    )


def test_checkpoint_key_rejects_high_accuracy_with_zero_recall() -> None:
    from bench.task_policy_multilabel_head import METHOD_LABELS

    def report(*, accuracy: float, recall: float) -> dict[str, object]:
        return {
            "per_label": {
                label: {
                    "accuracy": accuracy,
                    "precision": float(recall > 0.0),
                    "recall": recall,
                }
                for label in METHOD_LABELS
            },
            "exact_match": accuracy,
        }

    useful = report(accuracy=0.94, recall=0.5)
    degenerate = report(accuracy=0.99, recall=0.0)

    assert _individual_accuracy_checkpoint_key(
        useful, accuracy_target=0.95
    ) > _individual_accuracy_checkpoint_key(
        degenerate, accuracy_target=0.95
    )


def test_accuracy_threshold_selection_uses_a_supported_precision_floor() -> None:
    from bench.task_policy_multilabel_head import MultiLabelExample

    rows = [
        MultiLabelExample("zero", "x", (), "validation"),
        MultiLabelExample("bug", "x", ("bugfix.root_cause",), "validation"),
    ]
    method_scores = np.asarray([
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        [0.9, 0.1, 0.1, 0.1, 0.1, 0.1],
    ])

    _, _, report, precision_floor = _select_accuracy_thresholds(
        method_scores,
        np.asarray([0.1, 0.9]),
        rows,
        accuracy_target=0.95,
    )

    assert precision_floor in {0.80, 0.85, 0.90, 0.95, 1.0}
    assert report["per_label"]["bugfix.root_cause"]["accuracy"] == 1.0


def test_domain_accuracy_thresholds_prefer_supported_natural_labels() -> None:
    from bench.task_policy_multilabel_head import MultiLabelExample

    synthetic = [
        MultiLabelExample("synthetic-zero", "x", (), "validation"),
        MultiLabelExample("synthetic-bug", "x", ("bugfix.root_cause",), "validation"),
    ]
    natural = [
        MultiLabelExample(f"natural-zero-{index}", "x", (), "calibration")
        for index in range(5)
    ] + [
        MultiLabelExample(
            f"natural-bug-{index}", "x", ("bugfix.root_cause",), "calibration"
        )
        for index in range(5)
    ]
    synthetic_scores = np.full((2, 6), 0.1)
    synthetic_scores[1, 0] = 0.6
    natural_scores = np.full((10, 6), 0.1)
    natural_scores[:5, 0] = 0.7
    natural_scores[5:, 0] = 0.9

    thresholds, task_threshold, _, _ = _select_domain_accuracy_thresholds(
        synthetic_scores,
        np.ones(2),
        synthetic,
        natural_scores,
        np.ones(10),
        natural,
        accuracy_target=0.95,
    )

    assert thresholds[0] > 0.7
    assert task_threshold == 0.0


def test_natural_threshold_fallback_keeps_supported_recall() -> None:
    expected = np.asarray([True] * 5 + [False] * 95)
    scores = np.asarray(
        [0.20, 0.21, 0.22, 0.23, 0.24]
        + [0.30] * 10
        + [0.10] * 85
    )

    threshold = _select_natural_accuracy_threshold(
        scores,
        expected,
        accuracy_target=0.95,
    )

    predicted = scores >= threshold
    assert int((predicted & expected).sum()) == 5
    assert threshold <= 0.20


def test_domain_checkpoint_key_uses_natural_metrics_only_with_support() -> None:
    from bench.task_policy_multilabel_head import METHOD_LABELS

    def report(*, natural: bool) -> dict[str, object]:
        per_label = {}
        for index, label in enumerate(METHOD_LABELS):
            per_label[label] = {
                "accuracy": 0.80 if natural and index == 0 else 0.99,
                "precision": 0.9,
                "recall": 0.9,
                "support": 5 if natural and index == 0 else (0 if natural else 20),
            }
        return {"per_label": per_label, "exact_match": 0.8}

    key = _domain_checkpoint_key(
        report(natural=False),
        report(natural=True),
        accuracy_target=0.95,
    )

    assert key[1] == pytest.approx(5 / 6)
    assert key[2] == pytest.approx(1.0)
    assert key[3] == pytest.approx(0.80)


def test_domain_checkpoint_key_rejects_supported_zero_recall() -> None:
    from bench.task_policy_multilabel_head import METHOD_LABELS

    def report(*, accuracy: float, recall: float) -> dict[str, object]:
        return {
            "per_label": {
                label: {
                    "accuracy": accuracy,
                    "precision": float(recall > 0.0),
                    "recall": recall,
                    "support": 7,
                }
                for label in METHOD_LABELS
            },
            "exact_match": accuracy,
        }

    synthetic = report(accuracy=0.9, recall=0.5)
    useful = report(accuracy=0.94, recall=0.5)
    degenerate = report(accuracy=0.99, recall=0.0)

    assert _domain_checkpoint_key(
        synthetic,
        useful,
        accuracy_target=0.95,
    ) > _domain_checkpoint_key(
        synthetic,
        degenerate,
        accuracy_target=0.95,
    )


def test_model_scores_batches_tokenized_inputs() -> None:
    torch = pytest.importorskip("torch")

    class Model:
        def eval(self) -> None:
            pass

        def __call__(self, *, input_ids: object) -> tuple[object, object]:
            values = input_ids.to(dtype=torch.float32)
            return values, values[:, 0]

    encoded = {"input_ids": torch.tensor([[0, 1], [2, 3], [4, 5]])}

    methods, task = _model_scores(Model(), encoded, batch_size=2)

    assert methods.shape == (3, 2)
    assert task.shape == (3,)
    assert methods[0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("module_names", "expected"),
    [
        (
            ("encoder.layer.0.attention.query", "encoder.layer.0.attention.value"),
            ("query", "value"),
        ),
        (
            ("layers.0.self_attn.q_proj", "layers.0.self_attn.v_proj"),
            ("q_proj", "v_proj"),
        ),
    ],
)
def test_lora_target_modules_supports_encoder_projection_conventions(
    module_names: tuple[str, ...],
    expected: tuple[str, str],
) -> None:
    class Encoder:
        def named_modules(self) -> list[tuple[str, object]]:
            return [(name, object()) for name in module_names]

    assert _lora_target_modules(Encoder()) == expected


def test_lora_target_modules_rejects_unknown_attention_projections() -> None:
    class Encoder:
        def named_modules(self) -> list[tuple[str, object]]:
            return [("layers.0.attention.unknown", object())]

    with pytest.raises(ValueError, match="unsupported encoder attention projections"):
        _lora_target_modules(Encoder())


def test_model_scores_moves_cpu_batches_to_cuda_and_returns_numpy() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class Model:
        def eval(self) -> None:
            pass

        def __call__(self, *, input_ids: object) -> tuple[object, object]:
            assert input_ids.device.type == "cuda"
            values = input_ids.to(dtype=torch.float32)
            return values, values[:, 0]

    encoded = {"input_ids": torch.tensor([[0, 1], [2, 3]])}

    methods, task = _model_scores(
        Model(),
        encoded,
        batch_size=1,
        device=torch.device("cuda"),
    )

    assert isinstance(methods, np.ndarray)
    assert isinstance(task, np.ndarray)
    assert methods.shape == (2, 2)
    assert task.shape == (2,)


def test_device_selection_defaults_to_cpu_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert str(_resolve_device("auto")) == "cpu"
    assert str(_resolve_device("cpu")) == "cpu"


def test_device_selection_accepts_available_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert str(_resolve_device("auto")) == "cuda"
    assert str(_resolve_device("cuda")) == "cuda"
    assert str(_resolve_device("cuda:1")) == "cuda:1"
    with pytest.raises(ValueError, match="index 2 is unavailable"):
        _resolve_device("cuda:2")


def test_device_selection_rejects_cuda_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match=r"torch\.cuda\.is_available\(\) is false"):
        _resolve_device("cuda")


def test_cardinality_prediction_can_abstain_and_select_top_labels() -> None:
    method_scores = np.asarray([
        [0.9, 0.8, 0.1, 0.2, 0.3, 0.4],
        [0.1, 0.2, 0.7, 0.9, 0.3, 0.8],
    ])
    cardinality_scores = np.asarray([
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 0.1, 0.9],
    ])

    predictions = _predict_cardinality(method_scores, cardinality_scores)

    assert predictions[0] == ()
    assert predictions[1] == (
        "research.evidence_first",
        "performance.measure_first",
        "refactor.preserve_behavior",
    )


def test_consensus_predictions_require_a_strict_majority() -> None:
    predictions = _consensus_predictions([
        [("bugfix.root_cause",), ("feature.contract_first",)],
        [("bugfix.root_cause",), ()],
        [("performance.measure_first",), ("feature.contract_first",)],
        [("bugfix.root_cause",), ()],
    ])

    assert predictions == [("bugfix.root_cause",), ()]


def test_consensus_predictions_reject_mismatched_model_outputs() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        _consensus_predictions([[()], [(), ()]])


def test_external_partition_rejects_repository_leakage() -> None:
    def review(candidate_id: str, repo: str) -> ExternalReview:
        return ExternalReview(candidate_id, repo, "python", "request", (), "reviewed")

    _validate_external_partition(
        [review("train", "owner/train")],
        [review("evaluation", "owner/evaluation")],
    )
    with pytest.raises(ValueError, match="owner/shared"):
        _validate_external_partition(
            [review("train", "owner/shared")],
            [review("evaluation", "owner/shared")],
        )


def test_hashed_lexical_features_are_deterministic_bounded_and_normalized() -> None:
    texts = ["Fix parser crash\nLong reproduction", "Añade soporte para TOML"]

    first = _hashed_lexical_features(texts, dimensions=64)
    second = _hashed_lexical_features(texts, dimensions=64)

    assert first.shape == (2, 64)
    assert np.array_equal(first, second)
    assert np.linalg.norm(first, axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert not np.array_equal(first[0], first[1])
