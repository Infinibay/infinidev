from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from bench.task_policy_encoder_finetune import (
    _balanced_checkpoint_key,
    _build_model,
    _early_stopping_reached,
    _example_weights,
    _last_pool,
    _require_minimum_positive_support,
    _select_balanced_threshold,
)


def test_example_weights_keep_human_full_and_scale_model_confidence() -> None:
    rows = [
        SimpleNamespace(annotation_kind="human", annotation_confidence=0.2),
        SimpleNamespace(annotation_kind="model", annotation_confidence=0.8),
    ]

    weights = _example_weights(rows, 0.5)

    np.testing.assert_allclose(weights, np.asarray([1.0, 0.4], dtype=np.float32))


@pytest.mark.parametrize("weight", [0.0, -0.1, 1.1])
def test_example_weights_reject_invalid_global_weight(weight: float) -> None:
    with pytest.raises(ValueError, match="model_label_weight"):
        _example_weights([], weight)


def test_last_pool_handles_left_and_right_padding() -> None:
    torch = pytest.importorskip("torch")
    hidden = torch.tensor([
        [[1.0], [2.0], [3.0], [99.0]],
        [[99.0], [4.0], [5.0], [6.0]],
    ])
    attention_mask = torch.tensor([
        [1, 1, 1, 0],
        [0, 1, 1, 1],
    ])

    pooled = _last_pool(hidden, attention_mask)

    torch.testing.assert_close(pooled, torch.tensor([[3.0], [6.0]]))


def test_last_pool_rejects_empty_attention_rows() -> None:
    torch = pytest.importorskip("torch")

    with pytest.raises(ValueError, match="no valid tokens"):
        _last_pool(torch.zeros((1, 2, 3)), torch.zeros((1, 2), dtype=torch.long))


def test_early_stopping_counts_complete_non_improving_epochs() -> None:
    assert not _early_stopping_reached(epoch=3, best_epoch=2, patience=2)
    assert _early_stopping_reached(epoch=4, best_epoch=2, patience=2)
    assert not _early_stopping_reached(epoch=20, best_epoch=1, patience=0)



def test_training_support_gate_requires_every_category() -> None:
    all_labels = (
        "bugfix.root_cause", "feature.contract_first", "performance.measure_first",
        "refactor.preserve_behavior", "research.evidence_first", "review.read_only",
    )
    rows = [SimpleNamespace(policies=all_labels) for _ in range(3)]

    assert _require_minimum_positive_support(rows, 3) == {
        label: 3 for label in all_labels
    }


def test_training_support_gate_reports_under_supported_categories() -> None:
    rows = [SimpleNamespace(policies=("bugfix.root_cause",)) for _ in range(2)]

    with pytest.raises(ValueError, match=r"at least 2.*feature\.contract_first=0"):
        _require_minimum_positive_support(rows, 2)

    with pytest.raises(ValueError, match="must not be negative"):
        _require_minimum_positive_support(rows, -1)

def test_balanced_threshold_does_not_prefer_all_negative_accuracy() -> None:
    scores = np.asarray([0.9, 0.8, 0.7, 0.2, 0.1])
    expected = np.asarray([1, 1, 0, 0, 0])

    threshold = _select_balanced_threshold(
        scores,
        expected,
        minimum_precision=0.8,
        minimum_recall=0.5,
    )

    assert 0.7 < threshold <= 0.8


def test_checkpoint_key_prefers_better_worst_label_over_exact_match() -> None:
    def report(weak_f1: float, exact_match: float) -> dict:
        per_label = {
            label: {"accuracy": 0.96, "precision": 0.9, "recall": 0.8, "f1": 0.85}
            for label in (
                "bugfix.root_cause",
                "feature.contract_first",
                "performance.measure_first",
                "refactor.preserve_behavior",
                "research.evidence_first",
                "review.read_only",
            )
        }
        per_label["research.evidence_first"] = {
            "accuracy": 0.96,
            "precision": 0.9,
            "recall": weak_f1,
            "f1": weak_f1,
        }
        return {"per_label": per_label, "exact_match": exact_match}

    stronger = _balanced_checkpoint_key(
        report(0.7, 0.8), minimum_precision=0.85, minimum_recall=0.5,
    )
    weaker = _balanced_checkpoint_key(
        report(0.6, 0.9), minimum_precision=0.85, minimum_recall=0.5,
    )

    assert stronger > weaker


def test_query2label_produces_independent_label_logits(monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    class TinyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(hidden_size=16, use_cache=True)
            self.embedding = torch.nn.Embedding(32, 16)

        def resize_token_embeddings(self, _size: int) -> None:
            return None

        def forward(self, input_ids, **_kwargs):
            return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

    monkeypatch.setattr(
        transformers.AutoModel,
        "from_pretrained",
        lambda _model_name: TinyEncoder(),
    )
    model = _build_model(
        "unused",
        architecture="query2label",
        label_queries=torch.randn(6, 16),
        tokenizer_size=32,
    )
    model.eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
    }

    with torch.inference_mode():
        methods, task = model(**batch)

    assert methods.shape == (2, 6)
    assert task.shape == (2,)
    assert torch.isfinite(methods).all()
