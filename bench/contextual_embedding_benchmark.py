"""Compare frozen contextual and static embeddings for multi-label task routing.

This benchmark is intentionally optional. It imports PyTorch and Transformers
only inside the contextual backend and never changes the packaged runtime.
Downloaded model caches remain outside the repository.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import resource
import statistics
import time
from typing import Protocol

import numpy as np

from bench.task_policy_hierarchical_head import METHOD_LABELS
from bench.task_policy_multilabel_head import (
    MultiLabelExample,
    _targets,
    build_multilabel_corpus,
    metrics,
)
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


DEFAULT_CONTEXTUAL_MODEL = "intfloat/multilingual-e5-small"
BENCHMARK_VERSION = "contextual-embedding-task-policy-v1"
MAX_POLICY_CARDINALITY = 3


class _Head(Protocol):
    def state_dict(self) -> dict[str, object]: ...


@dataclass(frozen=True)
class BackendMeasurements:
    """Embedding runtime and memory observations from one process."""

    backend: str
    model: str
    dimensions: int
    load_seconds: float
    corpus_seconds: float
    corpus_examples_per_second: float
    warm_single_p50_ms: float
    warm_single_p95_ms: float
    peak_rss_delta_mib: float


@dataclass(frozen=True)
class TrainingParameters:
    """Frozen-head training parameters fixed before holdout evaluation."""

    hidden_size: int = 128
    batch_size: int = 256
    learning_rate: float = 0.002
    weight_decay: float = 0.001
    cardinality_loss_weight: float = 0.7
    cardinality_balance_power: float = 0.0
    minimum_method_precision: float = 0.85
    max_epochs: int = 250
    evaluate_every: int = 5
    patience_evaluations: int = 12
    seed: int = 17


def _format_model_inputs(texts: list[str], prefix: str) -> list[str]:
    """Apply the model-specific input convention without guessing by family."""
    return [f"{prefix}{text}" for text in texts]


def _apply_cpu_attention_config(config: object) -> object:
    """Disable optional xformers/unpadding paths for portable CPU inference."""
    config.use_memory_efficient_attention = False
    config.unpad_inputs = False
    config._attn_implementation = "eager"
    return config


def _cardinality_targets(examples: list[MultiLabelExample]) -> np.ndarray:
    """Return exact supported cardinalities instead of collapsing triples."""
    return np.asarray(
        [min(MAX_POLICY_CARDINALITY, len(item.policies)) for item in examples],
        dtype=np.int64,
    )


def _cardinality_class_weights(
    targets: np.ndarray,
    *,
    power: float = 1.0,
) -> np.ndarray:
    """Balance cardinality classes so abundant single-label rows do not dominate."""
    if not 0.0 <= power <= 1.0:
        raise ValueError("cardinality balance power must be between 0 and 1")
    counts = np.bincount(targets, minlength=MAX_POLICY_CARDINALITY + 1)
    if np.any(counts == 0):
        raise ValueError("every supported cardinality needs at least one example")
    inverse = len(targets) / ((MAX_POLICY_CARDINALITY + 1) * counts)
    return np.power(inverse, power)


def _rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / 1024.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1)
    return ordered[position]


def _encode_static(texts: list[str]) -> tuple[np.ndarray, BackendMeasurements]:
    rss_before = _rss_mib()
    load_start = time.perf_counter()
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    _ = embedder.dim
    load_seconds = time.perf_counter() - load_start
    corpus_start = time.perf_counter()
    vectors = np.asarray(embedder.embed_queries(texts), dtype=np.float32)
    corpus_seconds = time.perf_counter() - corpus_start
    timings = []
    for text in texts[:30]:
        started = time.perf_counter()
        embedder.embed_query(text)
        timings.append((time.perf_counter() - started) * 1000.0)
    return vectors, BackendMeasurements(
        backend="static",
        model=embedder.model_name,
        dimensions=embedder.dim,
        load_seconds=load_seconds,
        corpus_seconds=corpus_seconds,
        corpus_examples_per_second=len(texts) / max(corpus_seconds, 1e-9),
        warm_single_p50_ms=statistics.median(timings),
        warm_single_p95_ms=_percentile(timings, 0.95),
        peak_rss_delta_mib=max(0.0, _rss_mib() - rss_before),
    )


def _encode_contextual(
    texts: list[str],
    *,
    model_name: str,
    batch_size: int,
    max_length: int,
    input_prefix: str = "query: ",
    trust_remote_code: bool = False,
    model_revision: str | None = None,
    portable_cpu_attention: bool = False,
) -> tuple[np.ndarray, BackendMeasurements]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoConfig, AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "contextual benchmark requires the optional finetune dependencies"
        ) from exc

    rss_before = _rss_mib()
    load_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        revision=model_revision,
    )
    model_kwargs: dict[str, object] = {
        "trust_remote_code": trust_remote_code,
        "revision": model_revision,
    }
    if portable_cpu_attention:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=model_revision,
        )
        model_kwargs["config"] = _apply_cpu_attention_config(config)
    model = AutoModel.from_pretrained(model_name, **model_kwargs)
    model.eval()
    model.to("cpu")
    load_seconds = time.perf_counter() - load_start

    def encode(batch: list[str]) -> np.ndarray:
        prefixed = _format_model_inputs(batch, input_prefix)
        encoded = tokenizer(
            prefixed,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            output = model(**encoded).last_hidden_state
            mask = encoded["attention_mask"][..., None].bool()
            pooled = output.masked_fill(~mask, 0.0).sum(dim=1)
            pooled = pooled / mask.sum(dim=1).clamp(min=1)
            pooled = functional.normalize(pooled, p=2, dim=1)
        return pooled.cpu().numpy().astype(np.float32, copy=False)

    encode(texts[:1])
    corpus_start = time.perf_counter()
    vectors = np.concatenate([
        encode(texts[start:start + batch_size])
        for start in range(0, len(texts), batch_size)
    ])
    corpus_seconds = time.perf_counter() - corpus_start
    timings = []
    for text in texts[:30]:
        started = time.perf_counter()
        encode([text])
        timings.append((time.perf_counter() - started) * 1000.0)
    return vectors, BackendMeasurements(
        backend="contextual",
        model=model_name,
        dimensions=int(vectors.shape[1]),
        load_seconds=load_seconds,
        corpus_seconds=corpus_seconds,
        corpus_examples_per_second=len(texts) / max(corpus_seconds, 1e-9),
        warm_single_p50_ms=statistics.median(timings),
        warm_single_p95_ms=_percentile(timings, 0.95),
        peak_rss_delta_mib=max(0.0, _rss_mib() - rss_before),
    )


def _prediction_metrics(
    examples: list[MultiLabelExample],
    predictions: list[tuple[str, ...]],
) -> dict[str, object]:
    report = metrics(examples, predictions)
    expected = _targets(examples).astype(bool)
    predicted = np.zeros_like(expected)
    lookup = {label: index for index, label in enumerate(METHOD_LABELS)}
    for row, labels in enumerate(predictions):
        for label in labels:
            predicted[row, lookup[label]] = True
    per_label: dict[str, dict[str, float | int]] = {}
    f1_values = []
    for index, label in enumerate(METHOD_LABELS):
        true_positive = int(np.sum(predicted[:, index] & expected[:, index]))
        false_positive = int(np.sum(predicted[:, index] & ~expected[:, index]))
        false_negative = int(np.sum(~predicted[:, index] & expected[:, index]))
        true_negative = int(np.sum(~predicted[:, index] & ~expected[:, index]))
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        accuracy = (true_positive + true_negative) / len(examples)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        f1_values.append(f1)
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "f1": f1,
            "support": int(np.sum(expected[:, index])),
        }
    errors = list(report.pop("errors"))
    return {
        **report,
        "macro_f1": statistics.mean(f1_values),
        "per_label": per_label,
        "error_count": len(errors),
        "error_sample": errors[:20],
    }


def _train_frozen_head(
    calibration_vectors: np.ndarray,
    calibration: list[MultiLabelExample],
    validation_vectors: np.ndarray,
    validation: list[MultiLabelExample],
    parameters: TrainingParameters,
) -> tuple[_Head, dict[str, object]]:
    import torch
    from torch import nn

    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))

    class FrozenEmbeddingHead(nn.Module):
        def __init__(self, dimensions: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(dimensions, parameters.hidden_size),
                nn.LayerNorm(parameters.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.methods = nn.Linear(parameters.hidden_size, len(METHOD_LABELS))
            self.cardinality = nn.Linear(
                parameters.hidden_size, MAX_POLICY_CARDINALITY + 1
            )

        def forward(self, vectors: object) -> tuple[object, object]:
            hidden = self.body(vectors)
            return self.methods(hidden), self.cardinality(hidden)

    calibration_x = torch.from_numpy(calibration_vectors)
    validation_x = torch.from_numpy(validation_vectors)
    calibration_y = torch.from_numpy(_targets(calibration).astype(np.float32))
    validation_y = torch.from_numpy(_targets(validation).astype(np.float32))
    calibration_cardinality = torch.tensor(
        _cardinality_targets(calibration), dtype=torch.long
    )
    validation_cardinality = torch.tensor(
        _cardinality_targets(validation), dtype=torch.long
    )
    positive_counts = calibration_y.sum(dim=0)
    positive_weights = (len(calibration) - positive_counts) / positive_counts.clamp(min=1)
    method_loss = nn.BCEWithLogitsLoss(pos_weight=positive_weights)
    cardinality_loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(
            _cardinality_class_weights(
                calibration_cardinality.numpy(),
                power=parameters.cardinality_balance_power,
            )
        ).to(dtype=torch.float32)
    )
    head = FrozenEmbeddingHead(calibration_vectors.shape[1])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=parameters.learning_rate,
        weight_decay=parameters.weight_decay,
    )
    best_key = (-1.0, -1.0, -1.0)
    best_state: dict[str, object] | None = None
    best_epoch = 0
    stale_evaluations = 0
    for epoch in range(parameters.max_epochs + 1):
        head.train()
        order = torch.randperm(len(calibration))
        for start in range(0, len(calibration), parameters.batch_size):
            indices = order[start:start + parameters.batch_size]
            method_logits, cardinality_logits = head(calibration_x[indices])
            loss = method_loss(method_logits, calibration_y[indices])
            loss = loss + parameters.cardinality_loss_weight * cardinality_loss(
                cardinality_logits, calibration_cardinality[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % parameters.evaluate_every:
            continue
        predictions = _predict_head(head, validation_x)
        report = _prediction_metrics(validation, predictions)
        key = (
            float(report["exact_match"]),
            float(report["micro_precision"]),
            float(report["micro_recall"]),
        )
        if key > best_key:
            best_key = key
            best_state = {
                name: value.detach().clone()
                for name, value in head.state_dict().items()
            }
            best_epoch = epoch
            stale_evaluations = 0
        else:
            stale_evaluations += 1
            if stale_evaluations >= parameters.patience_evaluations:
                break
    if best_state is None:
        raise RuntimeError("frozen embedding head produced no checkpoint")
    head.load_state_dict(best_state)
    return head, {
        "best_epoch": best_epoch,
        "validation": _prediction_metrics(validation, _predict_head(head, validation_x)),
    }


def _predict_head(head: _Head, vectors: object) -> list[tuple[str, ...]]:
    import torch

    head.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        method_logits, cardinality_logits = head(vectors)  # type: ignore[operator]
    cardinalities = cardinality_logits.argmax(dim=1)
    predictions = []
    for row, cardinality in zip(method_logits, cardinalities.tolist(), strict=True):
        if cardinality == 0:
            predictions.append(())
            continue
        indices = row.topk(k=min(int(cardinality), len(METHOD_LABELS))).indices.tolist()
        predictions.append(tuple(METHOD_LABELS[index] for index in indices))
    return predictions


def run_benchmark(
    *,
    backend: str,
    model_name: str = DEFAULT_CONTEXTUAL_MODEL,
    encode_batch_size: int = 64,
    max_length: int = 128,
    training: TrainingParameters | None = None,
) -> dict[str, object]:
    """Run one frozen-embedding benchmark without changing runtime artifacts."""
    import torch

    training = training or TrainingParameters()
    calibration = build_multilabel_corpus("calibration")
    validation = build_multilabel_corpus("validation")
    holdout = build_multilabel_corpus("holdout")
    examples = calibration + validation + holdout
    texts = [item.text for item in examples]
    if backend == "static":
        vectors, measurements = _encode_static(texts)
    elif backend == "contextual":
        vectors, measurements = _encode_contextual(
            texts,
            model_name=model_name,
            batch_size=encode_batch_size,
            max_length=max_length,
        )
    else:
        raise ValueError(f"unknown embedding backend: {backend}")
    calibration_end = len(calibration)
    validation_end = calibration_end + len(validation)
    head, selection = _train_frozen_head(
        vectors[:calibration_end],
        calibration,
        vectors[calibration_end:validation_end],
        validation,
        training,
    )
    holdout_vectors = torch.from_numpy(vectors[validation_end:])
    holdout_report = _prediction_metrics(
        holdout,
        _predict_head(head, holdout_vectors),
    )
    passes_quality_gate = (
        holdout_report["exact_match"] > 0.95
        and holdout_report["macro_f1"] > 0.95
        and holdout_report["micro_precision"] > 0.95
        and holdout_report["micro_recall"] > 0.95
        and holdout_report["false_activations"] == 0
    )
    return {
        "version": BENCHMARK_VERSION,
        "backend": asdict(measurements),
        "training": asdict(training),
        "examples": {
            "calibration": len(calibration),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "best_epoch": selection["best_epoch"],
        "validation": selection["validation"],
        "holdout": holdout_report,
        "quality_gate": {
            "exact_match_gt": 0.95,
            "macro_f1_gt": 0.95,
            "micro_precision_gt": 0.95,
            "micro_recall_gt": 0.95,
            "false_activations": 0,
            "passed": passes_quality_gate,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("static", "contextual"), required=True)
    parser.add_argument("--model", default=DEFAULT_CONTEXTUAL_MODEL)
    parser.add_argument("--encode-batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(
        backend=args.backend,
        model_name=args.model,
        encode_batch_size=args.encode_batch_size,
        max_length=args.max_length,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()


__all__ = [
    "BENCHMARK_VERSION",
    "DEFAULT_CONTEXTUAL_MODEL",
    "MAX_POLICY_CARDINALITY",
    "BackendMeasurements",
    "TrainingParameters",
    "_cardinality_targets",
    "_cardinality_class_weights",
    "run_benchmark",
]
