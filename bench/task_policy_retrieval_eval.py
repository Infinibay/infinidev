"""Evaluate retrieval and prototype task-policy classifiers on human data.

The benchmark embeds requests once, selects every score formulation and
threshold on human calibration, then opens the development evaluation split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np

from bench.task_policy_gliclass_finetune import _binary_metrics, _load_partition, _targets
from bench.task_policy_multilabel_head import METHOD_LABELS
from bench.task_policy_pairwise_finetune import _select_threshold


ACCURACY_TARGET = 0.95
RECALL_TARGET = 0.95


def _mean_pool(hidden: object, attention_mask: object) -> object:
    mask = attention_mask[..., None].bool()
    return hidden.masked_fill(~mask, 0.0).sum(dim=1) / mask.sum(dim=1).clamp(min=1)


def _embed(
    model: object,
    tokenizer: object,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
    device: object,
) -> np.ndarray:
    import torch

    chunks = []
    model.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = tokenizer(
                ["query: " + text for text in texts[start:start + batch_size]],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            batch = {key: value.to(device) for key, value in batch.items()}
            hidden = model(**batch).last_hidden_state  # type: ignore[operator]
            pooled = _mean_pool(hidden, batch["attention_mask"])
            chunks.append(torch.nn.functional.normalize(pooled.float(), dim=-1).cpu().numpy())
    return np.concatenate(chunks)


def _weighted_knn_scores(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    query_vectors: np.ndarray,
    *,
    neighbors: int,
    temperature: float,
) -> np.ndarray:
    """Return similarity-weighted multi-label votes from nearest examples."""
    similarities = query_vectors @ train_vectors.T
    neighbors = min(neighbors, len(train_vectors))
    indices = np.argpartition(similarities, -neighbors, axis=1)[:, -neighbors:]
    selected = np.take_along_axis(similarities, indices, axis=1)
    weights = np.exp((selected - selected.max(axis=1, keepdims=True)) / temperature)
    labels = train_targets[indices]
    return (labels * weights[..., None]).sum(axis=1) / weights.sum(axis=1, keepdims=True)


def _margin_scores(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    query_vectors: np.ndarray,
    *,
    neighbors: int,
) -> np.ndarray:
    """Contrast each label's closest positive and negative examples."""
    similarities = query_vectors @ train_vectors.T
    columns = []
    for index in range(train_targets.shape[1]):
        positive = similarities[:, train_targets[:, index].astype(bool)]
        negative = similarities[:, ~train_targets[:, index].astype(bool)]
        positive_k = min(neighbors, positive.shape[1])
        negative_k = min(neighbors, negative.shape[1])
        positive_top = np.partition(positive, -positive_k, axis=1)[:, -positive_k:]
        negative_top = np.partition(negative, -negative_k, axis=1)[:, -negative_k:]
        columns.append(positive_top.mean(axis=1) - negative_top.mean(axis=1))
    return np.stack(columns, axis=1)


def _prototype_scores(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    query_vectors: np.ndarray,
) -> np.ndarray:
    """Contrast normalized positive and negative centroids per label."""
    columns = []
    for index in range(train_targets.shape[1]):
        positive = train_vectors[train_targets[:, index].astype(bool)].mean(axis=0)
        negative = train_vectors[~train_targets[:, index].astype(bool)].mean(axis=0)
        positive /= max(float(np.linalg.norm(positive)), 1e-12)
        negative /= max(float(np.linalg.norm(negative)), 1e-12)
        columns.append(query_vectors @ positive - query_vectors @ negative)
    return np.stack(columns, axis=1)


def _score_candidates(
    train_vectors: np.ndarray,
    train_targets: np.ndarray,
    query_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    candidates = {"prototype": _prototype_scores(train_vectors, train_targets, query_vectors)}
    for neighbors in (1, 3, 5, 9, 17, 33, 65):
        for temperature in (0.02, 0.05, 0.10):
            name = f"knn-k{neighbors}-t{temperature:.2f}"
            candidates[name] = _weighted_knn_scores(
                train_vectors,
                train_targets,
                query_vectors,
                neighbors=neighbors,
                temperature=temperature,
            )
    for neighbors in (1, 3, 5, 10, 20):
        candidates[f"margin-k{neighbors}"] = _margin_scores(
            train_vectors,
            train_targets,
            query_vectors,
            neighbors=neighbors,
        )
    return candidates


def _metric_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    accuracy = float(metrics["accuracy"])
    recall = float(metrics["recall"])
    return (
        float(accuracy >= ACCURACY_TARGET and recall >= RECALL_TARGET),
        min(accuracy / ACCURACY_TARGET, recall / RECALL_TARGET),
        min(accuracy, recall),
        float(metrics["f1"]),
        float(metrics["precision"]),
    )


def select_formulations(
    candidates: dict[str, np.ndarray],
    expected: np.ndarray,
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, Any]]:
    """Select score formulation and threshold independently for each label."""
    names = []
    thresholds = []
    diagnostics = {}
    for index, label in enumerate(METHOD_LABELS):
        best: tuple[tuple[float, ...], str, float, dict[str, float | int]] | None = None
        for name, scores in candidates.items():
            threshold = _select_threshold(
                scores[:, index],
                expected[:, index],
                accuracy_target=ACCURACY_TARGET,
                recall_target=RECALL_TARGET,
            )
            metrics = _binary_metrics(scores[:, index], expected[:, index], threshold)
            candidate = (_metric_key(metrics), name, threshold, metrics)
            if best is None or candidate[0] > best[0]:
                best = candidate
        if best is None:
            raise RuntimeError(f"no retrieval formulation selected for {label}")
        _, name, threshold, metrics = best
        names.append(name)
        thresholds.append(threshold)
        diagnostics[label] = {"formulation": name, "threshold": threshold, "metrics": metrics}
    return tuple(names), tuple(thresholds), diagnostics


def _selected_scores(
    candidates: dict[str, np.ndarray],
    names: tuple[str, ...],
) -> np.ndarray:
    return np.stack([candidates[name][:, index] for index, name in enumerate(names)], axis=1)


def _report(
    scores: np.ndarray,
    expected: np.ndarray,
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    predicted = scores >= np.asarray(thresholds)[None, :]
    per_label = {
        label: _binary_metrics(scores[:, index], expected[:, index], thresholds[index])
        for index, label in enumerate(METHOD_LABELS)
    }
    return {
        "examples": len(expected),
        "exact_match": float(np.mean(np.all(predicted == expected, axis=1))),
        "micro_accuracy": float(np.mean(predicted == expected)),
        "per_label": per_label,
        "gate": {
            "accuracy_target": ACCURACY_TARGET,
            "recall_target": RECALL_TARGET,
            "all_labels_pass": all(_metric_key(metrics)[0] for metrics in per_label.values()),
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="intfloat/multilingual-e5-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--latency-samples", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    import torch
    from transformers import AutoModel, AutoTokenizer

    requested_device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if requested_device == "auto":
        requested_device = "cpu"
    device = torch.device(requested_device)
    training = _load_partition(args.data_root, "training")
    calibration = _load_partition(args.data_root, "calibration")
    evaluation = _load_partition(args.data_root, "evaluation")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device)
    vectors = {}
    started = time.perf_counter()
    for name, rows in (
        ("training", training),
        ("calibration", calibration),
        ("evaluation", evaluation),
    ):
        vectors[name] = _embed(
            model,
            tokenizer,
            [row.text for row in rows],
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
        )
        print(json.dumps({"embedded": name, "rows": len(rows)}), flush=True)
    embedding_seconds = time.perf_counter() - started
    calibration_candidates = _score_candidates(
        vectors["training"], _targets(training), vectors["calibration"],
    )
    names, thresholds, selection = select_formulations(
        calibration_candidates, _targets(calibration),
    )
    calibration_scores = _selected_scores(calibration_candidates, names)
    evaluation_candidates = _score_candidates(
        vectors["training"], _targets(training), vectors["evaluation"],
    )
    evaluation_scores = _selected_scores(evaluation_candidates, names)
    timings = []
    for index in range(args.latency_samples + 5):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        vector = _embed(
            model,
            tokenizer,
            [evaluation[index % len(evaluation)].text],
            batch_size=1,
            max_length=args.max_length,
            device=device,
        )
        _score_candidates(vectors["training"], _targets(training), vector)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if index >= 5:
            timings.append((time.perf_counter() - started) * 1000)
    report = {
        "model": args.model,
        "device": str(device),
        "counts": {
            "training": len(training),
            "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        "embedding_seconds": embedding_seconds,
        "selection": selection,
        "calibration": _report(calibration_scores, _targets(calibration), thresholds),
        "evaluation": _report(evaluation_scores, _targets(evaluation), thresholds),
        "latency": {
            "samples": len(timings),
            "p50_ms": statistics.median(timings),
            "p95_ms": float(np.percentile(timings, 95)),
            "max_ms": max(timings),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "evaluation_exact_match": report["evaluation"]["exact_match"],
        "evaluation_gate": report["evaluation"]["gate"],
        "latency": report["latency"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
