"""Train and evaluate a selective multi-label task-method mini-head."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from bench.task_policy_compound_corpus import build_compound_corpus
from bench.task_policy_discourse_corpus import build_discourse_corpus
from bench.task_policy_hierarchical_head import METHOD_LABELS
from bench.task_policy_linear_head import (
    _build_development_corpus,
    _design_matrix,
    _fit_ridge,
    _materialize_calibration,
    build_ambiguity_challenge,
    build_linear_head_holdout,
)
from bench.task_policy_semantic_eval import build_semantic_validation_corpus
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


MODEL_VERSION = "static-qwen3-task-policy-multilabel-head-v5"


@dataclass(frozen=True)
class MultiLabelExample:
    """One request with zero, one, or several compatible methods."""

    id: str
    text: str
    policies: tuple[str, ...]
    split: str


@dataclass(frozen=True)
class HeadParameters:
    """Thresholds selected without observing the retained holdout."""

    discourse_ridge: float
    method_ridge: float
    discourse_threshold: float
    method_thresholds: tuple[float, ...]
    candidate_thresholds: tuple[float, ...]


def _single_examples(rows: list[object], split: str) -> list[MultiLabelExample]:
    return [
        MultiLabelExample(
            id=str(row.id),
            text=str(row.text),
            policies=() if row.policy is None else (str(row.policy),),
            split=split,
        )
        for row in rows
    ]


def build_multilabel_corpus(split: str) -> list[MultiLabelExample]:
    """Build leakage-separated zero/one/many-label data."""
    if split == "calibration":
        singles = _single_examples(_materialize_calibration(), split)
    elif split == "validation":
        singles = _single_examples(
            build_semantic_validation_corpus()
            + build_ambiguity_challenge()
            + _build_development_corpus(),
            split,
        )
    elif split == "holdout":
        singles = _single_examples(build_linear_head_holdout(), split)
    else:
        raise ValueError(f"unknown multi-label split: {split}")

    neutral = [
        MultiLabelExample(
            id=f"multilabel-{item.id}", text=item.text, policies=(), split=split,
        )
        for item in build_discourse_corpus(split)
    ]
    compound = [
        MultiLabelExample(
            id=f"multilabel-{item.id}", text=item.text,
            policies=tuple(item.policies), split=split,
        )
        for item in build_compound_corpus(split)
    ]
    return singles + neutral + compound


def _targets(examples: list[MultiLabelExample]) -> np.ndarray:
    positions = {label: index for index, label in enumerate(METHOD_LABELS)}
    result = np.zeros((len(examples), len(METHOD_LABELS)), dtype=np.float64)
    for row, example in enumerate(examples):
        for policy in example.policies:
            result[row, positions[policy]] = 1.0
    return result


def _balanced_binary_fit(
    x: np.ndarray,
    y: np.ndarray,
    ridge: float,
) -> np.ndarray:
    positives = max(1, int(np.sum(y == 1.0)))
    negatives = max(1, int(np.sum(y == 0.0)))
    weights = np.where(y == 1.0, len(y) / (2 * positives), len(y) / (2 * negatives))
    scale = np.sqrt(weights)[:, None]
    weighted_x = x * scale
    weighted_y = y * scale[:, 0]
    return np.linalg.solve(
        weighted_x.T @ weighted_x + ridge * np.eye(weighted_x.shape[1]),
        weighted_x.T @ weighted_y,
    )


def _fit_heads(
    x: np.ndarray,
    examples: list[MultiLabelExample],
    discourse_ridge: float,
    method_ridge: float,
) -> tuple[np.ndarray, np.ndarray]:
    targets = _targets(examples)
    task_target = np.asarray([bool(item.policies) for item in examples], dtype=np.float64)
    discourse_weights = _balanced_binary_fit(x, task_target, discourse_ridge)
    method_weights = np.column_stack([
        _balanced_binary_fit(x, targets[:, index], method_ridge)
        for index in range(len(METHOD_LABELS))
    ])
    return discourse_weights, method_weights


def _safe_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_precision: float,
) -> float:
    candidates = sorted({
        0.0, 1.0,
        *(float(value) for value in np.arange(0.10, 0.901, 0.025)),
        *(float(value + 1e-6) for value in scores),
    })
    best: tuple[float, float, float] | None = None
    chosen = 1.0
    for threshold in candidates:
        predicted = scores >= threshold
        true_positive = int(np.sum(predicted & expected))
        false_positive = int(np.sum(predicted & ~expected))
        false_negative = int(np.sum(~predicted & expected))
        precision = true_positive / (true_positive + false_positive) if predicted.any() else 1.0
        recall = true_positive / (true_positive + false_negative) if expected.any() else 1.0
        if precision < minimum_precision:
            continue
        key = (recall, precision, -threshold)
        if best is None or key > best:
            best = key
            chosen = threshold
    return chosen


def _select_parameters(
    calibration_x: np.ndarray,
    calibration: list[MultiLabelExample],
    validation_x: np.ndarray,
    validation: list[MultiLabelExample],
) -> tuple[HeadParameters, np.ndarray, np.ndarray, dict[str, object]]:
    expected = _targets(validation).astype(bool)
    task_expected = np.asarray([bool(item.policies) for item in validation])
    ridges = (0.01, 0.1, 1.0, 10.0)
    target_matrix = _targets(calibration)
    task_target = np.asarray([bool(item.policies) for item in calibration], dtype=np.float64)
    discourse_by_ridge = {
        ridge: _balanced_binary_fit(calibration_x, task_target, ridge)
        for ridge in ridges
    }
    methods_by_ridge = {
        ridge: np.column_stack([
            _balanced_binary_fit(calibration_x, target_matrix[:, index], ridge)
            for index in range(len(METHOD_LABELS))
        ])
        for ridge in ridges
    }
    candidates = []
    for discourse_ridge, discourse_weights in discourse_by_ridge.items():
        for method_ridge, method_weights in methods_by_ridge.items():
            discourse_scores = validation_x @ discourse_weights
            method_scores = validation_x @ method_weights
            discourse_threshold = _safe_threshold(
                discourse_scores, task_expected, minimum_precision=0.995
            )
            thresholds = tuple(
                _safe_threshold(
                    method_scores[:, index], expected[:, index], minimum_precision=0.99
                )
                for index in range(len(METHOD_LABELS))
            )
            candidate_thresholds = tuple(max(0.1, value - 0.08) for value in thresholds)
            parameters = HeadParameters(
                discourse_ridge=discourse_ridge,
                method_ridge=method_ridge,
                discourse_threshold=discourse_threshold,
                method_thresholds=thresholds,
                candidate_thresholds=candidate_thresholds,
            )
            predictions = _predict(
                validation_x, discourse_weights, method_weights, parameters
            )
            report = metrics(validation, predictions)
            key = (
                float(report["false_activations"] == 0),
                float(report["micro_precision"]),
                float(report["exact_match"]),
                float(report["task_coverage"]),
            )
            candidates.append((key, parameters, discourse_weights, method_weights, report))
    _, parameters, discourse_weights, method_weights, report = max(
        candidates, key=lambda item: item[0]
    )
    return parameters, discourse_weights, method_weights, report


def _predict(
    matrix: np.ndarray,
    discourse_weights: np.ndarray,
    method_weights: np.ndarray,
    parameters: HeadParameters,
) -> list[tuple[str, ...]]:
    discourse_scores = matrix @ discourse_weights
    method_scores = matrix @ method_weights
    predictions = []
    thresholds = np.asarray(parameters.method_thresholds)
    for discourse_score, row in zip(discourse_scores, method_scores, strict=True):
        if discourse_score < parameters.discourse_threshold:
            predictions.append(())
            continue
        selected = [
            (METHOD_LABELS[index], float(row[index]))
            for index in range(len(METHOD_LABELS))
            if row[index] >= thresholds[index]
        ]
        selected.sort(key=lambda item: item[1], reverse=True)
        predictions.append(tuple(label for label, _ in selected[:3]))
    return predictions


def metrics(
    examples: list[MultiLabelExample], predictions: list[tuple[str, ...]]
) -> dict[str, object]:
    expected_sets = [set(item.policies) for item in examples]
    predicted_sets = [set(item) for item in predictions]
    true_positive = sum(len(expected & predicted) for expected, predicted in zip(expected_sets, predicted_sets, strict=True))
    false_positive = sum(len(predicted - expected) for expected, predicted in zip(expected_sets, predicted_sets, strict=True))
    false_negative = sum(len(expected - predicted) for expected, predicted in zip(expected_sets, predicted_sets, strict=True))
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 1.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 1.0
    task_rows = sum(bool(expected) for expected in expected_sets)
    covered_task_rows = sum(bool(expected) and bool(predicted) for expected, predicted in zip(expected_sets, predicted_sets, strict=True))
    errors = [
        {"id": item.id, "expected": list(item.policies), "predicted": list(predicted)}
        for item, predicted in zip(examples, predictions, strict=True)
        if set(item.policies) != set(predicted)
    ]
    return {
        "examples": len(examples),
        "exact_match": sum(expected == predicted for expected, predicted in zip(expected_sets, predicted_sets, strict=True)) / len(examples),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "task_coverage": covered_task_rows / task_rows if task_rows else 1.0,
        "false_activations": sum(not expected and bool(predicted) for expected, predicted in zip(expected_sets, predicted_sets, strict=True)),
        "errors": errors,
    }


def _dataset_hash(examples: list[MultiLabelExample]) -> str:
    payload = "\n".join(
        json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def run_experiment(artifact: Path | None = None) -> dict[str, object]:
    """Train, select, and evaluate a true multi-label task head."""
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    calibration = build_multilabel_corpus("calibration")
    validation = build_multilabel_corpus("validation")
    holdout = build_multilabel_corpus("holdout")
    all_examples = calibration + validation + holdout
    vectors = embedder.embed_queries([item.text for item in all_examples])
    cal_end = len(calibration)
    val_end = cal_end + len(validation)
    calibration_x = _design_matrix(vectors[:cal_end])
    validation_x = _design_matrix(vectors[cal_end:val_end])
    holdout_x = _design_matrix(vectors[val_end:])
    parameters, discourse_weights, method_weights, validation_report = _select_parameters(
        calibration_x, calibration, validation_x, validation
    )
    holdout_report = metrics(
        holdout,
        _predict(holdout_x, discourse_weights, method_weights, parameters),
    )
    metadata = {
        "schema_version": 3,
        "model": MODEL_VERSION,
        "embedding_space_id": embedder.space_id,
        "method_labels": list(METHOD_LABELS),
        "parameters": asdict(parameters),
        "dataset_sha256": {
            "calibration": _dataset_hash(calibration),
            "validation": _dataset_hash(validation),
            "holdout": _dataset_hash(holdout),
        },
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            discourse_weights=discourse_weights.astype(np.float32),
            method_weights=method_weights.astype(np.float32),
            metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        )
    return {
        **metadata,
        "examples": {
            "calibration": len(calibration),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "validation": validation_report,
        "holdout": holdout_report,
        "artifact_bytes": artifact.stat().st_size if artifact else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = ["MODEL_VERSION", "MultiLabelExample", "build_multilabel_corpus", "metrics", "run_experiment"]
