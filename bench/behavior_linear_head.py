"""Fit an ultra-small behavior head over frozen static-Qwen3 embeddings."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from bench.behavior_semantic_eval import (
    LABELS,
    BehaviorExample,
    behavior_metrics,
    build_behavior_corpus,
    corpus_sha256,
)
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


@dataclass(frozen=True)
class HeadParameters:
    """Parameters selected only against the validation split."""

    ridge: float
    threshold: float
    margin: float


def _design(vectors: list[np.ndarray]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    return np.column_stack((matrix, np.ones(len(matrix), dtype=np.float64)))


def _targets(examples: list[BehaviorExample]) -> np.ndarray:
    result = np.zeros((len(examples), len(LABELS)), dtype=np.float64)
    positions = {label: index for index, label in enumerate(LABELS)}
    for row, example in enumerate(examples):
        if example.label:
            result[row, positions[example.label]] = 1.0
    return result


def _fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    dual = np.linalg.solve(x @ x.T + ridge * np.eye(len(x)), y)
    return x.T @ dual


def _predict(scores: np.ndarray, threshold: float, margin: float) -> list[str | None]:
    predictions: list[str | None] = []
    for row in scores:
        order = np.argsort(row)[::-1]
        top = int(order[0])
        if row[top] < threshold or row[top] - row[int(order[1])] < margin:
            predictions.append(None)
        else:
            predictions.append(LABELS[top])
    return predictions


def _select(
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    validation_x: np.ndarray,
    validation: list[BehaviorExample],
) -> tuple[HeadParameters, np.ndarray, dict[str, object]]:
    candidates: list[
        tuple[int, float, float, HeadParameters, np.ndarray, dict[str, object]]
    ] = []
    for ridge in (0.001, 0.01, 0.1, 1.0, 10.0):
        weights = _fit_ridge(calibration_x, calibration_y, ridge)
        scores = validation_x @ weights
        for threshold in np.arange(0.15, 0.701, 0.025):
            for margin in (0.0, 0.02, 0.04, 0.06, 0.10, 0.15):
                predictions = _predict(scores, float(threshold), margin)
                metrics = behavior_metrics(validation, predictions)
                safe = int(
                    metrics["selective_precision"] >= 0.99
                    and metrics["false_activation_rate"] == 0.0
                )
                candidates.append((
                    safe,
                    float(metrics["selective_precision"]),
                    float(metrics["coverage"]),
                    HeadParameters(ridge, float(threshold), margin),
                    weights,
                    metrics,
                ))
    _, _, _, parameters, weights, metrics = max(
        candidates, key=lambda item: item[:3]
    )
    return parameters, weights, metrics


def run_experiment(artifact: Path | None = None) -> dict[str, object]:
    """Fit, select thresholds, then evaluate the untouched holdout once."""
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    calibration = build_behavior_corpus("calibration")
    validation = build_behavior_corpus("validation")
    holdout = build_behavior_corpus("holdout")
    vectors = embedder.embed_queries(
        [item.text for item in calibration + validation + holdout]
    )
    cal_end = len(calibration)
    val_end = cal_end + len(validation)
    calibration_x = _design(vectors[:cal_end])
    validation_x = _design(vectors[cal_end:val_end])
    holdout_x = _design(vectors[val_end:])
    parameters, weights, validation_metrics = _select(
        calibration_x,
        _targets(calibration),
        validation_x,
        validation,
    )
    holdout_metrics = behavior_metrics(
        holdout,
        _predict(holdout_x @ weights, parameters.threshold, parameters.margin),
    )
    metadata = {
        "schema_version": 1,
        "model": "static-qwen3-behavior-linear-head-v1",
        "embedding_space_id": embedder.space_id,
        "labels": list(LABELS),
        "parameters": asdict(parameters),
        "calibration_sha256": corpus_sha256(calibration),
        "validation_sha256": corpus_sha256(validation),
        "holdout_sha256": corpus_sha256(holdout),
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            weights=weights.astype(np.float32),
            metadata=np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
            ),
        )
    return {
        **metadata,
        "calibration_examples": len(calibration),
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "artifact_bytes": artifact.stat().st_size if artifact else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
