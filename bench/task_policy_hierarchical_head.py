"""Train a selective discourse gate plus task-method head over static Qwen3.

The first head separates actionable work from conversation, quoted actions, and
conceptual questions. Only accepted task text reaches the second head, which
chooses a method. This avoids forcing ``uncategorized`` to compete with methods
in one flat output layer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from bench.task_policy_linear_head import (
    LABELS,
    _build_development_corpus,
    _dataset_hash,
    _design_matrix,
    _fit_ridge,
    _materialize_calibration,
    build_ambiguity_challenge,
    build_linear_head_holdout,
)
from bench.task_policy_semantic_eval import SemanticExample, build_semantic_validation_corpus
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


METHOD_LABELS = LABELS[:-1]
MODEL_VERSION = "static-qwen3-task-policy-hierarchical-head-v4"


@dataclass(frozen=True)
class HeadParameters:
    """Parameters selected before evaluating the retained holdout."""

    discourse_ridge: float
    method_ridge: float
    discourse_threshold: float
    discourse_margin: float
    method_threshold: float
    method_margin: float
    agreement_method_threshold: float
    agreement_method_margin: float


def _method_targets(examples: list[SemanticExample]) -> np.ndarray:
    lookup = {label: index for index, label in enumerate(METHOD_LABELS)}
    result = np.zeros((len(examples), len(METHOD_LABELS)), dtype=np.float64)
    for row, example in enumerate(examples):
        if example.policy is None:
            raise ValueError("method targets cannot contain uncategorized rows")
        result[row, lookup[example.policy]] = 1.0
    return result


def _discourse_targets(examples: list[SemanticExample]) -> np.ndarray:
    result = np.zeros((len(examples), 2), dtype=np.float64)
    for row, example in enumerate(examples):
        result[row, int(example.policy is not None)] = 1.0
    return result


def _predict(
    matrix: np.ndarray,
    discourse_weights: np.ndarray,
    method_weights: np.ndarray,
    parameters: HeadParameters,
) -> list[str | None]:
    discourse_scores = matrix @ discourse_weights
    method_scores = matrix @ method_weights
    predictions: list[str | None] = []
    for discourse_row, method_row in zip(
        discourse_scores, method_scores, strict=True
    ):
        discourse_order = np.argsort(discourse_row)[::-1]
        discourse_score = float(discourse_row[discourse_order[0]])
        discourse_margin = float(
            discourse_score - discourse_row[discourse_order[1]]
        )
        if (
            discourse_order[0] != 1
            or discourse_score < parameters.discourse_threshold
            or discourse_margin < parameters.discourse_margin
        ):
            predictions.append(None)
            continue

        method_order = np.argsort(method_row)[::-1]
        method_score = float(method_row[method_order[0]])
        method_margin = float(method_score - method_row[method_order[1]])
        if (
            method_score < parameters.method_threshold
            or method_margin < parameters.method_margin
        ):
            predictions.append(None)
        else:
            predictions.append(METHOD_LABELS[int(method_order[0])])
    return predictions


def _metrics(
    examples: list[SemanticExample], predictions: list[str | None]
) -> dict[str, object]:
    covered = sum(prediction is not None for prediction in predictions)
    correct = sum(
        prediction == example.policy
        for example, prediction in zip(examples, predictions, strict=True)
    )
    correct_covered = sum(
        prediction is not None and prediction == example.policy
        for example, prediction in zip(examples, predictions, strict=True)
    )
    errors = [
        {
            "id": example.id,
            "expected": example.policy,
            "predicted": prediction,
        }
        for example, prediction in zip(examples, predictions, strict=True)
        if prediction is not None and prediction != example.policy
    ]
    return {
        "examples": len(examples),
        "coverage": covered / len(examples),
        "exact_match": correct / len(examples),
        "selective_precision": correct_covered / covered if covered else 1.0,
        "false_activations": sum(
            prediction is not None and example.policy is None
            for example, prediction in zip(examples, predictions, strict=True)
        ),
        "classification_errors": errors,
    }


def _select_parameters(
    calibration_x: np.ndarray,
    calibration: list[SemanticExample],
    development_x: np.ndarray,
    development: list[SemanticExample],
) -> tuple[HeadParameters, np.ndarray, np.ndarray, dict[str, object]]:
    task_rows = [
        index for index, example in enumerate(calibration)
        if example.policy is not None
    ]
    task_examples = [calibration[index] for index in task_rows]
    candidates: list[
        tuple[
            tuple[float, float, float],
            HeadParameters,
            np.ndarray,
            np.ndarray,
            dict[str, object],
        ]
    ] = []
    for discourse_ridge in (0.01, 0.1, 1.0):
        discourse_weights = _fit_ridge(
            calibration_x,
            _discourse_targets(calibration),
            discourse_ridge,
        )
        for method_ridge in (0.01, 0.1, 1.0):
            method_weights = _fit_ridge(
                calibration_x[task_rows],
                _method_targets(task_examples),
                method_ridge,
            )
            for discourse_threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                for discourse_margin in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
                    for method_threshold in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8):
                        for method_margin in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25):
                            parameters = HeadParameters(
                                discourse_ridge=discourse_ridge,
                                method_ridge=method_ridge,
                                discourse_threshold=discourse_threshold,
                                discourse_margin=discourse_margin,
                                method_threshold=method_threshold,
                                method_margin=method_margin,
                                # Calibrated on the complete router validation:
                                # this tier is never sufficient without literal
                                # or contrastive agreement.
                                agreement_method_threshold=0.4,
                                agreement_method_margin=0.05,
                            )
                            metrics = _metrics(
                                development,
                                _predict(
                                    development_x,
                                    discourse_weights,
                                    method_weights,
                                    parameters,
                                ),
                            )
                            key = (
                                float(
                                    metrics["selective_precision"] == 1.0
                                    and metrics["false_activations"] == 0
                                ),
                                float(metrics["selective_precision"]),
                                float(metrics["coverage"]),
                            )
                            candidates.append((
                                key,
                                parameters,
                                discourse_weights,
                                method_weights,
                                metrics,
                            ))
    _, parameters, discourse_weights, method_weights, metrics = max(
        candidates, key=lambda item: item[0]
    )
    return parameters, discourse_weights, method_weights, metrics


def run_experiment(artifact: Path | None = None) -> dict[str, object]:
    """Fit/tune the hierarchy and evaluate the retained phrase-family holdout."""
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    calibration = _materialize_calibration()
    validation = build_semantic_validation_corpus()
    ambiguity_development = build_ambiguity_challenge()
    template_development = _build_development_corpus()
    holdout = build_linear_head_holdout()
    all_examples = (
        calibration
        + validation
        + ambiguity_development
        + template_development
        + holdout
    )
    vectors = embedder.embed_queries([example.text for example in all_examples])
    cal_end = len(calibration)
    validation_end = cal_end + len(validation)
    ambiguity_end = validation_end + len(ambiguity_development)
    template_end = ambiguity_end + len(template_development)
    calibration_x = _design_matrix(vectors[:cal_end])
    validation_x = _design_matrix(vectors[cal_end:validation_end])
    ambiguity_x = _design_matrix(vectors[validation_end:ambiguity_end])
    template_x = _design_matrix(vectors[ambiguity_end:template_end])
    holdout_x = _design_matrix(vectors[template_end:])
    selection_examples = validation + ambiguity_development
    selection_x = np.vstack((validation_x, ambiguity_x))
    parameters, discourse_weights, method_weights, selection_metrics = (
        _select_parameters(
            calibration_x,
            calibration,
            selection_x,
            selection_examples,
        )
    )
    metadata = {
        "schema_version": 2,
        "model": MODEL_VERSION,
        "embedding_space_id": embedder.space_id,
        "labels": list(LABELS),
        "method_labels": list(METHOD_LABELS),
        "discourse_labels": ["uncategorized", "task"],
        "parameters": asdict(parameters),
        "calibration_sha256": _dataset_hash(calibration),
        "validation_sha256": _dataset_hash(validation),
        "ambiguity_development_sha256": _dataset_hash(ambiguity_development),
        "template_development_sha256": _dataset_hash(template_development),
        "holdout_sha256": _dataset_hash(holdout),
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            discourse_weights=discourse_weights.astype(np.float32),
            method_weights=method_weights.astype(np.float32),
            metadata=np.frombuffer(
                json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8
            ),
        )
    return {
        **metadata,
        "calibration_examples": len(calibration),
        "selection": selection_metrics,
        "validation": _metrics(
            validation,
            _predict(validation_x, discourse_weights, method_weights, parameters),
        ),
        "ambiguity_development": _metrics(
            ambiguity_development,
            _predict(ambiguity_x, discourse_weights, method_weights, parameters),
        ),
        "template_development": _metrics(
            template_development,
            _predict(template_x, discourse_weights, method_weights, parameters),
        ),
        "holdout": _metrics(
            holdout,
            _predict(holdout_x, discourse_weights, method_weights, parameters),
        ),
        "artifact_bytes": artifact.stat().st_size if artifact is not None else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_experiment(args.artifact), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
