"""Diagnose E5 task-policy errors with family-separated cross-validation.

This development diagnostic never replaces the future sealed holdout. It embeds
the manually authored calibration corpus once, trains a fresh frozen head per
fold, and reports which semantic boundaries need more or better examples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.contextual_embedding_benchmark import (
    BackendMeasurements,
    DEFAULT_CONTEXTUAL_MODEL,
    TrainingParameters,
    _encode_contextual,
    _prediction_metrics,
    _predict_head,
    _train_frozen_head,
)
from bench.task_policy_manual_audit import load_examples
from bench.task_policy_linear_head import _design_matrix
from bench.task_policy_multilabel_head import (
    METHOD_LABELS,
    MultiLabelExample,
    _predict as _predict_selective,
    _safe_threshold,
    _select_parameters,
    _targets,
)


CV_VERSION = "manual-task-policy-e5-cv-v2"
FOLD_ASSIGNMENT_VERSION = "stable-family-hash-v2"


def render_input(row: dict[str, Any]) -> str:
    """Render prior turns explicitly so contextual negatives are learnable."""
    context = [str(item).strip() for item in row["context_before"] if str(item).strip()]
    if not context:
        return str(row["text"])
    previous = "\n".join(f"- {item}" for item in context)
    return f"Previous context:\n{previous}\n\nCurrent user message:\n{row['text']}"


def _label_signature(row: dict[str, Any]) -> str:
    policies = sorted(str(policy) for policy in row["policies"])
    return " + ".join(policies) if policies else "uncategorized"


def assign_folds(rows: list[dict[str, Any]], fold_count: int) -> list[int]:
    """Assign stable family folds while keeping authored contrast groups together.

    A row's fold must not change when unrelated examples are appended. Exact
    round-robin balancing violates that property and makes longitudinal model
    comparisons invalid, so folds use a stable hash and balance statistically.
    """
    if fold_count < 3:
        raise ValueError("fold_count must be at least 3")
    assignments = []
    for row in rows:
        group = row.get("evaluation_group")
        if group:
            family_key = f"evaluation-group:{group}"
        else:
            family_key = (
                f"signature:{_label_signature(row)}:"
                f"scenario:{row['scenario_family']}"
            )
        digest = hashlib.sha256(f"manual-cv-v2:{family_key}".encode()).digest()
        assignments.append(int.from_bytes(digest[:8], "big") % fold_count)
    return assignments


def subsample_training_indices(
    rows: list[dict[str, Any]],
    indices: np.ndarray,
    fraction: float,
) -> np.ndarray:
    """Select a deterministic, label-stratified fraction for learning curves."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("training_fraction must be greater than 0 and at most 1")
    if fraction == 1.0:
        return indices
    buckets: dict[str, list[int]] = {}
    for index in indices.tolist():
        buckets.setdefault(_label_signature(rows[index]), []).append(index)
    selected = []
    for signature, bucket in sorted(buckets.items()):
        ordered = sorted(
            bucket,
            key=lambda index: hashlib.sha256(
                f"learning-curve:{signature}:{rows[index]['scenario_family']}".encode()
            ).digest(),
        )
        count = max(1, int(np.ceil(len(ordered) * fraction)))
        selected.extend(ordered[:count])
    return np.asarray(sorted(selected), dtype=np.int64)


def _examples(rows: list[dict[str, Any]]) -> list[MultiLabelExample]:
    return [
        MultiLabelExample(
            id=str(row["id"]),
            text=render_input(row),
            policies=tuple(str(policy) for policy in row["policies"]),
            split="cross_validation",
        )
        for row in rows
    ]


def load_precomputed_vectors(
    path: Path,
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, BackendMeasurements, str, str]:
    """Load externally encoded vectors after verifying exact row identity."""
    archive = np.load(path, allow_pickle=False)
    expected_ids = [str(row["id"]) for row in rows]
    actual_ids = archive["ids"].astype(str).tolist()
    if actual_ids != expected_ids:
        raise ValueError("precomputed vector ids do not match the current manual corpus")
    vectors = archive["vectors"].astype(np.float32, copy=False)
    if vectors.shape[0] != len(rows):
        raise ValueError("precomputed vector count does not match the manual corpus")
    model = str(archive["model"].item())
    prefix = str(archive["prefix"].item())
    corpus_seconds = float(archive["corpus_seconds"].item())
    measurements = BackendMeasurements(
        backend="contextual_precomputed",
        model=model,
        dimensions=int(vectors.shape[1]),
        load_seconds=float(archive["load_seconds"].item()),
        corpus_seconds=corpus_seconds,
        corpus_examples_per_second=len(rows) / max(corpus_seconds, 1e-9),
        warm_single_p50_ms=float(archive["warm_single_p50_ms"].item()),
        warm_single_p95_ms=float(archive["warm_single_p95_ms"].item()),
        peak_rss_delta_mib=float(archive["rss_delta_mib"].item()),
    )
    return vectors, measurements, model, prefix


def _predict_independent(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    method_thresholds: tuple[float, ...],
    task_threshold: float,
) -> list[tuple[str, ...]]:
    thresholds = np.asarray(method_thresholds)
    predictions = []
    for row, task_score in zip(method_scores, task_scores, strict=True):
        if task_score < task_threshold:
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


def _threshold_selection_key(report: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(report["false_activations"] == 0),
        float(report["exact_match"]),
        float(report["micro_precision"]),
        float(report["macro_f1"]),
        float(report["micro_recall"]),
    )


def _score_thresholds(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    validation: list[MultiLabelExample],
    method_thresholds: tuple[float, ...],
    task_threshold: float,
) -> tuple[tuple[float, ...], dict[str, Any]]:
    predictions = _predict_independent(
        method_scores, task_scores, method_thresholds, task_threshold
    )
    report = _prediction_metrics(validation, predictions)
    return _threshold_selection_key(report), report


def _precision_first_threshold(
    scores: np.ndarray,
    expected: np.ndarray,
    *,
    minimum_precision: float,
    minimum_recall: float,
) -> float:
    """Choose the safest threshold that still satisfies precision and recall floors."""
    candidates = sorted({
        0.0,
        1.0,
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
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        if precision < minimum_precision or recall < minimum_recall:
            continue
        key = (precision, threshold, recall)
        if best is None or key > best:
            best = key
            chosen = threshold
    if best is not None:
        return chosen
    return _safe_threshold(
        scores,
        expected,
        minimum_precision=minimum_precision,
    )


def select_independent_thresholds(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    validation: list[MultiLabelExample],
    *,
    minimum_method_precision: float,
    minimum_method_recall: float | None = None,
) -> tuple[tuple[float, ...], float, dict[str, Any]]:
    """Select method and abstention thresholds using validation only."""
    validation_expected = _targets(validation).astype(bool)
    thresholds = tuple(
        (
            _safe_threshold(
                method_scores[:, index],
                validation_expected[:, index],
                minimum_precision=minimum_method_precision,
            )
            if minimum_method_recall is None
            else _precision_first_threshold(
                method_scores[:, index],
                validation_expected[:, index],
                minimum_precision=minimum_method_precision,
                minimum_recall=minimum_method_recall,
            )
        )
        for index in range(len(METHOD_LABELS))
    )
    task_candidates = sorted({
        0.0,
        1.0,
        *(float(value) for value in np.arange(0.10, 0.901, 0.025)),
        *(float(value + 1e-6) for value in task_scores),
    })
    best = None
    for task_threshold in task_candidates:
        key, report = _score_thresholds(
            method_scores,
            task_scores,
            validation,
            thresholds,
            task_threshold,
        )
        if best is None or key > best[0]:
            best = (key, task_threshold, report)
    if best is None:
        raise RuntimeError("independent head produced no threshold candidate")
    _, task_threshold, report = best
    return thresholds, task_threshold, report


def select_joint_thresholds(
    method_scores: np.ndarray,
    task_scores: np.ndarray,
    validation: list[MultiLabelExample],
    *,
    minimum_method_precision: float,
    minimum_method_recall: float | None = None,
    passes: int = 3,
) -> tuple[tuple[float, ...], float, dict[str, Any]]:
    """Coordinate-calibrate all thresholds for exact multi-label predictions."""
    if passes < 1:
        raise ValueError("joint threshold calibration requires at least one pass")
    thresholds, task_threshold, report = select_independent_thresholds(
        method_scores,
        task_scores,
        validation,
        minimum_method_precision=minimum_method_precision,
        minimum_method_recall=minimum_method_recall,
    )
    best_key = _threshold_selection_key(report)
    method_candidates = [
        sorted({
            0.0,
            1.0,
            *(float(value) for value in np.arange(0.10, 0.901, 0.025)),
            *(float(value + 1e-6) for value in method_scores[:, index]),
        })
        for index in range(len(METHOD_LABELS))
    ]
    task_candidates = sorted({
        0.0,
        1.0,
        *(float(value) for value in np.arange(0.10, 0.901, 0.025)),
        *(float(value + 1e-6) for value in task_scores),
    })
    for _ in range(passes):
        changed = False
        for index, candidates in enumerate(method_candidates):
            selected_threshold = thresholds[index]
            selected_key = best_key
            selected_report = report
            for candidate in candidates:
                trial = list(thresholds)
                trial[index] = candidate
                key, candidate_report = _score_thresholds(
                    method_scores,
                    task_scores,
                    validation,
                    tuple(trial),
                    task_threshold,
                )
                precision = candidate_report["per_label"][METHOD_LABELS[index]][
                    "precision"
                ]
                if precision < minimum_method_precision:
                    continue
                if key > selected_key:
                    selected_threshold = candidate
                    selected_key = key
                    selected_report = candidate_report
            if selected_threshold != thresholds[index]:
                changed = True
                updated = list(thresholds)
                updated[index] = selected_threshold
                thresholds = tuple(updated)
                best_key = selected_key
                report = selected_report

        selected_task_threshold = task_threshold
        selected_key = best_key
        selected_report = report
        for candidate in task_candidates:
            key, candidate_report = _score_thresholds(
                method_scores,
                task_scores,
                validation,
                thresholds,
                candidate,
            )
            if key > selected_key:
                selected_task_threshold = candidate
                selected_key = key
                selected_report = candidate_report
        if selected_task_threshold != task_threshold:
            changed = True
            task_threshold = selected_task_threshold
            best_key = selected_key
            report = selected_report
        if not changed:
            break
    return thresholds, task_threshold, report


def _rbf_kernel(
    left: np.ndarray,
    right: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """Return a numerically stable radial-basis kernel matrix."""
    squared = (
        np.square(left).sum(axis=1, keepdims=True)
        + np.square(right).sum(axis=1)[None, :]
        - 2.0 * left @ right.T
    )
    return np.exp(-gamma * np.maximum(squared, 0.0))


def _train_rbf_head(
    train_vectors: np.ndarray,
    train: list[MultiLabelExample],
    validation_vectors: np.ndarray,
    validation: list[MultiLabelExample],
    *,
    minimum_method_precision: float,
) -> tuple[np.ndarray, np.ndarray, float, tuple[float, ...], float, dict[str, Any]]:
    """Select a compact nonlinear kernel head using validation only."""
    method_targets = _targets(train).astype(np.float64) * 2.0 - 1.0
    task_targets = np.asarray(
        [bool(example.policies) for example in train], dtype=np.float64
    )[:, None] * 2.0 - 1.0
    targets = np.concatenate([method_targets, task_targets], axis=1)
    distances = np.maximum(
        np.square(train_vectors).sum(axis=1, keepdims=True)
        + np.square(train_vectors).sum(axis=1)[None, :]
        - 2.0 * train_vectors @ train_vectors.T,
        0.0,
    )
    nonzero = distances[distances > 1e-8]
    median_distance = float(np.median(nonzero)) if len(nonzero) else 1.0
    gamma_candidates = [
        multiplier / max(median_distance, 1e-8)
        for multiplier in (0.25, 0.5, 1.0, 2.0, 4.0)
    ]
    best = None
    identity = np.eye(len(train_vectors), dtype=np.float64)
    for gamma in gamma_candidates:
        train_kernel = _rbf_kernel(train_vectors, train_vectors, gamma)
        validation_kernel = _rbf_kernel(
            validation_vectors, train_vectors, gamma
        )
        for regularization in (1e-4, 1e-3, 1e-2, 0.1, 1.0):
            coefficients = np.linalg.solve(
                train_kernel + regularization * identity,
                targets,
            )
            scores = validation_kernel @ coefficients
            thresholds, task_threshold, report = select_independent_thresholds(
                scores[:, :len(METHOD_LABELS)],
                scores[:, -1],
                validation,
                minimum_method_precision=minimum_method_precision,
            )
            key = (
                float(report["false_activations"] == 0),
                float(report["exact_match"]),
                float(report["micro_precision"]),
                float(report["micro_recall"]),
                -regularization,
            )
            if best is None or key > best[0]:
                best = (
                    key,
                    coefficients,
                    gamma,
                    regularization,
                    thresholds,
                    task_threshold,
                    report,
                )
    if best is None:
        raise RuntimeError("RBF head produced no candidate")
    _, coefficients, gamma, regularization, thresholds, task_threshold, report = best
    return (
        train_vectors,
        coefficients,
        gamma,
        thresholds,
        task_threshold,
        {
            "gamma": gamma,
            "regularization": regularization,
            "method_thresholds": thresholds,
            "task_threshold": task_threshold,
            "validation": report,
        },
    )


def _train_independent_head(
    calibration_vectors: np.ndarray,
    calibration: list[MultiLabelExample],
    validation_vectors: np.ndarray,
    validation: list[MultiLabelExample],
    training: TrainingParameters,
) -> tuple[object, tuple[float, ...], float, dict[str, Any]]:
    """Train independent method logits plus an explicit task/abstention logit."""
    import torch
    from torch import nn

    torch.manual_seed(training.seed)
    np.random.seed(training.seed)

    class IndependentHead(nn.Module):
        def __init__(self, dimensions: int) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Linear(dimensions, training.hidden_size),
                nn.LayerNorm(training.hidden_size),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            self.methods = nn.Linear(training.hidden_size, len(METHOD_LABELS))
            self.task = nn.Linear(training.hidden_size, 1)

        def forward(self, vectors: object) -> tuple[object, object]:
            hidden = self.body(vectors)
            return self.methods(hidden), self.task(hidden).squeeze(1)

    calibration_x = torch.from_numpy(calibration_vectors)
    validation_x = torch.from_numpy(validation_vectors)
    calibration_y = torch.from_numpy(_targets(calibration).astype(np.float32))
    task_y = torch.tensor(
        [bool(item.policies) for item in calibration], dtype=torch.float32
    )
    positive_counts = calibration_y.sum(dim=0)
    method_loss = nn.BCEWithLogitsLoss(
        pos_weight=(len(calibration) - positive_counts)
        / positive_counts.clamp(min=1)
    )
    task_positives = task_y.sum()
    task_loss = nn.BCEWithLogitsLoss(
        pos_weight=(len(calibration) - task_positives)
        / task_positives.clamp(min=1)
    )
    head = IndependentHead(calibration_vectors.shape[1])
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    best_key = (-1.0, -1.0, -1.0, -1.0)
    best_state = None
    best_thresholds: tuple[float, ...] = (0.5,) * len(METHOD_LABELS)
    best_task_threshold = 0.5
    best_report: dict[str, Any] = {}
    stale = 0
    for epoch in range(training.max_epochs + 1):
        head.train()
        order = torch.randperm(len(calibration))
        for start in range(0, len(calibration), training.batch_size):
            indices = order[start:start + training.batch_size]
            method_logits, task_logits = head(calibration_x[indices])
            loss = method_loss(method_logits, calibration_y[indices])
            loss = loss + training.cardinality_loss_weight * task_loss(
                task_logits, task_y[indices]
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % training.evaluate_every:
            continue
        head.eval()
        with torch.inference_mode():
            method_logits, task_logits = head(validation_x)
            method_scores = method_logits.sigmoid().numpy()
            task_scores = task_logits.sigmoid().numpy()
        thresholds, task_threshold, report = select_independent_thresholds(
            method_scores,
            task_scores,
            validation,
            minimum_method_precision=training.minimum_method_precision,
        )
        key = (
            float(report["false_activations"] == 0),
            float(report["exact_match"]),
            float(report["micro_precision"]),
            float(report["micro_recall"]),
        )
        if key > best_key:
            best_key = key
            best_state = {
                name: value.detach().clone() for name, value in head.state_dict().items()
            }
            best_thresholds = thresholds
            best_task_threshold = task_threshold
            best_report = report
            stale = 0
        else:
            stale += 1
            if stale >= training.patience_evaluations:
                break
    if best_state is None:
        raise RuntimeError("independent head produced no checkpoint")
    head.load_state_dict(best_state)
    return head, best_thresholds, best_task_threshold, {
        "validation": best_report,
        "method_thresholds": best_thresholds,
        "task_threshold": best_task_threshold,
    }


def _independent_scores(head: object, vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    head.eval()  # type: ignore[attr-defined]
    with torch.inference_mode():
        method_logits, task_logits = head(torch.from_numpy(vectors))  # type: ignore[operator]
    return method_logits.sigmoid().numpy(), task_logits.sigmoid().numpy()


def run_cross_validation(
    *,
    fold_count: int = 5,
    model_name: str = DEFAULT_CONTEXTUAL_MODEL,
    max_length: int = 192,
    encode_batch_size: int = 64,
    head_kind: str = "selective_ridge",
    training: TrainingParameters | None = None,
    training_fraction: float = 1.0,
    input_prefix: str = "query: ",
    trust_remote_code: bool = False,
    model_revision: str | None = None,
    portable_cpu_attention: bool = False,
    vectors_path: Path | None = None,
) -> dict[str, Any]:
    """Return out-of-fold predictions for every manual calibration example."""
    import torch

    rows = load_examples()
    examples = _examples(rows)
    folds = assign_folds(rows, fold_count)
    if vectors_path is None:
        vectors, measurements = _encode_contextual(
            [example.text for example in examples],
            model_name=model_name,
            batch_size=encode_batch_size,
            max_length=max_length,
            input_prefix=input_prefix,
            trust_remote_code=trust_remote_code,
            model_revision=model_revision,
            portable_cpu_attention=portable_cpu_attention,
        )
    else:
        vectors, measurements, model_name, input_prefix = load_precomputed_vectors(
            vectors_path, rows
        )
    training = training or TrainingParameters(
        hidden_size=96,
        max_epochs=180,
        patience_evaluations=10,
        seed=31,
    )
    predictions: list[tuple[str, ...] | None] = [None] * len(examples)
    fold_reports = []
    for test_fold in range(fold_count):
        validation_fold = (test_fold + 1) % fold_count
        available_train_indices = np.asarray([
            index
            for index, fold in enumerate(folds)
            if fold not in {test_fold, validation_fold}
        ])
        train_indices = subsample_training_indices(
            rows, available_train_indices, training_fraction
        )
        validation_indices = np.asarray([
            index for index, fold in enumerate(folds) if fold == validation_fold
        ])
        test_indices = np.asarray([
            index for index, fold in enumerate(folds) if fold == test_fold
        ])
        train_examples = [examples[index] for index in train_indices]
        validation_examples = [examples[index] for index in validation_indices]
        if head_kind == "selective_ridge":
            parameters, discourse_weights, method_weights, validation_report = (
                _select_parameters(
                    _design_matrix(vectors[train_indices]),
                    train_examples,
                    _design_matrix(vectors[validation_indices]),
                    validation_examples,
                )
            )
            fold_predictions = _predict_selective(
                _design_matrix(vectors[test_indices]),
                discourse_weights,
                method_weights,
                parameters,
            )
            selection_summary: dict[str, Any] = {
                "parameters": asdict(parameters),
                "validation": validation_report,
            }
        elif head_kind == "cardinality_mlp":
            head, selection = _train_frozen_head(
                vectors[train_indices],
                train_examples,
                vectors[validation_indices],
                validation_examples,
                training,
            )
            fold_predictions = _predict_head(
                head, torch.from_numpy(vectors[test_indices])
            )
            selection_summary = {
                "best_epoch": selection["best_epoch"],
                "validation": selection["validation"],
            }
        elif head_kind == "independent_mlp":
            head, method_thresholds, task_threshold, selection_summary = (
                _train_independent_head(
                    vectors[train_indices],
                    train_examples,
                    vectors[validation_indices],
                    validation_examples,
                    training,
                )
            )
            method_scores, task_scores = _independent_scores(
                head, vectors[test_indices]
            )
            fold_predictions = _predict_independent(
                method_scores,
                task_scores,
                method_thresholds,
                task_threshold,
            )
        elif head_kind == "rbf_kernel":
            (
                support_vectors,
                coefficients,
                gamma,
                method_thresholds,
                task_threshold,
                selection_summary,
            ) = _train_rbf_head(
                vectors[train_indices],
                train_examples,
                vectors[validation_indices],
                validation_examples,
                minimum_method_precision=training.minimum_method_precision,
            )
            scores = _rbf_kernel(
                vectors[test_indices], support_vectors, gamma
            ) @ coefficients
            fold_predictions = _predict_independent(
                scores[:, :len(METHOD_LABELS)],
                scores[:, -1],
                method_thresholds,
                task_threshold,
            )
        else:
            raise ValueError(f"unknown head_kind: {head_kind}")
        for index, prediction in zip(test_indices, fold_predictions, strict=True):
            predictions[int(index)] = prediction
        fold_reports.append({
            "fold": test_fold,
            "train": len(train_indices),
            "validation": len(validation_indices),
            "test": len(test_indices),
            "selection": selection_summary,
            "metrics": _prediction_metrics(
                [examples[index] for index in test_indices], fold_predictions
            ),
        })
    if any(prediction is None for prediction in predictions):
        raise RuntimeError("cross-validation left examples without predictions")
    final_predictions = [prediction for prediction in predictions if prediction is not None]
    aggregate = _prediction_metrics(examples, final_predictions)
    errors = []
    false_activation_reasons: Counter[str] = Counter()
    for row, expected, predicted in zip(rows, examples, final_predictions, strict=True):
        if set(expected.policies) == set(predicted):
            continue
        if not expected.policies and predicted:
            false_activation_reasons[str(row["uncategorized_reason"])] += 1
        errors.append({
            "id": row["id"],
            "expected": list(expected.policies),
            "predicted": list(predicted),
            "uncategorized_reason": row["uncategorized_reason"],
            "language": row["language"],
            "project_type": row["project_type"],
            "style": row["style"],
            "scenario_family": row["scenario_family"],
        })
    aggregate.pop("error_sample", None)
    return {
        "version": CV_VERSION,
        "purpose": "development diagnostic; not sealed-holdout evidence",
        "model": model_name,
        "model_input": {
            "prefix": input_prefix,
            "trust_remote_code": trust_remote_code,
            "revision": model_revision,
            "portable_cpu_attention": portable_cpu_attention,
        },
        "head": head_kind,
        "backend": asdict(measurements),
        "training": asdict(training),
        "training_fraction": training_fraction,
        "examples": len(examples),
        "folds": fold_reports,
        "aggregate": aggregate,
        "false_activation_reasons": dict(sorted(false_activation_reasons.items())),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_CONTEXTUAL_MODEL)
    parser.add_argument("--max-length", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--max-epochs", type=int, default=180)
    parser.add_argument("--patience-evaluations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--cardinality-balance-power", type=float, default=0.0)
    parser.add_argument("--minimum-method-precision", type=float, default=0.85)
    parser.add_argument("--training-fraction", type=float, default=1.0)
    parser.add_argument("--input-prefix", default="query: ")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--model-revision")
    parser.add_argument("--portable-cpu-attention", action="store_true")
    parser.add_argument("--vectors", type=Path)
    parser.add_argument(
        "--head",
        choices=("selective_ridge", "cardinality_mlp", "independent_mlp", "rbf_kernel"),
        default="selective_ridge",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_cross_validation(
        fold_count=args.folds,
        model_name=args.model,
        max_length=args.max_length,
        head_kind=args.head,
        training_fraction=args.training_fraction,
        input_prefix=args.input_prefix,
        trust_remote_code=args.trust_remote_code,
        model_revision=args.model_revision,
        portable_cpu_attention=args.portable_cpu_attention,
        vectors_path=args.vectors,
        training=TrainingParameters(
            hidden_size=args.hidden_size,
            max_epochs=args.max_epochs,
            patience_evaluations=args.patience_evaluations,
            seed=args.seed,
            cardinality_balance_power=args.cardinality_balance_power,
            minimum_method_precision=args.minimum_method_precision,
        ),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(json.dumps({
            "output": str(args.output),
            "head": report["head"],
            "examples": report["examples"],
            "aggregate": report["aggregate"],
            "false_activation_reasons": report["false_activation_reasons"],
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(rendered)


if __name__ == "__main__":
    main()


__all__ = ["CV_VERSION", "assign_folds", "render_input", "run_cross_validation"]
