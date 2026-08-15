"""Evaluate sparse word/character classifiers on human task-policy data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from bench.task_policy_gliclass_finetune import _binary_metrics, _load_partition, _targets
from bench.task_policy_multilabel_head import METHOD_LABELS
from bench.task_policy_pairwise_finetune import _select_threshold


def _classifier_candidates(
    training_texts: list[str],
    training_targets: np.ndarray,
    query_texts: list[str],
) -> dict[str, np.ndarray]:
    from scipy.sparse import hstack
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC

    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.995,
        sublinear_tf=True,
        max_features=120_000,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=180_000,
    )
    train_word = word.fit_transform(training_texts)
    train_char = char.fit_transform(training_texts)
    query_word = word.transform(query_texts)
    query_char = char.transform(query_texts)
    spaces = {
        "word": (train_word, query_word),
        "char": (train_char, query_char),
        "word-char": (hstack((train_word, train_char), format="csr"),
                      hstack((query_word, query_char), format="csr")),
    }
    candidates = {}
    for space_name, (train, query) in spaces.items():
        for class_weight in ("balanced", None):
            weight_name = "balanced" if class_weight else "uniform"
            for c_value in (0.1, 0.3, 1.0, 3.0, 10.0):
                columns = []
                for index in range(training_targets.shape[1]):
                    classifier = LogisticRegression(
                        C=c_value,
                        class_weight=class_weight,
                        max_iter=1000,
                        solver="liblinear",
                        random_state=41,
                    )
                    classifier.fit(train, training_targets[:, index])
                    columns.append(classifier.predict_proba(query)[:, 1])
                candidates[f"logreg-{space_name}-{weight_name}-c{c_value:g}"] = np.stack(
                    columns, axis=1,
                )
            for c_value in (0.01, 0.03, 0.1, 0.3, 1.0):
                columns = []
                for index in range(training_targets.shape[1]):
                    classifier = LinearSVC(
                        C=c_value,
                        class_weight=class_weight,
                        random_state=41,
                    )
                    classifier.fit(train, training_targets[:, index])
                    columns.append(classifier.decision_function(query))
                candidates[f"svm-{space_name}-{weight_name}-c{c_value:g}"] = np.stack(
                    columns, axis=1,
                )
    return candidates


def _metric_key(metrics: dict[str, float | int]) -> tuple[float, ...]:
    accuracy = float(metrics["accuracy"])
    recall = float(metrics["recall"])
    return (
        float(accuracy >= 0.95 and recall >= 0.95),
        min(accuracy / 0.95, recall / 0.95),
        min(accuracy, recall),
        float(metrics["f1"]),
        float(metrics["precision"]),
    )


def _select(
    candidates: dict[str, np.ndarray], expected: np.ndarray,
) -> tuple[tuple[str, ...], tuple[float, ...], dict[str, Any]]:
    names = []
    thresholds = []
    diagnostics = {}
    for index, label in enumerate(METHOD_LABELS):
        best = None
        for name, scores in candidates.items():
            threshold = _select_threshold(
                scores[:, index], expected[:, index],
                accuracy_target=0.95, recall_target=0.95,
            )
            metrics = _binary_metrics(scores[:, index], expected[:, index], threshold)
            value = (_metric_key(metrics), name, threshold, metrics)
            if best is None or value[0] > best[0]:
                best = value
        if best is None:
            raise RuntimeError(f"no sparse classifier candidate for {label}")
        _, name, threshold, metrics = best
        names.append(name)
        thresholds.append(threshold)
        diagnostics[label] = {"classifier": name, "threshold": threshold, "metrics": metrics}
    return tuple(names), tuple(thresholds), diagnostics


def _selected(candidates: dict[str, np.ndarray], names: tuple[str, ...]) -> np.ndarray:
    return np.stack([candidates[name][:, index] for index, name in enumerate(names)], axis=1)


def _report(
    scores: np.ndarray, expected: np.ndarray, thresholds: tuple[float, ...],
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
            "accuracy_target": 0.95,
            "recall_target": 0.95,
            "all_labels_pass": all(_metric_key(metrics)[0] for metrics in per_label.values()),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--latency-samples", type=int, default=100)
    args = parser.parse_args()

    training = _load_partition(args.data_root, "training")
    calibration = _load_partition(args.data_root, "calibration")
    evaluation = _load_partition(args.data_root, "evaluation")
    training_texts = [row.text for row in training]
    calibration_candidates = _classifier_candidates(
        training_texts, _targets(training), [row.text for row in calibration],
    )
    names, thresholds, selection = _select(calibration_candidates, _targets(calibration))
    calibration_scores = _selected(calibration_candidates, names)
    started = time.perf_counter()
    evaluation_candidates = _classifier_candidates(
        training_texts, _targets(training), [row.text for row in evaluation],
    )
    fit_and_evaluation_seconds = time.perf_counter() - started
    evaluation_scores = _selected(evaluation_candidates, names)
    # Sparse inference is dominated by vectorization; measure the complete pipeline
    # conservatively by amortizing a fresh fit/evaluation run over evaluation rows.
    estimated_single_ms = fit_and_evaluation_seconds * 1000 / max(1, len(evaluation))
    report = {
        "counts": {
            "training": len(training), "calibration": len(calibration),
            "evaluation": len(evaluation),
        },
        "selection": selection,
        "calibration": _report(calibration_scores, _targets(calibration), thresholds),
        "evaluation": _report(evaluation_scores, _targets(evaluation), thresholds),
        "fit_and_evaluation_seconds": fit_and_evaluation_seconds,
        "latency_upper_bound": {
            "interpretation": "fresh-fit amortized; deployed warm inference is lower",
            "estimated_single_ms": estimated_single_ms,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "report": str(args.output),
        "evaluation_exact_match": report["evaluation"]["exact_match"],
        "evaluation_gate": report["evaluation"]["gate"],
        "estimated_single_ms_upper_bound": estimated_single_ms,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
