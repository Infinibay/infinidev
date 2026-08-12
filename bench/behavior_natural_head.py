"""Train an ultra-small behavior head on natural, project-separated trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from infinidev.engine.behavior.semantic_classifier import classify_step_behavior
from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


POSITIVE_LABELS = ("excessive_exploration", "healthy_progress", "retry_loop")
CLASSES = (*POSITIVE_LABELS, "uncategorized")
CALIBRATION_FAMILIES = frozenset(
    {
        "code_review",
        "p-map",
        "requests",
        "reversible_ambiguity",
        "tool_failure_recovery",
    }
)
VALIDATION_FAMILIES = frozenset(
    {"complex_plan", "ripgrep-15.1.0", "test_selection", "user_owned_tradeoff"}
)
HOLDOUT_FAMILIES = frozenset({"cxxopts", "jsmn"})
FEATURE_NAMES = (
    "modifying_task",
    "read_only_task",
    "tool_prefix",
    "tool_calls_log",
    "discovery_ratio",
    "read_ratio",
    "edit_calls_log",
    "test_calls_log",
    "failed_calls_log",
    "repeated_call_max_log",
    "net_workspace_changed",
)
_MODIFYING_CATEGORIES = frozenset(
    {"bugfix", "feature", "implementation", "migration", "performance", "refactor"}
)
_READ_ONLY_CATEGORIES = frozenset({"code_review", "investigation", "research"})


@dataclass(frozen=True)
class NaturalExample:
    """One approved observable behavior window."""

    id: str
    project_family: str
    text: str
    label: str | None
    features: tuple[float, ...] = ()


def _observable_features(row: dict[str, object]) -> tuple[float, ...]:
    """Encode only signals available at the end of the captured window."""
    category = str(row.get("task_category") or "")
    tool_calls = max(int(row.get("tool_calls") or 0), 1)
    scale = np.log1p(32.0)
    return (
        float(category in _MODIFYING_CATEGORIES),
        float(category in _READ_ONLY_CATEGORIES),
        float(row.get("window_kind") == "tool_prefix"),
        float(np.log1p(tool_calls) / scale),
        float(int(row.get("discovery_calls") or 0) / tool_calls),
        float(int(row.get("read_calls") or 0) / tool_calls),
        float(np.log1p(int(row.get("edit_calls") or 0)) / scale),
        float(np.log1p(int(row.get("test_calls") or 0)) / scale),
        float(np.log1p(int(row.get("failed_calls") or 0)) / scale),
        float(np.log1p(int(row.get("repeated_call_max") or 0)) / scale),
        float(bool(row.get("net_workspace_changed"))),
    )


@dataclass(frozen=True)
class HeadParameters:
    """Abstention parameters selected without holdout access."""

    ridge: float
    threshold: float
    class_margin: float
    neutral_margin: float


def load_examples(path: Path) -> list[NaturalExample]:
    """Load approved supported labels from a privacy-reduced JSONL corpus."""
    examples: list[NaturalExample] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = row.get("label")
        if label not in (*POSITIVE_LABELS, None):
            continue
        if row.get("review_status") != "approved":
            raise ValueError("natural head input must contain only approved windows")
        examples.append(NaturalExample(
            id=str(row["id"]),
            project_family=str(row["project_family"]),
            text=str(row["text"]),
            label=label,
            features=_observable_features(row),
        ))
    if not examples:
        raise ValueError("natural head corpus is empty")
    return examples


def corpus_sha256(examples: list[NaturalExample]) -> str:
    payload = "\n".join(
        json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) for item in examples
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def split_examples(
    historical: list[NaturalExample], holdout: list[NaturalExample]
) -> tuple[list[NaturalExample], list[NaturalExample], list[NaturalExample]]:
    """Apply immutable project-family splits and reject leakage."""
    calibration = [item for item in historical if item.project_family in CALIBRATION_FAMILIES]
    validation = [item for item in historical if item.project_family in VALIDATION_FAMILIES]
    final_holdout = [item for item in holdout if item.project_family in HOLDOUT_FAMILIES]
    assigned = calibration + validation
    unknown_historical = {
        item.project_family for item in historical if item not in assigned
    }
    unknown_holdout = {
        item.project_family for item in holdout if item not in final_holdout
    }
    if unknown_historical or unknown_holdout:
        raise ValueError(
            f"unassigned project families: historical={sorted(unknown_historical)}, "
            f"holdout={sorted(unknown_holdout)}"
        )
    split_families = [
        {item.project_family for item in values}
        for values in (calibration, validation, final_holdout)
    ]
    if any(split_families[i] & split_families[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("project family leaked across corpus splits")
    if any(not values for values in (calibration, validation, final_holdout)):
        raise ValueError("every natural corpus split must be non-empty")
    return calibration, validation, final_holdout


def _design(
    vectors: list[np.ndarray],
    examples: list[NaturalExample],
    *,
    include_embeddings: bool = True,
    include_observables: bool = True,
) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    observable = np.asarray([item.features for item in examples], dtype=np.float64)
    if observable.shape != (len(examples), len(FEATURE_NAMES)):
        raise ValueError("natural examples have an invalid observable feature vector")
    parts = []
    if include_embeddings:
        parts.append(matrix)
    if include_observables:
        parts.append(observable)
    if not parts:
        raise ValueError("natural head needs embeddings, observables, or both")
    return np.column_stack((*parts, np.ones(len(matrix), dtype=np.float64)))


def _targets(examples: list[NaturalExample]) -> np.ndarray:
    positions = {label: index for index, label in enumerate(CLASSES)}
    result = np.zeros((len(examples), len(CLASSES)), dtype=np.float64)
    for row, example in enumerate(examples):
        result[row, positions[example.label or "uncategorized"]] = 1.0
    return result


def _balanced_weights(examples: list[NaturalExample]) -> np.ndarray:
    labels = [item.label or "uncategorized" for item in examples]
    counts = Counter(labels)
    missing = set(CLASSES) - counts.keys()
    if missing:
        raise ValueError(f"calibration split lacks classes: {sorted(missing)}")
    return np.asarray(
        [len(examples) / (len(CLASSES) * counts[label]) for label in labels],
        dtype=np.float64,
    )


def _fit_ridge(
    x: np.ndarray, y: np.ndarray, sample_weights: np.ndarray, ridge: float
) -> np.ndarray:
    scale = np.sqrt(sample_weights)[:, None]
    weighted_x = x * scale
    weighted_y = y * scale
    dual = np.linalg.solve(
        weighted_x @ weighted_x.T + ridge * np.eye(len(weighted_x)),
        weighted_y,
    )
    return weighted_x.T @ dual


def _predict(scores: np.ndarray, parameters: HeadParameters) -> list[str | None]:
    predictions: list[str | None] = []
    neutral_index = CLASSES.index("uncategorized")
    for row in scores:
        positive = row[:neutral_index]
        order = np.argsort(positive)[::-1]
        top = int(order[0])
        if (
            positive[top] < parameters.threshold
            or positive[top] - positive[int(order[1])] < parameters.class_margin
            or positive[top] - row[neutral_index] < parameters.neutral_margin
        ):
            predictions.append(None)
        else:
            predictions.append(POSITIVE_LABELS[top])
    return predictions


def behavior_metrics(
    examples: list[NaturalExample], predictions: list[str | None]
) -> dict[str, object]:
    """Return safety-first selective metrics, including hard-neutral activations."""
    selected = sum(item is not None for item in predictions)
    correct_selected = sum(
        predicted is not None and predicted == example.label
        for example, predicted in zip(examples, predictions, strict=True)
    )
    neutrals = sum(example.label is None for example in examples)
    false_activations = sum(
        example.label is None and predicted is not None
        for example, predicted in zip(examples, predictions, strict=True)
    )
    positive_recalls: list[float] = []
    per_label: dict[str, dict[str, int]] = defaultdict(
        lambda: {"expected": 0, "selected": 0, "correct": 0}
    )
    for example, predicted in zip(examples, predictions, strict=True):
        expected = example.label or "uncategorized"
        per_label[expected]["expected"] += 1
        if predicted:
            per_label[predicted]["selected"] += 1
        if predicted == example.label:
            per_label[expected]["correct"] += 1
    for label in POSITIVE_LABELS:
        expected = per_label[label]["expected"]
        if expected:
            positive_recalls.append(per_label[label]["correct"] / expected)
    return {
        "examples": len(examples),
        "coverage": selected / len(examples),
        "exact_match": sum(
            predicted == example.label
            for example, predicted in zip(examples, predictions, strict=True)
        ) / len(examples),
        "selective_precision": correct_selected / selected if selected else 1.0,
        "positive_macro_recall": sum(positive_recalls) / len(positive_recalls)
        if positive_recalls else 0.0,
        "neutral_false_activation_rate": false_activations / neutrals if neutrals else 0.0,
        "per_label": dict(sorted(per_label.items())),
    }


def _select(
    calibration_x: np.ndarray,
    calibration_y: np.ndarray,
    sample_weights: np.ndarray,
    validation_x: np.ndarray,
    validation: list[NaturalExample],
) -> tuple[HeadParameters, np.ndarray, dict[str, object]]:
    candidates: list[tuple[tuple[float, ...], HeadParameters, np.ndarray, dict[str, object]]] = []
    for ridge in (0.01, 0.1, 1.0, 10.0):
        weights = _fit_ridge(calibration_x, calibration_y, sample_weights, ridge)
        scores = validation_x @ weights
        positive_scores = scores[:, : len(POSITIVE_LABELS)]
        low = float(np.min(positive_scores)) - 0.05
        high = float(np.max(positive_scores)) + 0.05
        for threshold in np.linspace(low, high, 25):
            for class_margin in (0.0, 0.025, 0.05, 0.1, 0.15):
                for neutral_margin in (-0.05, 0.0, 0.05, 0.1, 0.2):
                    parameters = HeadParameters(
                        ridge, float(threshold), class_margin, neutral_margin
                    )
                    metrics = behavior_metrics(validation, _predict(scores, parameters))
                    safe = float(
                        metrics["selective_precision"] >= 0.9
                        and metrics["neutral_false_activation_rate"] == 0.0
                        and metrics["coverage"] > 0.0
                    )
                    key = (
                        safe,
                        float(metrics["positive_macro_recall"]),
                        float(metrics["selective_precision"]),
                        float(metrics["coverage"]),
                    )
                    candidates.append((key, parameters, weights, metrics))
    _, parameters, weights, metrics = max(candidates, key=lambda item: item[0])
    return parameters, weights, metrics


def _evaluate_head(
    vectors: list[np.ndarray],
    calibration: list[NaturalExample],
    validation: list[NaturalExample],
    holdout: list[NaturalExample],
    *,
    include_embeddings: bool,
    include_observables: bool,
) -> tuple[HeadParameters, np.ndarray, dict[str, object], dict[str, object]]:
    calibration_end = len(calibration)
    validation_end = calibration_end + len(validation)
    calibration_x = _design(
        vectors[:calibration_end],
        calibration,
        include_embeddings=include_embeddings,
        include_observables=include_observables,
    )
    validation_x = _design(
        vectors[calibration_end:validation_end],
        validation,
        include_embeddings=include_embeddings,
        include_observables=include_observables,
    )
    holdout_x = _design(
        vectors[validation_end:],
        holdout,
        include_embeddings=include_embeddings,
        include_observables=include_observables,
    )
    parameters, weights, validation_metrics = _select(
        calibration_x,
        _targets(calibration),
        _balanced_weights(calibration),
        validation_x,
        validation,
    )
    return (
        parameters,
        weights,
        validation_metrics,
        behavior_metrics(holdout, _predict(holdout_x @ weights, parameters)),
    )


def run_experiment(
    historical_path: Path, holdout_path: Path, artifact: Path | None = None
) -> dict[str, object]:
    """Fit on historical families, calibrate separately, then open the holdout once."""
    historical = load_examples(historical_path)
    holdout_input = load_examples(holdout_path)
    calibration, validation, holdout = split_examples(historical, holdout_input)
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    all_examples = calibration + validation + holdout
    vectors = embedder.embed_queries([item.text for item in all_examples])
    hybrid = _evaluate_head(
        vectors,
        calibration,
        validation,
        holdout,
        include_embeddings=True,
        include_observables=True,
    )
    embeddings_only = _evaluate_head(
        vectors,
        calibration,
        validation,
        holdout,
        include_embeddings=True,
        include_observables=False,
    )
    observables_only = _evaluate_head(
        vectors,
        calibration,
        validation,
        holdout,
        include_embeddings=False,
        include_observables=True,
    )
    architectures = {
        "hybrid": hybrid,
        "embeddings_only": embeddings_only,
        "observables_only": observables_only,
    }
    simplicity_priority = {
        "embeddings_only": 0.0,
        "hybrid": 1.0,
        "observables_only": 2.0,
    }

    def selection_key(name: str) -> tuple[float, ...]:
        validation_metrics = architectures[name][2]
        safe = float(
            validation_metrics["selective_precision"] >= 0.9
            and validation_metrics["neutral_false_activation_rate"] == 0.0
        )
        return (
            safe,
            float(validation_metrics["positive_macro_recall"]),
            float(validation_metrics["selective_precision"]),
            float(validation_metrics["exact_match"]),
            simplicity_priority[name],
        )

    selected_architecture = max(architectures, key=selection_key)
    parameters, weights, validation_metrics, holdout_metrics = architectures[
        selected_architecture
    ]
    embeddings_validation, embeddings_holdout = embeddings_only[2:]
    observables_validation, observables_holdout = observables_only[2:]
    hybrid_validation, hybrid_holdout = hybrid[2:]
    prototype_predictions = [classify_step_behavior(item.text).label for item in holdout]
    metadata = {
        "schema_version": 1,
        "model": "natural-observable-behavior-linear-head-v2",
        "selected_architecture": selected_architecture,
        "architecture_selection_source": "validation_only",
        "embedding_space_id": embedder.space_id,
        "classes": list(CLASSES),
        "observable_features": list(FEATURE_NAMES),
        "parameters": asdict(parameters),
        "project_splits": {
            "calibration": sorted(CALIBRATION_FAMILIES),
            "validation": sorted(VALIDATION_FAMILIES),
            "holdout": sorted(HOLDOUT_FAMILIES),
        },
        "corpus_sha256": {
            "calibration": corpus_sha256(calibration),
            "validation": corpus_sha256(validation),
            "holdout": corpus_sha256(holdout),
        },
    }
    if artifact is not None:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            artifact,
            weights=weights.astype(np.float32),
            metadata=np.frombuffer(json.dumps(metadata, sort_keys=True).encode(), dtype=np.uint8),
        )
    return {
        **metadata,
        "examples": {
            "calibration": len(calibration),
            "validation": len(validation),
            "holdout": len(holdout),
        },
        "validation": validation_metrics,
        "holdout": holdout_metrics,
        "ablations": {
            "hybrid": {
                "validation": hybrid_validation,
                "holdout": hybrid_holdout,
            },
            "embeddings_only": {
                "validation": embeddings_validation,
                "holdout": embeddings_holdout,
            },
            "observables_only": {
                "validation": observables_validation,
                "holdout": observables_holdout,
            },
        },
        "prototype_holdout": behavior_metrics(holdout, prototype_predictions),
        "deployment_recommendation": "shadow_only",
        "deployment_blockers": [
            "holdout has only two project families and fifteen windows",
            "holdout contains no natural retry_loop positive",
        ],
        "artifact_bytes": artifact.stat().st_size if artifact else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("historical", type=Path)
    parser.add_argument("holdout", type=Path)
    parser.add_argument("--artifact", type=Path)
    args = parser.parse_args()
    print(json.dumps(
        run_experiment(args.historical, args.holdout, args.artifact),
        indent=2,
        sort_keys=True,
    ))


if __name__ == "__main__":
    main()
