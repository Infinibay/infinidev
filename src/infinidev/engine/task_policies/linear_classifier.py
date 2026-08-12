"""Selective task-method classifier over the bundled static Qwen3 vectors."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import logging

import numpy as np

from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


logger = logging.getLogger(__name__)
CLASSIFIER_VERSION = "static-qwen3-task-policy-hierarchical-head-v4"
LABELS = (
    "bugfix.root_cause",
    "feature.contract_first",
    "refactor.preserve_behavior",
    "research.evidence_first",
    "review.read_only",
    "performance.measure_first",
    "uncategorized",
)
METHOD_LABELS = LABELS[:-1]
_ARTIFACT = "artifacts/task_policy_hierarchical_head_v4.npz"


@dataclass(frozen=True)
class TaskMethodPrediction:
    """One audited prediction, including explicit abstention evidence."""

    policy_id: str | None
    score: float
    threshold: float
    runner_up_margin: float
    space_id: str | None
    classifier_version: str = CLASSIFIER_VERSION
    abstention_reason: str = ""
    candidate_policy_id: str | None = None
    agreement_eligible: bool = False


@dataclass(frozen=True)
class _Head:
    discourse_weights: np.ndarray
    method_weights: np.ndarray
    discourse_threshold: float
    discourse_margin: float
    method_threshold: float
    method_margin: float
    agreement_method_threshold: float
    agreement_method_margin: float
    space_id: str


@lru_cache(maxsize=1)
def _load_head() -> _Head:
    resource = files("infinidev.engine.task_policies").joinpath(_ARTIFACT)
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        discourse_weights = np.asarray(payload["discourse_weights"], dtype=np.float32)
        method_weights = np.asarray(payload["method_weights"], dtype=np.float32)
        metadata = json.loads(payload["metadata"].tobytes())
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    if metadata.get("schema_version") != 2 or metadata.get("model") != CLASSIFIER_VERSION:
        raise ValueError("task-policy mini-head metadata is incompatible")
    if tuple(metadata.get("labels") or ()) != LABELS:
        raise ValueError("task-policy mini-head label order changed")
    if tuple(metadata.get("method_labels") or ()) != METHOD_LABELS:
        raise ValueError("task-policy mini-head method label order changed")
    if tuple(metadata.get("discourse_labels") or ()) != ("uncategorized", "task"):
        raise ValueError("task-policy mini-head discourse label order changed")
    if metadata.get("embedding_space_id") != embedder.space_id:
        raise ValueError("task-policy mini-head embedding space does not match runtime")
    expected_discourse_shape = (embedder.dim + 1, 2)
    expected_method_shape = (embedder.dim + 1, len(METHOD_LABELS))
    if (
        discourse_weights.shape != expected_discourse_shape
        or not np.all(np.isfinite(discourse_weights))
    ):
        raise ValueError(
            "task-policy discourse weights have shape "
            f"{discourse_weights.shape}, expected {expected_discourse_shape}"
        )
    if method_weights.shape != expected_method_shape or not np.all(np.isfinite(method_weights)):
        raise ValueError(
            f"task-policy method weights have shape {method_weights.shape}, "
            f"expected {expected_method_shape}"
        )
    parameters = metadata.get("parameters") or {}
    parameter_names = (
        "discourse_threshold",
        "discourse_margin",
        "method_threshold",
        "method_margin",
        "agreement_method_threshold",
        "agreement_method_margin",
    )
    runtime_parameters = {
        name: float(parameters.get(name, float("nan")))
        for name in parameter_names
    }
    if not all(np.isfinite(value) for value in runtime_parameters.values()):
        raise ValueError("task-policy mini-head thresholds are invalid")
    return _Head(
        discourse_weights=discourse_weights,
        method_weights=method_weights,
        **runtime_parameters,
        space_id=embedder.space_id,
    )


def classify_task_method(text: str) -> TaskMethodPrediction:
    """Predict one task method locally, abstaining on ambiguity or neutral input."""
    normalized = " ".join(text.split())
    if not normalized:
        return TaskMethodPrediction(
            policy_id=None,
            score=0.0,
            threshold=0.0,
            runner_up_margin=0.0,
            space_id=None,
            abstention_reason="request is empty",
        )
    try:
        head = _load_head()
        embedder = get_static_qwen3_embedder()
        if embedder is None:
            raise RuntimeError("bundled static Qwen3 artifact is unavailable")
        vector = np.asarray(embedder.embed_query(normalized), dtype=np.float32)
        design = np.concatenate((vector, np.ones(1, dtype=np.float32)))
        discourse_scores = design @ head.discourse_weights
        method_scores = design @ head.method_weights
    except Exception as exc:
        logger.debug("Task-policy mini-head classification failed", exc_info=True)
        return TaskMethodPrediction(
            policy_id=None,
            score=0.0,
            threshold=0.0,
            runner_up_margin=0.0,
            space_id=None,
            abstention_reason=str(exc),
        )

    discourse_order = np.argsort(discourse_scores)[::-1]
    discourse_top, discourse_runner_up = map(int, discourse_order[:2])
    discourse_score = float(discourse_scores[discourse_top])
    discourse_margin = float(
        discourse_score - discourse_scores[discourse_runner_up]
    )
    if discourse_top == 0:
        reason = "discourse gate selected uncategorized"
    elif discourse_score < head.discourse_threshold:
        reason = (
            f"task score {discourse_score:.3f} below "
            f"{head.discourse_threshold:.3f}"
        )
    elif discourse_margin < head.discourse_margin:
        reason = (
            f"task margin {discourse_margin:.3f} below "
            f"{head.discourse_margin:.3f}"
        )
    else:
        method_order = np.argsort(method_scores)[::-1]
        method_top, method_runner_up = map(int, method_order[:2])
        score = float(method_scores[method_top])
        margin = float(score - method_scores[method_runner_up])
        label = METHOD_LABELS[method_top]
        if score < head.method_threshold:
            reason = f"method score {score:.3f} below {head.method_threshold:.3f}"
        elif margin < head.method_margin:
            reason = f"method margin {margin:.3f} below {head.method_margin:.3f}"
        else:
            return TaskMethodPrediction(
                policy_id=label,
                score=score,
                threshold=head.method_threshold,
                runner_up_margin=margin,
                space_id=head.space_id,
                candidate_policy_id=label,
            )
        return TaskMethodPrediction(
            policy_id=None,
            score=score,
            threshold=head.method_threshold,
            runner_up_margin=margin,
            space_id=head.space_id,
            abstention_reason=reason,
            candidate_policy_id=(
                label
                if score >= head.agreement_method_threshold
                and margin >= head.agreement_method_margin
                else None
            ),
            agreement_eligible=(
                score >= head.agreement_method_threshold
                and margin >= head.agreement_method_margin
            ),
        )
    return TaskMethodPrediction(
        policy_id=None,
        score=discourse_score,
        threshold=head.discourse_threshold,
        runner_up_margin=discourse_margin,
        space_id=head.space_id,
        abstention_reason=reason,
    )


def clear_task_method_classifier_cache() -> None:
    """Reset the packaged head for tests that swap artifacts or embedders."""
    _load_head.cache_clear()


__all__ = [
    "CLASSIFIER_VERSION",
    "LABELS",
    "METHOD_LABELS",
    "TaskMethodPrediction",
    "classify_task_method",
    "clear_task_method_classifier_cache",
]
