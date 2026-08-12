"""Tiny Qwen-embedding head for provider-exposed reasoning windows."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
import logging

import numpy as np

from infinidev.tools.base.static_qwen3_embedder import get_static_qwen3_embedder


logger = logging.getLogger(__name__)
CLASSIFIER_VERSION = "static-qwen3-reasoning-linear-head-v1"
LABELS = (
    "excessive_exploration",
    "retry_loop",
    "premature_completion",
    "speculative_claim",
    "verification_gap",
    "healthy_progress",
    "uncategorized",
)
FEATURE_NAMES = (
    "modifying_task",
    "discovery_pressure",
    "edit_seen",
    "test_seen",
    "failure_pressure",
    "repeat_pressure",
    "required_work_pending",
    "completion_attempt",
    "evidence_seen",
)
_ARTIFACT = "artifacts/reasoning_pattern_head_v1.npz"
_WINDOW_CHARS = 1_600
_WINDOW_OVERLAP = 240
_MAX_WINDOWS = 4


@dataclass(frozen=True)
class ReasoningFeatures:
    """Small observable state vector accompanying a reasoning window."""

    modifying_task: float = 0.0
    discovery_pressure: float = 0.0
    edit_seen: float = 0.0
    test_seen: float = 0.0
    failure_pressure: float = 0.0
    repeat_pressure: float = 0.0
    required_work_pending: float = 0.0
    completion_attempt: float = 0.0
    evidence_seen: float = 0.0

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [getattr(self, name) for name in FEATURE_NAMES], dtype=np.float32
        )


@dataclass(frozen=True)
class ReasoningPatternResult:
    """Selective prediction for one or more visible-reasoning windows."""

    label: str | None
    score: float
    threshold: float
    runner_up_margin: float
    window: str
    space_id: str | None
    classifier_version: str = CLASSIFIER_VERSION
    abstention_reason: str = ""


@dataclass(frozen=True)
class _Head:
    weights: np.ndarray
    thresholds: np.ndarray
    margin: float
    space_id: str


@lru_cache(maxsize=1)
def _load_head() -> _Head:
    resource = files("infinidev.engine.behavior").joinpath(_ARTIFACT)
    with resource.open("rb") as handle, np.load(handle, allow_pickle=False) as payload:
        weights = np.asarray(payload["weights"], dtype=np.float32)
        metadata = json.loads(payload["metadata"].tobytes())
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        raise RuntimeError("bundled static Qwen3 artifact is unavailable")
    if metadata.get("schema_version") != 1 or metadata.get("model") != CLASSIFIER_VERSION:
        raise ValueError("reasoning mini-head metadata is incompatible")
    if tuple(metadata.get("labels") or ()) != LABELS:
        raise ValueError("reasoning mini-head label order changed")
    if tuple(metadata.get("observable_features") or ()) != FEATURE_NAMES:
        raise ValueError("reasoning mini-head feature order changed")
    if metadata.get("embedding_space_id") != embedder.space_id:
        raise ValueError("reasoning mini-head embedding space does not match runtime")
    expected_shape = (embedder.dim + len(FEATURE_NAMES) + 1, len(LABELS))
    if weights.shape != expected_shape:
        raise ValueError(
            f"reasoning mini-head weights have shape {weights.shape}, expected {expected_shape}"
        )
    parameters = metadata.get("parameters") or {}
    thresholds = np.asarray(parameters.get("thresholds") or (), dtype=np.float32)
    if thresholds.shape != (len(LABELS),) or not np.all(np.isfinite(thresholds)):
        raise ValueError("reasoning mini-head thresholds are invalid")
    return _Head(
        weights=weights,
        thresholds=thresholds,
        margin=float(parameters.get("margin") or 0.0),
        space_id=embedder.space_id,
    )


def _windows(text: str) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= _WINDOW_CHARS:
        return [normalized] if normalized else []
    stride = _WINDOW_CHARS - _WINDOW_OVERLAP
    starts = list(range(0, max(1, len(normalized) - _WINDOW_OVERLAP), stride))
    if len(starts) > _MAX_WINDOWS:
        indices = np.linspace(0, len(starts) - 1, _MAX_WINDOWS, dtype=int)
        starts = [starts[int(index)] for index in indices]
    return [normalized[start : start + _WINDOW_CHARS] for start in starts]


def classify_reasoning(
    text: str,
    features: ReasoningFeatures,
) -> ReasoningPatternResult:
    """Classify bounded windows and abstain when no class clears its gate."""
    windows = _windows(text)
    if not windows:
        return ReasoningPatternResult(
            label=None,
            score=0.0,
            threshold=0.0,
            runner_up_margin=0.0,
            window="",
            space_id=None,
            abstention_reason="visible reasoning is empty",
        )
    try:
        head = _load_head()
        embedder = get_static_qwen3_embedder()
        if embedder is None:
            raise RuntimeError("bundled static Qwen3 artifact is unavailable")
        embeddings = np.asarray(embedder.embed_queries(windows), dtype=np.float32)
        feature_rows = np.repeat(features.as_array()[None, :], len(windows), axis=0)
        design = np.column_stack(
            (embeddings, feature_rows, np.ones(len(windows), dtype=np.float32))
        )
        scores = design @ head.weights
    except Exception as exc:
        logger.debug("Reasoning mini-head classification failed", exc_info=True)
        return ReasoningPatternResult(
            label=None,
            score=0.0,
            threshold=0.0,
            runner_up_margin=0.0,
            window="",
            space_id=None,
            abstention_reason=str(exc),
        )

    candidates: list[tuple[float, int, int, float]] = []
    for window_index, row in enumerate(scores):
        order = np.argsort(row)[::-1]
        top, runner_up = int(order[0]), int(order[1])
        margin = float(row[top] - row[runner_up])
        excess = float(row[top] - head.thresholds[top])
        if excess >= 0.0 and margin >= head.margin:
            candidates.append((excess, window_index, top, margin))
    if not candidates:
        flat_index = int(np.argmax(scores))
        window_index, top = np.unravel_index(flat_index, scores.shape)
        return ReasoningPatternResult(
            label=None,
            score=float(scores[window_index, top]),
            threshold=float(head.thresholds[top]),
            runner_up_margin=float(
                scores[window_index, top]
                - np.partition(scores[window_index], -2)[-2]
            ),
            window=windows[window_index],
            space_id=head.space_id,
            abstention_reason="no window cleared its class threshold and margin",
        )
    _, window_index, top, margin = max(candidates)
    return ReasoningPatternResult(
        label=LABELS[top],
        score=float(scores[window_index, top]),
        threshold=float(head.thresholds[top]),
        runner_up_margin=margin,
        window=windows[window_index],
        space_id=head.space_id,
    )


def clear_reasoning_classifier_cache() -> None:
    """Reset the packaged head for focused tests."""
    _load_head.cache_clear()


__all__ = [
    "CLASSIFIER_VERSION",
    "FEATURE_NAMES",
    "LABELS",
    "ReasoningFeatures",
    "ReasoningPatternResult",
    "classify_reasoning",
    "clear_reasoning_classifier_cache",
]
