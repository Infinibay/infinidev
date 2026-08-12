"""Contrastive task-policy retrieval over the bundled static Qwen3 space."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

import numpy as np

from infinidev.engine.task_policies.registry import POLICIES, TaskPolicy
from infinidev.engine.task_policies.semantic_prototypes import (
    PROTOTYPES,
    PROTOTYPE_SET_VERSION,
)
from infinidev.tools.base.static_qwen3_embedder import (
    STATIC_QWEN3_MODEL,
    get_static_qwen3_embedder,
)

logger = logging.getLogger(__name__)

SEMANTIC_CLASSIFIER_VERSION = f"static-qwen3-contrastive-v2:{PROTOTYPE_SET_VERSION}"
_NEGATIVE_VETO_TOLERANCE = 0.02


@dataclass(frozen=True)
class SemanticPolicyCandidate:
    """One embedding candidate with enough evidence to audit the decision."""

    policy: TaskPolicy
    score: float
    runner_up_margin: float
    negative_margin: float | None
    evidence: str
    space_id: str


@dataclass(frozen=True)
class SemanticRetrieval:
    """Result of one local retrieval, including explicit abstention metadata."""

    candidates: tuple[SemanticPolicyCandidate, ...]
    space_id: str | None
    classifier_version: str
    abstained: bool
    reason: str = ""


@dataclass(frozen=True)
class _PrototypeIndex:
    vectors: np.ndarray
    owners: tuple[tuple[TaskPolicy, bool, int], ...]
    space_id: str


@lru_cache(maxsize=1)
def _prototype_index() -> _PrototypeIndex | None:
    """Build a small, process-local index in the exact bundled vector space."""
    embedder = get_static_qwen3_embedder()
    if embedder is None:
        return None
    if embedder.model_name != STATIC_QWEN3_MODEL:
        logger.warning(
            "Task-policy retrieval requires %s; found %s",
            STATIC_QWEN3_MODEL,
            embedder.model_name,
        )
        return None

    texts: list[str] = []
    owners: list[tuple[TaskPolicy, bool, int]] = []
    for policy in POLICIES:
        if not policy.operations:
            continue
        prototypes = PROTOTYPES.get(policy.id)
        if prototypes is None:
            raise ValueError(f"missing semantic prototypes for policy {policy.id}")
        for index, example in enumerate(prototypes.positive):
            texts.append(example)
            owners.append((policy, True, index))
        for index, example in enumerate(prototypes.negative):
            texts.append(example)
            owners.append((policy, False, index))
    vectors = np.asarray(embedder.embed_passages(texts), dtype=np.float32)
    return _PrototypeIndex(
        vectors=vectors,
        owners=tuple(owners),
        space_id=embedder.space_id,
    )


def retrieve_policy_candidates(
    text: str,
    *,
    min_score: float,
    min_margin: float,
) -> SemanticRetrieval:
    """Retrieve at most one high-confidence method policy, otherwise abstain.

    This intentionally behaves like a selective classifier rather than a
    universal cosine threshold. Positive prototypes nominate a policy,
    counterexamples can veto it, and a close runner-up forces abstention. The
    caller remains solely responsible for literal authority and negation.
    """
    try:
        prototype_index = _prototype_index()
        embedder = get_static_qwen3_embedder()
        if prototype_index is None or embedder is None:
            return SemanticRetrieval(
                candidates=(), space_id=None,
                classifier_version=SEMANTIC_CLASSIFIER_VERSION,
                abstained=True, reason="bundled embedding artifact unavailable",
            )
        query = np.asarray(embedder.embed_query(text), dtype=np.float32)
        if not np.any(query):
            return SemanticRetrieval(
                candidates=(), space_id=prototype_index.space_id,
                classifier_version=SEMANTIC_CLASSIFIER_VERSION,
                abstained=True, reason="empty query embedding",
            )
        similarities = prototype_index.vectors @ query
    except Exception:
        logger.debug("Task-policy semantic retrieval failed", exc_info=True)
        return SemanticRetrieval(
            candidates=(), space_id=None,
            classifier_version=SEMANTIC_CLASSIFIER_VERSION,
            abstained=True, reason="embedding backend failed",
        )

    by_policy: dict[str, tuple[TaskPolicy, float, int, float | None]] = {}
    for policy in POLICIES:
        positive = [
            (float(similarities[index]), example_index)
            for index, (owner, is_positive, example_index) in enumerate(
                prototype_index.owners
            )
            if owner.id == policy.id and is_positive
        ]
        if not positive:
            continue
        best_score, best_index = max(positive)
        negatives = [
            float(similarities[index])
            for index, (owner, is_positive, _) in enumerate(prototype_index.owners)
            if owner.id == policy.id and not is_positive
        ]
        best_negative = max(negatives) if negatives else None
        by_policy[policy.id] = (policy, best_score, best_index, best_negative)

    ranked = sorted(by_policy.values(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return SemanticRetrieval(
            candidates=(), space_id=prototype_index.space_id,
            classifier_version=SEMANTIC_CLASSIFIER_VERSION,
            abstained=True, reason="no operation policy prototypes",
        )
    policy, score, example_index, best_negative = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    runner_up_margin = score - runner_up
    negative_margin = score - best_negative if best_negative is not None else None

    if score < min_score:
        reason = f"top score {score:.3f} below {min_score:.3f}"
    elif runner_up_margin < min_margin:
        reason = f"runner-up margin {runner_up_margin:.3f} below {min_margin:.3f}"
    elif negative_margin is not None and negative_margin < -_NEGATIVE_VETO_TOLERANCE:
        reason = f"hard negative outranked positive by {-negative_margin:.3f}"
    else:
        candidate = SemanticPolicyCandidate(
            policy=policy,
            score=score,
            runner_up_margin=runner_up_margin,
            negative_margin=negative_margin,
            evidence=f"positive-example:{policy.id}:{example_index}",
            space_id=prototype_index.space_id,
        )
        return SemanticRetrieval(
            candidates=(candidate,), space_id=prototype_index.space_id,
            classifier_version=SEMANTIC_CLASSIFIER_VERSION,
            abstained=False,
        )

    return SemanticRetrieval(
        candidates=(), space_id=prototype_index.space_id,
        classifier_version=SEMANTIC_CLASSIFIER_VERSION,
        abstained=True, reason=reason,
    )


def clear_semantic_index_cache() -> None:
    """Clear prototype vectors for tests that swap the bundled backend."""
    _prototype_index.cache_clear()


__all__ = [
    "SEMANTIC_CLASSIFIER_VERSION",
    "SemanticPolicyCandidate",
    "SemanticRetrieval",
    "clear_semantic_index_cache",
    "retrieve_policy_candidates",
]
