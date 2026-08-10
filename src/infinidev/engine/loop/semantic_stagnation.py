"""Embedding-backed stagnation detection with deterministic evidence gates."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from infinidev.engine.loop.action_record import ActionRecord

logger = logging.getLogger(__name__)

# Calibrated on held-out adjacent records plus the 2026-08-09 MiniMax M3
# pytest-dev__pytest-5103 trace. Legitimate transitions peaked at 0.7793;
# the repeated implementation-step sequence scored 0.9195 then 0.8343.
SEMANTIC_STAGNATION_MIN_COSINE = 0.80
SEMANTIC_STAGNATION_WINDOW = 3
SEMANTIC_STAGNATION_STRONG_COSINE = 0.90
SEMANTIC_STAGNATION_STRONG_WINDOW = 2

# A recovery Step gets a tiny, deterministic context allowance before its
# action-only surface closes. Full-file results above the opened-file cache
# limit do not survive a Step summary, so banning every read would strand a
# model with a remembered location but no source body to edit.
SEMANTIC_RECOVERY_CONTEXT_TOOL_NAMES = frozenset({"read_file"})
SEMANTIC_RECOVERY_CONTEXT_CALLS = 2


@dataclass(frozen=True)
class SemanticStagnation:
    """Evidence returned when one Step repeats without workspace progress."""

    step_index: int
    similarities: tuple[float, ...]
    reason: Literal["semantic", "deterministic"] = "semantic"


def detect_semantic_stagnation(
    history: list[ActionRecord],
    *,
    minimum_cosine: float = SEMANTIC_STAGNATION_MIN_COSINE,
    strong_cosine: float = SEMANTIC_STAGNATION_STRONG_COSINE,
) -> SemanticStagnation | None:
    """Detect repeated same-Step paraphrases when hard evidence is unchanged.

    Cosine is never sufficient on its own. Every record must report no net
    workspace transition, and the deterministic test fingerprints must be
    identical across the window. Two records require the strong threshold;
    otherwise three must all clear the ordinary threshold. Any missing
    embedding backend abstains.
    """
    if len(history) < SEMANTIC_STAGNATION_STRONG_WINDOW:
        return None
    records = history[-min(len(history), SEMANTIC_STAGNATION_WINDOW):]
    step_indices = {record.step_index for record in records}
    if len(step_indices) != 1:
        records = history[-SEMANTIC_STAGNATION_STRONG_WINDOW:]
        step_indices = {record.step_index for record in records}
        if len(step_indices) != 1:
            return None
    if any(record.net_workspace_changed for record in records):
        return None
    fingerprints = {record.test_outcome_fingerprints for record in records}
    if len(fingerprints) != 1:
        return None
    summaries = [record.summary.strip() for record in records]
    if any(len(summary) < 40 for summary in summaries):
        return None

    try:
        from infinidev.tools.base.embeddings import embed_passages

        vectors = np.asarray(embed_passages(summaries), dtype=np.float32)
    except Exception:
        logger.debug("semantic stagnation embedding failed", exc_info=True)
        return None
    if vectors.shape[0] != len(summaries) or vectors.ndim != 2:
        return None
    similarities = tuple(
        float(vectors[index - 1] @ vectors[index])
        for index in range(1, len(vectors))
    )
    if len(records) == SEMANTIC_STAGNATION_STRONG_WINDOW:
        if similarities[-1] < strong_cosine:
            return None
    elif similarities[-1] >= strong_cosine:
        records = records[-SEMANTIC_STAGNATION_STRONG_WINDOW:]
        similarities = similarities[-1:]
    elif min(similarities) < minimum_cosine:
        return None
    return SemanticStagnation(
        step_index=records[-1].step_index,
        similarities=similarities,
    )


__all__ = [
    "SEMANTIC_STAGNATION_MIN_COSINE",
    "SEMANTIC_STAGNATION_STRONG_COSINE",
    "SEMANTIC_STAGNATION_STRONG_WINDOW",
    "SEMANTIC_RECOVERY_CONTEXT_CALLS",
    "SEMANTIC_RECOVERY_CONTEXT_TOOL_NAMES",
    "SEMANTIC_STAGNATION_WINDOW",
    "SemanticStagnation",
    "detect_semantic_stagnation",
]
