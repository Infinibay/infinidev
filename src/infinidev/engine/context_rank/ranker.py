"""ContextRanker — scores files, symbols, and findings by relevance.

Combines two signals:

* **Reactive** — tool calls from the current session, weighted by
  recency (exponential decay per iteration) and frequency (log boost).

* **Predictive** — embedding similarity between the current user input
  and historical context messages (task inputs, step titles, step
  descriptions).  Matched contexts propagate their linked interaction
  scores to the current ranking, weighted by cosine similarity and a
  per-level multiplier (*escalera*).

The two signals are blended via an adaptive alpha that starts at 0
(pure prediction on iteration 0) and rises toward 0.85 as the session
accumulates its own reactive signal.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from infinidev.code_intel._db import execute_with_retry
from infinidev.config.settings import settings
from infinidev.engine.context_rank.models import ContextRankResult, RankedItem

logger = logging.getLogger(__name__)

# ── Escalera level weights ───────────────────────────────────────────
# Higher = more specific match contributes more to predicted score.
_LEVEL_WEIGHTS: dict[str, float] = {
    "task_input": 1.0,
    "step_title": 1.5,
    "step_description": 2.0,
}


# ── Public API ───────────────────────────────────────────────────────

def rank(
    current_input: str,
    session_id: str,
    task_id: str,
    iteration: int,
    *,
    top_k_files: int | None = None,
    top_k_symbols: int | None = None,
    top_k_findings: int | None = None,
    cached_embedding: bytes | None = None,
) -> ContextRankResult:
    """Compute ranked resources for prompt injection.

    Returns a :class:`ContextRankResult` with the top-k files, symbols,
    and findings ordered by combined score.

    Pass *cached_embedding* to skip re-embedding the query (~267ms
    saved). The hooks layer caches this from the initial task input.
    """
    if top_k_files is None:
        top_k_files = settings.CONTEXT_RANK_TOP_K_FILES
    if top_k_symbols is None:
        top_k_symbols = settings.CONTEXT_RANK_TOP_K_SYMBOLS
    if top_k_findings is None:
        top_k_findings = settings.CONTEXT_RANK_TOP_K_FINDINGS

    reactive = _compute_reactive_scores(session_id, task_id, iteration)
    predictive = _compute_predictive_scores(current_input, session_id, cached_embedding=cached_embedding)

    alpha = _compute_alpha(iteration, len(reactive))

    # Merge scores
    all_targets = set(reactive) | set(predictive)
    combined: dict[str, tuple[float, str, str]] = {}  # target → (score, target_type, reason)
    for key in all_targets:
        r_score, r_type, r_reason = reactive.get(key, (0.0, "", ""))
        p_score, p_type, p_reason = predictive.get(key, (0.0, "", ""))
        target_type = r_type or p_type
        score = alpha * r_score + (1 - alpha) * p_score
        # Pick the more informative reason
        if r_score > p_score:
            reason = r_reason
        elif p_reason:
            reason = p_reason
        else:
            reason = r_reason
        combined[key] = (score, target_type, reason)

    return ContextRankResult(
        files=_top_k(combined, "file", top_k_files),
        symbols=_top_k(combined, "symbol", top_k_symbols),
        findings=_top_k(combined, "finding", top_k_findings),
    )


# ── Reactive scoring ────────────────────────────────────────────────

def _compute_reactive_scores(
    session_id: str, task_id: str, current_iteration: int,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes based on tool calls in the current session.

    Returns ``{target: (score, target_type, reason)}``.
    """
    decay_lambda = settings.CONTEXT_RANK_REACTIVE_DECAY

    try:
        def _query(conn):
            return conn.execute(
                "SELECT target, target_type, iteration, weight "
                "FROM cr_interactions "
                "WHERE session_id = ? AND task_id = ?",
                (session_id, task_id),
            ).fetchall()
        rows = execute_with_retry(_query)
    except Exception:
        logger.debug("Reactive scoring query failed", exc_info=True)
        return {}

    # Aggregate per target
    accum: dict[str, dict[str, Any]] = {}
    for row in rows:
        target = row["target"]
        if target not in accum:
            accum[target] = {
                "target_type": row["target_type"],
                "weighted_sum": 0.0,
                "count": 0,
            }
        delta = current_iteration - row["iteration"]
        decay = math.exp(-decay_lambda * max(delta, 0))
        accum[target]["weighted_sum"] += row["weight"] * decay
        accum[target]["count"] += 1

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        freq_boost = math.log(1 + info["count"])
        score = info["weighted_sum"] * freq_boost
        reason = f"accessed {info['count']}x this session"
        result[target] = (score, info["target_type"], reason)
    return result


# ── Predictive scoring (escalera) ───────────────────────────────────

def _compute_predictive_scores(
    current_input: str, exclude_session: str,
    *, cached_embedding: bytes | None = None,
) -> dict[str, tuple[float, str, str]]:
    """Score nodes based on similarity to historical contexts.

    Embeds *current_input* and compares against stored context
    embeddings across all three escalera levels.  Matching contexts
    propagate their linked interaction scores weighted by cosine
    similarity and level weight.

    Pass *cached_embedding* (raw float32 bytes) to skip the ~267ms
    embedding computation.
    """
    from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob
    from infinidev.tools.base.dedup import _cosine_similarity

    min_sim = settings.CONTEXT_RANK_MIN_SIMILARITY
    session_decay = settings.CONTEXT_RANK_SESSION_DECAY

    query_emb_bytes = cached_embedding or compute_embedding(current_input)
    if query_emb_bytes is None:
        return {}
    query_vec = np.frombuffer(query_emb_bytes, dtype=np.float32)

    try:
        def _fetch_contexts(conn):
            return conn.execute(
                "SELECT id, session_id, context_type, embedding "
                "FROM cr_contexts "
                "WHERE embedding IS NOT NULL AND session_id != ? "
                "ORDER BY created_at DESC LIMIT 500",
                (exclude_session,),
            ).fetchall()
        ctx_rows = execute_with_retry(_fetch_contexts)
    except Exception:
        logger.debug("Predictive scoring context query failed", exc_info=True)
        return {}

    if not ctx_rows:
        return {}

    # Compute similarity for each context and collect matching ones
    matched_contexts: list[tuple[int, float, float]] = []  # (id, contribution, session_decay)
    session_order: dict[str, int] = {}
    order_counter = 0
    for row in ctx_rows:
        sid = row["session_id"]
        if sid not in session_order:
            session_order[sid] = order_counter
            order_counter += 1

        ctx_vec = embedding_from_blob(row["embedding"])
        sim = float(_cosine_similarity(query_vec, ctx_vec))
        if sim < min_sim:
            continue

        level_weight = _LEVEL_WEIGHTS.get(row["context_type"], 1.0)
        s_decay = session_decay ** session_order[sid]
        contribution = sim * level_weight * s_decay
        matched_contexts.append((row["id"], contribution, sim))

    if not matched_contexts:
        return {}

    # Fetch interactions linked to matched contexts
    context_ids = [m[0] for m in matched_contexts]
    contrib_by_id = {m[0]: (m[1], m[2]) for m in matched_contexts}

    try:
        def _fetch_interactions(conn):
            placeholders = ",".join("?" * len(context_ids))
            return conn.execute(
                f"SELECT context_id, target, target_type, weight "
                f"FROM cr_interactions "
                f"WHERE context_id IN ({placeholders})",
                context_ids,
            ).fetchall()
        int_rows = execute_with_retry(_fetch_interactions)
    except Exception:
        logger.debug("Predictive scoring interaction query failed", exc_info=True)
        return {}

    # Aggregate scores
    accum: dict[str, dict[str, Any]] = {}
    for row in int_rows:
        ctx_id = row["context_id"]
        contribution, sim = contrib_by_id[ctx_id]
        target = row["target"]
        if target not in accum:
            accum[target] = {
                "target_type": row["target_type"],
                "score": 0.0,
                "best_sim": 0.0,
            }
        accum[target]["score"] += contribution * row["weight"]
        accum[target]["best_sim"] = max(accum[target]["best_sim"], sim)

    result: dict[str, tuple[float, str, str]] = {}
    for target, info in accum.items():
        reason = f"predicted (similarity={info['best_sim']:.2f} to past contexts)"
        result[target] = (info["score"], info["target_type"], reason)
    return result


# ── Helpers ──────────────────────────────────────────────────────────

def _compute_alpha(iteration: int, reactive_signal_count: int) -> float:
    """Adaptive blend factor: 0 = pure prediction, 1 = pure reactive."""
    base_alpha = min(0.85, iteration / 8)
    if reactive_signal_count < 3:
        base_alpha *= 0.5
    return base_alpha


def _top_k(
    combined: dict[str, tuple[float, str, str]],
    target_type: str,
    k: int,
) -> list[RankedItem]:
    """Select the top *k* items of a given target type."""
    filtered = [
        (target, score, reason)
        for target, (score, tt, reason) in combined.items()
        if tt == target_type and score > 0
    ]
    filtered.sort(key=lambda x: x[1], reverse=True)
    return [
        RankedItem(target=t, target_type=target_type, score=s, reason=r)
        for t, s, r in filtered[:k]
    ]
