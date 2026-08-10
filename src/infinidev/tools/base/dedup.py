"""Semantic duplicate detection for epics, milestones, and tasks.

Uses Infinidev's shared embedding backend to embed titles and compute cosine
similarity.  The primary backend is the bundled ``ken/static-qwen3-r512-v2``
table; legacy MNN/Chroma MiniLM paths remain compatibility fallbacks.
"""

from __future__ import annotations

import logging

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

logger = logging.getLogger(__name__)

# Module-level singleton — matches the pattern in backend/tools/rag/base.py.
_embed_fn = None


def _get_embed_fn():
    """Return the active embedder.

    Prefers the static Qwen3 table: it is bundled, multilingual, offline, and
    requires only table lookup plus one projection.  The old MiniLM backends
    remain available so stripped source packages and explicit legacy installs
    fail soft rather than losing semantic features entirely.
    """
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
    try:
        from infinidev.tools.base.static_qwen3_embedder import (
            get_static_qwen3_embedder,
        )

        static = get_static_qwen3_embedder()
        if static is not None:
            _embed_fn = static
            return _embed_fn
    except Exception:
        logger.debug("Static Qwen3 embedder probe failed; trying MiniLM", exc_info=True)
    try:
        from infinidev.tools.base.mnn_embedder import get_mnn_embedder
        mnn = get_mnn_embedder()
        if mnn is not None:
            _embed_fn = mnn
            return _embed_fn
    except Exception:
        logger.debug("MNN embedder probe failed; using ChromaDB", exc_info=True)
    _embed_fn = DefaultEmbeddingFunction()
    return _embed_fn


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D vectors."""
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def find_semantic_duplicate(
    new_title: str,
    existing_items: list[dict],
    threshold: float = 0.82,
) -> dict | None:
    """Return the best matching existing item if above *threshold*, else None.

    Parameters
    ----------
    new_title:
        The title of the item about to be created.
    existing_items:
        ``[{"id": int, "title": str}, ...]`` — the items already in the DB
        for the relevant scope (project / epic).
    threshold:
        Cosine-similarity cutoff.  0.82 is high enough to avoid false
        positives on genuinely different titles, low enough to catch
        rephrasings like "Design System Architecture" vs
        "System Architecture Design".

    Returns
    -------
    ``{"id": int, "title": str, "similarity": float}`` when a duplicate is
    found, otherwise ``None``.
    """
    # Strip once and use the stripped value for BOTH the guard/length check and
    # the embedding, so incidental whitespace doesn't shift the vector away from
    # an equivalent already-stored (stripped) title and let a dup slip through.
    new_title = new_title.strip()
    if not existing_items or not new_title:
        return None

    # Titles under 10 characters are too short for meaningful semantic
    # comparison — they lack enough signal to distinguish (e.g. "Task 1"
    # vs "Task 2" would false-positive).  Real titles are always longer.
    if len(new_title) < 10:
        return None

    embed_fn = _get_embed_fn()

    existing_titles = [item["title"] for item in existing_items]

    try:
        query_method = getattr(embed_fn, "embed_queries", None)
        passage_method = getattr(embed_fn, "embed_passages", None)
        new_embedding = (
            query_method([new_title])[0]
            if callable(query_method)
            else embed_fn([new_title])[0]
        )
        embeddings = (
            passage_method(existing_titles)
            if callable(passage_method)
            else embed_fn(existing_titles)
        )
    except Exception:
        logger.warning("Embedding failed during dedup check; skipping", exc_info=True)
        return None

    new_vec = np.asarray(new_embedding)
    best_sim = -1.0
    best_idx = -1

    for i, emb in enumerate(embeddings):
        sim = _cosine_similarity(new_vec, np.asarray(emb))
        if sim > best_sim:
            best_sim = sim
            best_idx = i

    if best_sim >= threshold:
        match = existing_items[best_idx]
        result = dict(match)
        result["similarity"] = best_sim
        return result
    return None
