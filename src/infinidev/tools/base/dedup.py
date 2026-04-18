"""Semantic duplicate detection for epics, milestones, and tasks.

Uses ChromaDB's built-in DefaultEmbeddingFunction (all-MiniLM-L6-v2) to
embed titles and compute cosine similarity.  This avoids coupling to the
RAG pipeline or its settings — the number of titles per project is small
(tens, not thousands) so a transient embed + numpy cosine is simpler than
managing a persistent ChromaDB collection.
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

    Prefers the MNN-backed embedder (~10x faster on CPU, identical output)
    when `INFINIDEV_MNN_MODEL_PATH` is configured and loadable. Falls back
    to ChromaDB's ONNX default otherwise.
    """
    global _embed_fn
    if _embed_fn is not None:
        return _embed_fn
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
    if not existing_items or not new_title.strip():
        return None

    # Titles under 10 characters are too short for meaningful semantic
    # comparison — they lack enough signal to distinguish (e.g. "Task 1"
    # vs "Task 2" would false-positive).  Real titles are always longer.
    if len(new_title.strip()) < 10:
        return None

    embed_fn = _get_embed_fn()

    existing_titles = [item["title"] for item in existing_items]

    # Embed everything in a single batch call (new title + all existing).
    all_texts = [new_title] + existing_titles
    try:
        embeddings = embed_fn(all_texts)
    except Exception:
        logger.warning("Embedding failed during dedup check; skipping", exc_info=True)
        return None

    new_vec = np.asarray(embeddings[0])
    best_sim = -1.0
    best_idx = -1

    for i, emb in enumerate(embeddings[1:]):
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
