"""Embedding-based similarity primitive.

Wraps the existing in-process embedder from
:mod:`infinidev.tools.base.dedup` (used by finding semantic dedup at
0.82 cosine) so repetition detection adds *zero* new dependencies.

An in-process LRU cache avoids re-embedding identical inputs across
steps — reasoning blocks often repeat near-verbatim across iterations.
"""

from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)


def _hash_key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()


@lru_cache(maxsize=256)
def _embed_cached(text_hash: str, text: str) -> tuple[float, ...] | None:
    """Return the embedding for *text*, cached by hash."""
    try:
        from infinidev.tools.base.dedup import _get_embed_fn

        vec = _get_embed_fn()([text])[0]
        return tuple(float(x) for x in np.asarray(vec).ravel())
    except Exception:
        logger.debug("Embedding failed for behavior primitive", exc_info=True)
        return None


def embed(text: str) -> np.ndarray | None:
    """Return a 1-D numpy embedding for *text*, or ``None`` on failure."""
    if not text or not text.strip():
        return None
    # Truncate extreme inputs to keep the embedder fast and cacheable.
    clipped = text[:4000]
    v = _embed_cached(_hash_key(clipped), clipped)
    if v is None:
        return None
    return np.asarray(v, dtype=np.float32)


def cosine_sim(a: str, b: str) -> float:
    """Cosine similarity between two pieces of text, 0..1 range.

    Returns 0.0 if either embedding fails.
    """
    if not a or not b:
        return 0.0
    va = embed(a)
    vb = embed(b)
    if va is None or vb is None:
        return 0.0
    try:
        from infinidev.tools.base.dedup import _cosine_similarity

        return float(_cosine_similarity(va, vb))
    except Exception:
        return 0.0


def max_cosine_sim(text: str, candidates: list[str]) -> float:
    """Return the highest cosine similarity between *text* and any candidate."""
    if not text or not candidates:
        return 0.0
    best = 0.0
    for c in candidates:
        s = cosine_sim(text, c)
        if s > best:
            best = s
    return best
