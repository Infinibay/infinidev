"""Pre-computed embedding helpers for findings and wiki pages.

Stores embeddings as BLOB (numpy float32 bytes) in the DB so semantic
search only needs to embed the query — not all candidates.
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)

_SPACE_PROBE = "Hola mundo — Infinidev embedding space probe / prueba de codificación"


def embed_passages(texts: list[str]) -> list[np.ndarray]:
    """Embed stored/indexed texts in the stable passage space."""
    from infinidev.tools.base.dedup import _get_embed_fn

    embedder = _get_embed_fn()
    method = getattr(embedder, "embed_passages", None)
    return method(texts) if callable(method) else embedder(texts)


def embed_queries(texts: list[str]) -> list[np.ndarray]:
    """Embed search queries, allowing backend-specific query adaptation."""
    from infinidev.tools.base.dedup import _get_embed_fn

    embedder = _get_embed_fn()
    method = getattr(embedder, "embed_queries", None)
    return method(texts) if callable(method) else embedder(texts)


def compute_embedding(text: str) -> bytes | None:
    """Embed stored passage *text* and return float32 bytes, or None."""
    try:
        vectors = embed_passages([text])
        arr = np.asarray(vectors[0], dtype=np.float32)
        return arr.tobytes()
    except Exception:
        logger.debug("compute_embedding failed", exc_info=True)
        return None


def compute_query_embedding(text: str) -> bytes | None:
    """Embed query *text* with calibrated query-side adaptation."""
    try:
        vectors = embed_queries([text])
        return np.asarray(vectors[0], dtype=np.float32).tobytes()
    except Exception:
        logger.debug("compute_query_embedding failed", exc_info=True)
        return None


def current_embedding_space() -> str:
    """Return an identity that changes whenever the live vector space changes."""
    from infinidev.tools.base.dedup import _get_embed_fn

    embed_fn = _get_embed_fn()
    declared = getattr(embed_fn, "space_id", None)
    if declared:
        return str(declared)
    vector = np.asarray(embed_fn([_SPACE_PROBE])[0], dtype=np.float32)
    digest = hashlib.sha256(vector.tobytes()).hexdigest()[:16]
    model = getattr(
        embed_fn,
        "model_name",
        f"{type(embed_fn).__module__}.{type(embed_fn).__qualname__}",
    )
    return f"{model}:{vector.size}:{digest}"


def embedding_from_blob(blob: bytes | memoryview) -> np.ndarray:
    """Deserialize a BLOB back to a numpy float32 vector."""
    return np.frombuffer(bytes(blob), dtype=np.float32)


def embedding_is_current(
    blob: bytes | memoryview | None,
    stored_space: str | None,
    *,
    live_space: str,
    dim: int,
) -> bool:
    """Whether a stored vector is safe to compare with the live embedder."""
    if blob is None or stored_space != live_space:
        return False
    try:
        return embedding_from_blob(blob).shape == (dim,)
    except ValueError:
        return False


def stack_compatible_embeddings(
    blobs: list[bytes | memoryview | None],
    *,
    dim: int,
) -> tuple[np.ndarray, list[int]]:
    """Stack only stored vectors belonging to the live embedding space.

    Infinidev historically stored raw 384-dimensional MiniLM vectors.  The
    static Qwen3 backend emits 1024 dimensions, so an existing database can
    legitimately contain both generations while it is incrementally refreshed.
    Cosine between those spaces is meaningless and ``numpy.stack`` would fail
    on the mixed shapes.  Returning the kept positions lets callers preserve
    row alignment while safely ignoring stale derived data.
    """
    vectors: list[np.ndarray] = []
    kept: list[int] = []
    for index, blob in enumerate(blobs):
        if blob is None:
            continue
        vector = embedding_from_blob(blob)
        if vector.shape != (dim,):
            continue
        vectors.append(vector)
        kept.append(index)
    if not vectors:
        return np.zeros((0, dim), dtype=np.float32), []
    return np.asarray(vectors, dtype=np.float32), kept


def store_finding_embedding(conn: sqlite3.Connection, finding_id: int, text: str) -> None:
    """Compute and store an embedding for a finding (topic + content prefix)."""
    emb = compute_embedding(text)
    if emb is not None:
        conn.execute(
            "UPDATE findings SET embedding = ?, embedding_space = ? WHERE id = ?",
            (emb, current_embedding_space(), finding_id),
        )


def store_wiki_embedding(conn: sqlite3.Connection, page_id: int, text: str) -> None:
    """Compute and store an embedding for a wiki page (title + content prefix)."""
    emb = compute_embedding(text)
    if emb is not None:
        conn.execute(
            "UPDATE wiki_pages SET embedding = ? WHERE id = ?",
            (emb, page_id),
        )


def store_context_embedding(conn: sqlite3.Connection, context_id: int, text: str) -> None:
    """Compute and store an embedding for a ContextRank context entry."""
    emb = compute_embedding(text)
    if emb is not None:
        conn.execute(
            "UPDATE cr_contexts SET embedding = ?, embedding_space = ? WHERE id = ?",
            (emb, current_embedding_space(), context_id),
        )
