"""Pre-computed embedding helpers for findings and wiki pages.

Stores embeddings as BLOB (numpy float32 bytes) in the DB so semantic
search only needs to embed the query — not all candidates.
"""

from __future__ import annotations

import logging
import sqlite3

import numpy as np

logger = logging.getLogger(__name__)


def compute_embedding(text: str) -> bytes | None:
    """Embed *text* and return raw float32 bytes, or None on failure."""
    try:
        from infinidev.tools.base.dedup import _get_embed_fn

        embed_fn = _get_embed_fn()
        vectors = embed_fn([text])
        arr = np.asarray(vectors[0], dtype=np.float32)
        return arr.tobytes()
    except Exception:
        logger.debug("compute_embedding failed", exc_info=True)
        return None


def embedding_from_blob(blob: bytes | memoryview) -> np.ndarray:
    """Deserialize a BLOB back to a numpy float32 vector."""
    return np.frombuffer(bytes(blob), dtype=np.float32)


def store_finding_embedding(conn: sqlite3.Connection, finding_id: int, text: str) -> None:
    """Compute and store an embedding for a finding (topic + content prefix)."""
    emb = compute_embedding(text)
    if emb is not None:
        conn.execute(
            "UPDATE findings SET embedding = ? WHERE id = ?",
            (emb, finding_id),
        )


def store_wiki_embedding(conn: sqlite3.Connection, page_id: int, text: str) -> None:
    """Compute and store an embedding for a wiki page (title + content prefix)."""
    emb = compute_embedding(text)
    if emb is not None:
        conn.execute(
            "UPDATE wiki_pages SET embedding = ? WHERE id = ?",
            (emb, page_id),
        )
