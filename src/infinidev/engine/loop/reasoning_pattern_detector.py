"""Detect risky reasoning patterns in the principal's ``reasoning_content``.

Reuses the embedding pipeline already in the repo (MNN if configured,
ChromaDB otherwise — see :mod:`infinidev.tools.base.dedup`) so the
incremental cost per turn is ~11 ms with MNN, ~115 ms with the ONNX
default. Both are negligible vs. the principal LLM call.

Catalog source of truth is the ``critic_reasoning_patterns`` table.
The detector loads it once at construction and reuses the in-memory
copy across calls — patterns evolve on the order of releases, not
turns, so a single SQL hit per session is fine. Pass ``reload=True``
to :meth:`detect` if a future refresher needs it.

The detector NEVER raises on the hot path: any failure (DB unavailable,
embedding backend dead, malformed catalog row) returns ``[]`` and logs
at ``debug``. This mirrors the critic's own "never block the loop"
contract — pattern detection is a *signal source*, not a precondition.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from infinidev.config.settings import get_db_path
from infinidev.tools.base.dedup import _cosine_similarity
from infinidev.tools.base.embeddings import compute_embedding, embedding_from_blob

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PatternMatch:
    """A reasoning-pattern hit above its similarity threshold.

    Attributes:
        pattern_name: e.g. ``"victory_lap"``. Stable identifier — the
            engine routes on this.
        signature: The catalog signature that matched (the closest of
            its pattern's signatures). Useful for logs and for the
            optional ``confirmed_useful_count`` learning hook.
        similarity: Cosine similarity in [0, 1] of the principal's
            reasoning vs. the matched signature.
        socratic_question: Pre-curated question for the principal,
            tied to this pattern. The critic may weave it into its
            verdict or skip it depending on context.
        triggered_check: Optional ``CheckCode.value`` (string) the
            engine should add to the active checks for this turn.
            ``None`` means the pattern surfaces only via the socratic
            question.
    """

    pattern_name: str
    signature: str
    similarity: float
    socratic_question: str
    triggered_check: str | None


@dataclass(frozen=True)
class _CatalogEntry:
    signature: str
    embedding: np.ndarray
    threshold: float
    socratic_question: str
    triggered_check: str | None


class ReasoningPatternDetector:
    """Detects pattern matches against the seeded catalog.

    One instance is built per developer-loop run (parallel to the
    :class:`AssistantCritic`). Construction loads the catalog once.
    ``detect()`` is safe to call from any thread — it doesn't mutate
    state.
    """

    def __init__(self, db_path: Path | str | None = None):
        self._db_path = Path(db_path) if db_path else get_db_path()
        # ``dict[pattern_name, list[_CatalogEntry]]``. A pattern with
        # no entries (deleted from the seed but still referenced) maps
        # to an empty list and is silently skipped during detect().
        self._catalog: dict[str, list[_CatalogEntry]] = self._load_catalog()

    @property
    def catalog_size(self) -> int:
        return sum(len(v) for v in self._catalog.values())

    def _load_catalog(self) -> dict[str, list[_CatalogEntry]]:
        if not self._db_path.exists():
            logger.debug("pattern detector: db not found at %s", self._db_path)
            return {}
        catalog: dict[str, list[_CatalogEntry]] = {}
        try:
            conn = sqlite3.connect(self._db_path)
            try:
                rows = conn.execute(
                    "SELECT pattern_name, signature, embedding, threshold, "
                    "socratic_question, triggered_check "
                    "FROM critic_reasoning_patterns"
                ).fetchall()
            finally:
                conn.close()
        except sqlite3.Error as exc:
            logger.debug("pattern detector: catalog load failed: %s", exc)
            return {}

        for name, sig, emb_blob, threshold, question, check in rows:
            if not emb_blob:
                continue
            try:
                vec = embedding_from_blob(emb_blob)
            except Exception:
                logger.debug(
                    "pattern detector: skipping malformed embedding for %r/%r",
                    name, sig,
                )
                continue
            catalog.setdefault(name, []).append(
                _CatalogEntry(
                    signature=sig,
                    embedding=vec,
                    threshold=float(threshold),
                    socratic_question=question,
                    triggered_check=check,
                )
            )
        if not catalog:
            logger.debug("pattern detector: empty catalog (seed not run yet?)")
        return catalog

    def detect(self, reasoning: str | None) -> list[PatternMatch]:
        """Return the best match per pattern that crosses its threshold.

        At most one :class:`PatternMatch` per pattern_name — the closest
        signature wins for that pattern. Different patterns can match
        independently in the same call; the engine merges their
        ``triggered_check`` values into the active-checks set.

        Empty / whitespace-only reasoning, missing catalog, or embedding
        failure all yield ``[]`` silently.
        """
        if not reasoning or not reasoning.strip():
            return []
        if not self._catalog:
            return []

        emb_blob = compute_embedding(reasoning)
        if emb_blob is None:
            return []
        try:
            query = embedding_from_blob(emb_blob)
        except Exception:
            logger.debug("pattern detector: query embedding deserialise failed")
            return []

        hits: list[PatternMatch] = []
        for pattern_name, entries in self._catalog.items():
            best: tuple[float, _CatalogEntry] | None = None
            for entry in entries:
                sim = _cosine_similarity(query, entry.embedding)
                if best is None or sim > best[0]:
                    best = (sim, entry)
            if best is None:
                continue
            sim, entry = best
            if sim >= entry.threshold:
                hits.append(PatternMatch(
                    pattern_name=pattern_name,
                    signature=entry.signature,
                    similarity=sim,
                    socratic_question=entry.socratic_question,
                    triggered_check=entry.triggered_check,
                ))
                logger.info(
                    "pattern_match: %s sim=%.3f signature=%r",
                    pattern_name, sim, entry.signature,
                )
        return hits


def render_pattern_matches_block(matches: list[PatternMatch]) -> str:
    """Render the ``<reasoning-patterns-detected>`` block for the critic.

    Returns an empty string when *matches* is empty so callers can
    concat unconditionally. The block tells the critic *what* was
    detected and *which question to consider weaving in* — but does
    not force the critic to use it. The critic still owns the verdict.
    """
    if not matches:
        return ""
    lines = ["<reasoning-patterns-detected>"]
    lines.append(
        "Detector de patrones encontró las siguientes señales en el "
        "reasoning del principal este turno. Considerá tejer la "
        "pregunta socrática asociada en tu mensaje (si emitís uno). "
        "No la cites textual — adaptala al contexto. Si el reasoning "
        "del principal ya tiene una justificación que considera "
        "satisfactoria, callate ('continue')."
    )
    for m in matches:
        lines.append(
            f"\n- {m.pattern_name} (sim {m.similarity:.2f}, "
            f"matched: \"{m.signature}\")\n"
            f"  Pregunta sugerida: {m.socratic_question}"
        )
    lines.append("</reasoning-patterns-detected>")
    return "\n".join(lines)
