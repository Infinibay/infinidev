"""Shadow-only semantic classification of completed execution steps."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import logging

import numpy as np

from infinidev.tools.base.static_qwen3_embedder import (
    STATIC_QWEN3_MODEL,
    get_static_qwen3_embedder,
)


logger = logging.getLogger(__name__)
SEMANTIC_BEHAVIOR_CLASSIFIER_VERSION = "static-qwen3-behavior-prototypes-v1"

_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "excessive_exploration": (
        "I kept inspecting more files and planning without editing or running a relevant test.",
        "Seguí buscando y leyendo archivos aunque el objetivo ya estaba localizado.",
        "J'ai prolongé l'exploration sans modification ni vérification concrète.",
        "Continuei investigando o repositório sem transformar a evidência em uma alteração.",
    ),
    "healthy_progress": (
        "Implemented the scoped change and the focused tests passed.",
        "Localicé el problema, hice el cambio mínimo y verifiqué el resultado.",
        "La modification ciblée est terminée avec une preuve de test concrète.",
        "A alteração pequena foi aplicada e a verificação relevante passou.",
    ),
    "premature_completion": (
        "Claimed completion while planned work or acceptance criteria were still pending.",
        "Intenté cerrar la tarea aunque todavía quedaban pasos obligatorios.",
        "J'ai annoncé la fin alors que des éléments requis restaient ouverts.",
        "Marquei como concluído antes de executar todas as etapas necessárias.",
    ),
    "retry_loop": (
        "Repeated the same failed command or edit without changing the approach.",
        "Repetí el intento fallido sin incorporar evidencia nueva ni cambiar la estrategia.",
        "La même commande en échec a été relancée sans correction matérielle.",
        "Repeti a mesma tentativa com erro sem adaptar parâmetros ou hipótese.",
    ),
    "speculative_claim": (
        "Reported a correctness or security conclusion without code or test evidence.",
        "Afirmé una causa o garantía que no estaba respaldada por el código observado.",
        "La conclusion dépasse les preuves disponibles dans le dépôt.",
        "A conclusão foi apresentada como certa sem evidência verificável.",
    ),
    "verification_gap": (
        "Changed implementation code but finished without running the relevant verification.",
        "Edité el código y avancé sin ejecutar una prueba que cubriera el cambio.",
        "Le code a changé mais aucune vérification pertinente n'a été exécutée.",
        "A implementação foi alterada sem teste ou checagem correspondente.",
    ),
}
_NEUTRAL_PROTOTYPES = (
    "Read one target file to answer a concrete question and recorded the finding.",
    "Planned the next small step before any implementation was requested.",
    "The user asked a conversational question unrelated to repository execution.",
    "Ran a focused failing test once to obtain diagnostic evidence.",
    "El paso de investigación produjo la evidencia solicitada y terminó a tiempo.",
    "Aucune catégorie de problème de comportement ne s'applique à ce message.",
)


@dataclass(frozen=True)
class SemanticBehaviorResult:
    """Selective semantic prediction with enough metadata for replay."""

    label: str | None
    score: float
    runner_up_margin: float
    neutral_margin: float
    space_id: str | None
    classifier_version: str = SEMANTIC_BEHAVIOR_CLASSIFIER_VERSION
    abstention_reason: str = ""


@dataclass(frozen=True)
class _Index:
    vectors: np.ndarray
    owners: tuple[str, ...]
    neutral_vectors: np.ndarray
    space_id: str


@lru_cache(maxsize=1)
def _prototype_index() -> _Index | None:
    embedder = get_static_qwen3_embedder()
    if embedder is None or embedder.model_name != STATIC_QWEN3_MODEL:
        return None
    texts: list[str] = []
    owners: list[str] = []
    for label, examples in _PROTOTYPES.items():
        texts.extend(examples)
        owners.extend([label] * len(examples))
    return _Index(
        vectors=np.asarray(embedder.embed_passages(texts), dtype=np.float32),
        owners=tuple(owners),
        neutral_vectors=np.asarray(
            embedder.embed_passages(list(_NEUTRAL_PROTOTYPES)), dtype=np.float32
        ),
        space_id=embedder.space_id,
    )


def classify_step_behavior(
    text: str,
    *,
    min_score: float = 0.24,
    min_margin: float = 0.035,
    neutral_veto_margin: float = 0.01,
) -> SemanticBehaviorResult:
    """Classify one compact step summary or explicitly abstain."""
    if len(text.strip()) < 24:
        return SemanticBehaviorResult(
            label=None,
            score=0.0,
            runner_up_margin=0.0,
            neutral_margin=0.0,
            space_id=None,
            abstention_reason="summary too short",
        )
    try:
        index = _prototype_index()
        embedder = get_static_qwen3_embedder()
        if index is None or embedder is None:
            raise RuntimeError("bundled embedding artifact unavailable")
        query = np.asarray(embedder.embed_query(text), dtype=np.float32)
        similarities = index.vectors @ query
        neutral_score = float(np.max(index.neutral_vectors @ query))
    except Exception as exc:
        logger.debug("Semantic behavior classification failed", exc_info=True)
        return SemanticBehaviorResult(
            label=None,
            score=0.0,
            runner_up_margin=0.0,
            neutral_margin=0.0,
            space_id=None,
            abstention_reason=str(exc),
        )

    scores = {
        label: max(
            float(similarities[index_position])
            for index_position, owner in enumerate(index.owners)
            if owner == label
        )
        for label in _PROTOTYPES
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    label, score = ranked[0]
    runner_up_margin = score - ranked[1][1]
    neutral_margin = score - neutral_score
    reason = ""
    if score < min_score:
        reason = f"top score {score:.3f} below {min_score:.3f}"
    elif runner_up_margin < min_margin:
        reason = f"runner-up margin {runner_up_margin:.3f} below {min_margin:.3f}"
    elif neutral_margin < neutral_veto_margin:
        reason = f"neutral margin {neutral_margin:.3f} below {neutral_veto_margin:.3f}"
    if reason:
        label = None
    return SemanticBehaviorResult(
        label=label,
        score=score,
        runner_up_margin=runner_up_margin,
        neutral_margin=neutral_margin,
        space_id=index.space_id,
        abstention_reason=reason,
    )


def clear_semantic_behavior_cache() -> None:
    """Reset the cached prototype vectors for tests."""
    _prototype_index.cache_clear()


__all__ = [
    "SEMANTIC_BEHAVIOR_CLASSIFIER_VERSION",
    "SemanticBehaviorResult",
    "classify_step_behavior",
    "clear_semantic_behavior_cache",
]
