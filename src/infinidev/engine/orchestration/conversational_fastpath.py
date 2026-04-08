"""Heuristic conversational fast-path for the orchestration pipeline.

The full ``_run_analysis_phase`` makes an LLM call (the analyst) on
every user turn. For trivial input — greetings, thanks, "are you
there?" — that call adds 5-30 seconds of latency before the user sees
anything. The fast-path here intercepts those inputs with pure-Python
matching and synthesises an :class:`AnalysisResult` with
``flow="done"`` and a hardcoded reply, skipping the analyst entirely.

Design rules:

1. **Pure heuristics, zero LLM**. The whole point is sub-100 ms.
   If a case is ambiguous, FALL THROUGH — never guess. The full
   pipeline catches the cases this misses.
2. **Bilingual**. The agent is used in both English and Spanish.
   Patterns and replies live in pairs.
3. **Anchored matching**. We match the WHOLE normalised input
   against patterns, not substrings — otherwise "fix the hello
   world bug" would trigger the "hello" rule.
4. **No conversational memory needed**. v1 only handles inputs
   that have a context-free correct reply. "Thanks for that" needs
   memory, so it falls through to the analyst.
"""

from __future__ import annotations

import re
from typing import Optional

from infinidev.engine.analysis.analysis_result import AnalysisResult


# ─────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────
#
# Each entry is (pattern, reply). The pattern is a regex anchored
# (^...$) against the normalised input. The reply is the literal
# string we send back through ``analysis.reason``.
#
# Patterns are evaluated in order; first match wins. Order them so
# the more specific patterns come before the more general ones.

_PATTERNS: list[tuple[str, str]] = [
    # ── Greetings ────────────────────────────────────────────────────
    (
        r"^(hi|hello|hey|hola|hola+|buenas|buenas? d[ií]as|buenas? "
        r"tardes|buenas? noches|hi there|hello there|hey there|"
        r"good (morning|afternoon|evening))[!.\s]*$",
        "¡Hola! ¿En qué te puedo ayudar?",
    ),

    # ── Status checks ────────────────────────────────────────────────
    (
        r"^(are you (there|alive|awake|ready|working)|"
        r"est[áa]s (ah[íi]|listo|despierto|funcionando)|"
        r"hay alguien( ah[íi])?|"
        r"ping|ping\?)[!.?\s]*$",
        "Sí, acá estoy. Decime qué necesitás.",
    ),

    # ── Thanks (without specific reference — generic) ────────────────
    (
        r"^(thanks|thank you|thx|ty|gracias|muchas gracias|"
        r"thanks!|cheers)[!.\s]*$",
        "¡De nada! Si necesitás algo más, decime.",
    ),

    # ── Goodbyes ─────────────────────────────────────────────────────
    (
        r"^(bye|goodbye|see you( later)?|cya|cu|"
        r"chau|chao|adi[óo]s|hasta luego|nos vemos|"
        r"see ya|later|au revoir)[!.\s]*$",
        "¡Chau! Cuando me necesites, acá estoy.",
    ),

    # ── Small affirmations / acknowledgements ────────────────────────
    (
        r"^(ok|okay|okey|vale|dale|listo|sounds good|got it|"
        r"perfect|perfecto|great|genial|excelente|cool)[!.\s]*$",
        "Listo. ¿Pasamos a algo más?",
    ),

    # ── Self-identity questions ──────────────────────────────────────
    (
        r"^(who are you|what are you|qu[ée]n eres|qu[ée]n sos|"
        r"qu[ée] eres|qu[ée] sos|what is this|qu[ée] es esto)[?.!\s]*$",
        "Soy Infinidev, un asistente de programación que corre en tu "
        "máquina con modelos locales. Decime una tarea (un bug, una "
        "feature, una pregunta sobre el código) y la ataco.",
    ),

    # ── "What can you do" ────────────────────────────────────────────
    (
        r"^(what can you do|qu[ée] pod[ée]s hacer|qu[ée] sabes hacer|"
        r"help me|help|ayuda|how do you work|c[óo]mo funcionas)[?.!\s]*$",
        "Puedo leer tu código, buscar símbolos, hacer ediciones "
        "quirúrgicas, correr tests, y razonar sobre arquitectura. "
        "Decime una tarea concreta — por ejemplo: \"arregla el bug "
        "de auth en src/auth.py\" o \"añadí logging a la función X\" — "
        "y voy paso a paso. Para ver más comandos del CLI: /help.",
    ),
]


# ─────────────────────────────────────────────────────────────────────────
# Compiled regex cache (built once at import time)
# ─────────────────────────────────────────────────────────────────────────


_COMPILED: list[tuple[re.Pattern[str], str]] = [
    (re.compile(pattern, re.IGNORECASE), reply)
    for pattern, reply in _PATTERNS
]


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Strip whitespace and collapse internal runs of whitespace."""
    return re.sub(r"\s+", " ", text.strip())


# Maximum length of input we'll even consider for fast-path matching.
# Anything longer is almost certainly a real task, even if it starts
# with "hi". This is the main false-positive guard.
_MAX_FASTPATH_LEN = 60


def try_conversational_fastpath(user_input: str) -> Optional[AnalysisResult]:
    """Return a synthesised AnalysisResult or None.

    Returns ``None`` for any input that doesn't match a pattern, or
    is too long to be plausibly conversational. The caller should
    fall through to the normal analyst pipeline in that case.
    """
    if not user_input:
        return None

    text = _normalise(user_input)
    if len(text) > _MAX_FASTPATH_LEN:
        return None

    for compiled, reply in _COMPILED:
        if compiled.match(text):
            return AnalysisResult(
                action="passthrough",
                original_input=user_input,
                reason=reply,
                flow="done",
            )

    return None


__all__ = ["try_conversational_fastpath"]
