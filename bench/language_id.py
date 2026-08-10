"""Conservative language identification for short developer text.

Generic language detectors are overconfident on terse commit subjects.  This
module compares lexical likelihoods from wordfreq across the five supported
languages and abstains when neither coverage nor separation is sufficient.
It is deterministic, local, and does not require a model download.
"""

from __future__ import annotations

import re

from wordfreq import zipf_frequency


TARGET_LANGUAGES = ("en", "es", "pt", "fr", "it")
_WORD = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_target_language(text: str) -> tuple[str, float]:
    """Return a target language or abstain as ``other``/``unknown``."""
    words = [word for word in _WORD.findall(text.casefold()) if len(word) > 1]
    if not words:
        return "unknown", 0.0
    scores = sorted(
        (
            sum(zipf_frequency(word, language) for word in words),
            language,
        )
        for language in TARGET_LANGUAGES
    )
    top_score, language = scores[-1]
    runner_up = scores[-2][0]
    average_score = top_score / len(words)
    average_margin = (top_score - runner_up) / len(words)
    if average_score < 3.0 or average_margin < 0.25:
        return "other", max(0.0, min(0.49, average_margin / 0.5))
    confidence = min(0.999, 0.5 + average_margin / 2.0)
    return language, confidence
