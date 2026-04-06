"""Text primitives: regex scans, keyword presence, fuzzy string ratio."""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Iterable

from infinidev.engine.behavior.primitives.scoring import Confidence


def regex_scan(
    text: str,
    patterns: dict[str, re.Pattern[str]],
    per_hit_weight: float = 0.5,
) -> Confidence:
    """Scan *text* for any of the named regex *patterns*.

    Confidence rises with the number of matches, capped at 1.0.
    Evidence names the first pattern that fired.
    """
    if not text or not patterns:
        return Confidence.none()
    total = 0
    first_name: str | None = None
    first_snippet: str = ""
    for name, pat in patterns.items():
        m = pat.search(text)
        if m is None:
            continue
        # Count all occurrences for this pattern too
        total += len(pat.findall(text))
        if first_name is None:
            first_name = name
            first_snippet = m.group(0)[:60]
    if total == 0:
        return Confidence.none()
    value = min(1.0, total * per_hit_weight)
    return Confidence(value, f"{first_name}: {first_snippet}")


def keyword_presence(
    text: str,
    keywords: Iterable[str],
    case_sensitive: bool = False,
) -> Confidence:
    """Return how many of *keywords* appear in *text* (0..1)."""
    if not text:
        return Confidence.none()
    haystack = text if case_sensitive else text.lower()
    hits: list[str] = []
    for kw in keywords:
        needle = kw if case_sensitive else kw.lower()
        if needle in haystack:
            hits.append(kw)
    if not hits:
        return Confidence.none()
    # Presence of any keyword is a strong signal; saturate fast.
    value = min(1.0, 0.6 + 0.1 * (len(hits) - 1))
    return Confidence(value, f"keywords: {', '.join(hits[:3])}")


def fuzzy_ratio(a: str, b: str) -> float:
    """Return a 0..1 similarity ratio between two short strings.

    Uses stdlib ``difflib.SequenceMatcher`` — no external dependency.
    Good enough for plan-step/summary drift detection.
    """
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()
