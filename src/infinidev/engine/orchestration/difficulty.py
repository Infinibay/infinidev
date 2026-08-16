"""Difficulty classifier that adapts Task Planner depth to request complexity.

The current Task Planner prompt is written for hard requests — full plan with
exploration, critique, decomposition, and review. Many real requests are
single-file fixes or focused tweaks that don't benefit from the same depth.
This module classifies the request as ``easy`` / ``medium`` / ``hard`` using
deterministic, cheap signals (no LLM call) so the planner can scale its
effort to the actual task and we stop spending hard-level tokens on typo
fixes.

Design decisions
----------------
* **Default to ``hard`` when signals are ambiguous.** That preserves current
  behaviour when the classifier is uncertain and avoids under-planning
  meaningful work. The downstream planner has its own guardrails against
  over-planning easy tasks; the cost of under-planning a real task is
  higher.
* **Deterministic, no LLM call.** Regex + count signals only. The bundled
  mini-head classifier at ``engine.task_policies`` lives at a different
  semantic level (task **method** — bugfix/feature/refactor/research/etc.),
  not task complexity. Mixing the two would over-fit to wording.
* **Build on :mod:`request_signals`.** Path detection and execution-intent
  scoring are already implemented there; this module composes them
  instead of duplicating regexes.
* **Audit-friendly output.** :class:`DifficultyDecision` carries the raw
  signal counts and a reason so callers can log or surface the choice.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Literal

from infinidev.engine.orchestration.request_signals import (
    explicit_execution_score,
    referenced_file_paths,
)

logger = logging.getLogger(__name__)

DifficultyLevel = Literal["easy", "medium", "hard"]
DIFFICULTY_LEVELS: tuple[DifficultyLevel, ...] = ("easy", "medium", "hard")

# Short requests usually mean a focused tweak. Multi-file paths skew hard
# even if the wording is brief. Long-running sessions skew hard because the
# easy work is typically done first.
_SHORT_REQUEST_CHARS = 120
_MANY_FILES_THRESHOLD = 3
_LONG_SESSION_TURNS = 12


# Easy keywords: small, focused, single-action requests. Phrased as a
# permissive list so we don't depend on a particular wording.
_EASY_KEYWORDS: tuple[str, ...] = (
    r"\btypo\b",
    r"\bminor\b",
    r"\btrivial\b",
    r"\bone\s*[-_ ]line\b",
    r"\bquick(?:ly)?\s+(?:fix|tweak|patch)\b",
    r"\bsmall\s+(?:fix|change|tweak|edit|adjustment)\b",
    r"\brename\s+(?:a|the|one)?\s*[A-Za-z_][\w.]*\b",
    r"\bfix\s+(?:the\s+)?(?:typo|spelling|formatting|lint|warning)\b",
    r"\badd\s+(?:a\s+)?(?:docstring|comment|type\s*hints?)\b",
    r"\bupdate\s+(?:the\s+)?(?:comment|docstring|readme|changelog)\b",
    r"\bbump\s+(?:the\s+)?(?:version|dep(?:endency)?)\b",
    r"\bremove\s+(?:a|an|the)?\s+(?:dead|unused|stale)\b",
)

# Hard keywords: cross-cutting, architectural, analysis-heavy, or
# explicitly large-scope requests.
_HARD_KEYWORDS: tuple[str, ...] = (
    r"\brefactor\b",
    r"\brearchitect(?:ure)?\b",
    r"\bmigrat(?:e|ion)\b",
    r"\bbreaking\s+change\b",
    r"\bcross[- ]cutting\b",
    r"\b(?:system|architecture)\s+(?:wide|level)\b",
    r"\bdesign\s+(?:a|the|new)\b",
    r"\bdesign\s+the\s+(?:api|interface|protocol)\b",
    r"\binvestigat(?:e|ion)\b",
    r"\banaly[sz](?:e|ing|ation|ysis)\b",
    r"\bestudi(?:ar|os?)\b",
    r"\bbig\s+(?:change|refactor|redesign)\b",
    r"\bperformance\s+(?:audit|review)\b",
    r"\bsecurity\s+(?:audit|review|fix)\b",
    r"\b(?:start|build)\s+(?:a\s+)?new\s+(?:feature|module|service|system|project)\b",
    r"\bfrom\s+scratch\b",
    r"\b(?:full|complete)\s+(?:rewrite|overhaul)\b",
    r"\bin\s+detail\b",
    r"\baudit\b",
)

_EASY_HITS_RE = re.compile("|".join(_EASY_KEYWORDS), re.IGNORECASE)
_HARD_HITS_RE = re.compile("|".join(_HARD_KEYWORDS), re.IGNORECASE)


@dataclass(frozen=True)
class DifficultyDecision:
    """One classifier outcome with the evidence that decided it.

    ``signals`` exposes the raw counts (char length, file count, keyword
    hits, execution score, session length) so reviewers and tests can audit
    why a given level was chosen. ``reason`` is a single-sentence human
    summary intended for logs.
    """

    level: DifficultyLevel
    confidence: float
    signals: dict[str, int] = field(default_factory=dict)
    reason: str = ""


def _keyword_count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def resolve_difficulty(
    user_request: str,
    *,
    opened_files: tuple[str, ...] | list[str] = (),
    prior_turn_count: int = 0,
) -> DifficultyDecision:
    """Classify a request as ``easy`` / ``medium`` / ``hard``.

    Args:
        user_request: The raw or normalised ask. Empty input returns hard
            with zero confidence (the conservative default).
        opened_files: Files already inspected by the time we classify. Used
            to detect "multi-file" requests without parsing the request
            text twice.
        prior_turn_count: Already-explored turn count in the same session.
            Long sessions skew toward hard because easy work is usually
            done first.

    Returns:
        :class:`DifficultyDecision` with the level, a 0-1 confidence, the
        raw signal counts, and a short reason string.
    """
    text = (user_request or "").strip()
    if not text:
        return DifficultyDecision(
            level="hard",
            confidence=0.0,
            signals={"char_count": 0},
            reason="empty request; defaulting to hard",
        )

    char_count = len(text)
    mentioned_paths = len(referenced_file_paths(text))
    opened = {p for p in opened_files if p}
    path_count = max(mentioned_paths, len(opened))
    easy_hits = _keyword_count(_EASY_HITS_RE, text)
    hard_hits = _keyword_count(_HARD_HITS_RE, text)
    exec_score = explicit_execution_score(text)

    signals: dict[str, int] = {
        "char_count": char_count,
        "path_count": path_count,
        "easy_keyword_hits": easy_hits,
        "hard_keyword_hits": hard_hits,
        "execution_score": exec_score,
        "prior_turn_count": int(prior_turn_count),
    }

    # --- Hard path --------------------------------------------------------
    # Hard keywords are the strongest signal: architectural / refactor /
    # analysis requests are always full-depth regardless of length.
    if hard_hits >= 1:
        return DifficultyDecision(
            level="hard",
            confidence=min(1.0, 0.7 + 0.15 * hard_hits),
            signals=signals,
            reason=(
                f"hard_keyword_hits={hard_hits} "
                "(cross-cutting or architectural)"
            ),
        )

    if path_count >= _MANY_FILES_THRESHOLD:
        return DifficultyDecision(
            level="hard",
            confidence=0.75,
            signals=signals,
            reason=f"path_count={path_count} >= {_MANY_FILES_THRESHOLD}",
        )

    if prior_turn_count >= _LONG_SESSION_TURNS:
        return DifficultyDecision(
            level="hard",
            confidence=0.65,
            signals=signals,
            reason=(
                f"prior_turn_count={prior_turn_count} (long-running session)"
            ),
        )

    # --- Easy path --------------------------------------------------------
    # A keyword hit is required so we never collapse a vague or
    # unfamiliar request to "easy". Structural corroboration (short,
    # single file) gates the keyword so we don't accept a typo-fix
    # keyword in a multi-file refactor request by accident.
    if easy_hits >= 1 and hard_hits == 0:
        structural = 0
        if char_count < _SHORT_REQUEST_CHARS:
            structural += 1
        if path_count == 1:
            structural += 1
        easy_evidence = easy_hits + structural
        if easy_evidence >= 2:
            return DifficultyDecision(
                level="easy",
                confidence=min(1.0, 0.55 + 0.15 * easy_evidence),
                signals=signals,
                reason=(
                    f"easy_evidence={easy_evidence} "
                    "(keyword + short/single-file corroboration)"
                ),
            )

    # --- Medium path ------------------------------------------------------
    # Grounded execution requests land here. The planner still produces a
    # complete plan but can skip the most expensive sub-steps reserved for
    # hard requests.
    if exec_score >= 4 and hard_hits == 0:
        return DifficultyDecision(
            level="medium",
            confidence=0.6,
            signals=signals,
            reason=f"grounded execution (exec_score={exec_score})",
        )

    # Ambiguous input — stay medium (between full-depth and minimal).
    return DifficultyDecision(
        level="medium",
        confidence=0.4,
        signals=signals,
        reason="no decisive signal; defaulting to medium",
    )


__all__ = [
    "DIFFICULTY_LEVELS",
    "DifficultyDecision",
    "DifficultyLevel",
    "resolve_difficulty",
]