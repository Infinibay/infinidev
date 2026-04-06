"""Confidence scoring and delta mapping for stochastic checkers.

Every primitive returns a :class:`Confidence` — a numeric value in
``[0, 1]`` plus a short evidence string. Checkers compose one or more
confidences (via :func:`combine`) and map the result to an integer
delta in the checker's declared range via :func:`confidence_to_delta`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Confidence:
    """How strongly a stochastic check believes the signal fired."""

    value: float = 0.0          # 0.0 = no signal, 1.0 = certain
    evidence: str = ""          # short human-readable snippet

    @classmethod
    def none(cls) -> "Confidence":
        return cls(0.0, "")

    def __bool__(self) -> bool:
        return self.value > 0.0


def combine(*cs: Confidence, mode: str = "max") -> Confidence:
    """Combine multiple confidences into one.

    Modes:
      - "max"   : take the strongest signal (default; OR-like)
      - "mean"  : average the values (AND-like when all fire weakly)
      - "any"   : first non-zero wins (ordered fallback)
    """
    cs = tuple(c for c in cs if c is not None)
    if not cs:
        return Confidence.none()
    if mode == "mean":
        val = sum(c.value for c in cs) / len(cs)
        ev = " | ".join(c.evidence for c in cs if c.evidence)[:200]
        return Confidence(val, ev)
    if mode == "any":
        for c in cs:
            if c.value > 0:
                return c
        return Confidence.none()
    # max (default)
    best = max(cs, key=lambda c: c.value)
    return Confidence(best.value, best.evidence)


def confidence_to_delta(
    delta_range: tuple[int, int],
    conf: Confidence,
    threshold: float = 0.5,
) -> int:
    """Map a confidence value onto the checker's integer delta range.

    Rules:
      - ``conf.value < threshold`` → 0 (no signal, do not fire).
      - Otherwise, scale linearly within the *signed half* of the range.
        If the range is punish-only (``hi <= 0``), return a negative delta.
        If reward-only (``lo >= 0``), return a positive delta.
        Bidirectional ranges are not auto-scaled here — checkers with
        ``(-2, +2)`` etc. must build their own Verdict explicitly.
    """
    if conf.value < threshold:
        return 0
    lo, hi = delta_range
    if hi <= 0:
        # punish-only: stronger confidence → more negative
        span = lo  # e.g. -3
        # Map [threshold..1] → [0..span]
        t = (conf.value - threshold) / max(1e-9, 1.0 - threshold)
        return max(lo, min(0, round(t * span)))
    if lo >= 0:
        # reward-only: stronger confidence → more positive
        span = hi
        t = (conf.value - threshold) / max(1e-9, 1.0 - threshold)
        return min(hi, max(0, round(t * span)))
    # bidirectional: caller should handle it
    return 0
