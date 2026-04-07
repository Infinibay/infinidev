"""Lightweight accumulator for static-analysis latency.

Used to answer one question with real numbers: across a full task
run, how much wall-clock time did the static-analysis stack
(tree-sitter syntax check, silent-deletion diff, guidance detector,
plan validator) actually consume between LLM calls?

OFF BY DEFAULT. The accumulator is opt-in: set the env var
``INFINIDEV_ENABLE_SA_TIMER=1`` to turn on measurement. With the
flag unset, ``measure`` is a no-op of nanoseconds (one env-var
read + a yield) and the counters stay zero. This means a normal
end-user run pays effectively zero overhead and never produces a
benchmark report it doesn't want.

The categories are:

  * ``syntax_check``    — tree-sitter parse + ERROR walk
  * ``silent_deletion`` — extract_top_level_symbols × 2 + set diff
  * ``guidance``        — detect_stuck_pattern + maybe_queue_guidance
  * ``plan_validate``   — _looks_concrete regex check on add_step

Each entry stores the cumulative elapsed time in seconds AND the
number of times the category was measured, so the report can show
both totals and averages.

The implementation is intentionally tiny — a module-level dict, a
context manager, a reset function, a render function — and never
imports anything that isn't pure stdlib.
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Iterator


# Categories the engine measures. Order is the display order in
# ``render``. Adding a new category is one new key here + one new
# ``with measure("name"):`` block at the call site.
_CATEGORIES: tuple[str, ...] = (
    "syntax_check",
    "silent_deletion",
    "guidance",
    "plan_validate",
)


def _zero_state() -> dict[str, dict[str, float]]:
    return {cat: {"total_s": 0.0, "calls": 0} for cat in _CATEGORIES}


_state: dict[str, dict[str, float]] = _zero_state()


def is_enabled() -> bool:
    """True when the timer should record measurements.

    Off by default. Enable with ``INFINIDEV_ENABLE_SA_TIMER=1`` (or
    any of: ``true``, ``yes``, ``on`` — case insensitive). When off,
    ``measure`` is a no-op of nanoseconds and the counters stay zero.
    """
    raw = os.environ.get("INFINIDEV_ENABLE_SA_TIMER", "")
    return raw.strip().lower() in ("1", "true", "yes", "on")


def reset() -> None:
    """Clear all accumulators. Call at the start of each engine run."""
    global _state
    _state = _zero_state()


@contextmanager
def measure(category: str) -> Iterator[None]:
    """Time the wrapped block and add the elapsed seconds to *category*.

    Cheap no-op when ``INFINIDEV_ENABLE_SA_TIMER`` is unset (the
    default): one env-var read + a yield. Unknown category names are
    silently ignored so adding new instrumentation in a downstream
    module doesn't crash older versions of the loop.
    """
    if not is_enabled() or category not in _state:
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        bucket = _state[category]
        bucket["total_s"] += elapsed
        bucket["calls"] = int(bucket["calls"]) + 1


def snapshot() -> dict[str, dict[str, float]]:
    """Return a copy of the current accumulators (for tests / programmatic use)."""
    return {cat: dict(values) for cat, values in _state.items()}


def render(*, indent: str = "  ") -> str:
    """Format the accumulated totals as a multi-line summary."""
    lines: list[str] = ["Static analysis accumulated latency:"]
    grand_total_ms = 0.0
    grand_calls = 0
    for cat in _CATEGORIES:
        bucket = _state[cat]
        total_ms = float(bucket["total_s"]) * 1000.0
        calls = int(bucket["calls"])
        grand_total_ms += total_ms
        grand_calls += calls
        avg_ms = total_ms / calls if calls else 0.0
        lines.append(
            f"{indent}{cat:18} {total_ms:8.2f} ms total  "
            f"{calls:5d} calls  {avg_ms:6.2f} ms avg"
        )
    lines.append(
        f"{indent}{'TOTAL':18} {grand_total_ms:8.2f} ms total  "
        f"{grand_calls:5d} calls"
    )
    return "\n".join(lines)


__all__ = ["is_enabled", "measure", "reset", "snapshot", "render"]
