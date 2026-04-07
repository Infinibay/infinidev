"""Lightweight accumulator for static-analysis latency.

Used to answer one question with real numbers: across a full task
run, how much wall-clock time did the static-analysis stack
(tree-sitter syntax check, silent-deletion diff, guidance detector,
plan validator) actually consume between LLM calls?

The synthetic benchmarks in ``static_analysis`` show ~15ms per
typical step and ~62ms per worst-case step. This module records
the *real* number across an actual run by accumulating per-category
elapsed time and exposing a getter the engine prints at the end.

Cleanly opt-out: when ``INFINIDEV_DISABLE_SA_TIMER=1`` is set the
``measure`` context manager is a no-op and the counters stay zero.

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


def _is_disabled() -> bool:
    return os.environ.get("INFINIDEV_DISABLE_SA_TIMER") == "1"


def reset() -> None:
    """Clear all accumulators. Call at the start of each engine run."""
    global _state
    _state = _zero_state()


@contextmanager
def measure(category: str) -> Iterator[None]:
    """Time the wrapped block and add the elapsed seconds to *category*.

    Unknown category names are silently ignored so adding new
    instrumentation in a downstream module doesn't crash older
    versions of the loop.
    """
    if _is_disabled() or category not in _state:
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


__all__ = ["measure", "reset", "snapshot", "render"]
