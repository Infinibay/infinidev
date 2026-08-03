"""Reactive guidance system for small models that get stuck.

Most models don't need this. Frontier and 30B+ local models plan and
edit fine on their own — injecting guidance for them is just token
waste. The system is designed to fire **only** when a model has
demonstrated a clear stuck-pattern that pre-baked advice can resolve,
and only when ``is_small`` is true.

The package is organised by responsibility:

  * :mod:`library` — :class:`GuidanceEntry` dataclass and the dict
    of pre-baked entries. Pure data, no imports from the rest of the
    engine. Add a new entry by appending to ``_LIBRARY``.

  * :mod:`test_runners` — multi-language test runner detection
    (:func:`is_test_command`) and outcome fingerprinting
    (:func:`test_outcome_fingerprint`). Used by the test-loop
    detectors and reusable from anywhere else.

  * :mod:`detectors` — the stuck-pattern detector functions and the
    priority-ordered registry. :func:`detect_stuck_pattern` walks the
    registry and returns the first key that fires.

  * :mod:`hooks` — the two public hook functions consumed by the
    engine: :func:`maybe_queue_guidance` (called after each step) and
    :func:`drain_pending_guidance` (called by the prompt builder).

Hard guarantees:
  * Never delivers the same entry twice in one task.
  * Never delivers more than ``LOOP_GUIDANCE_MAX_PER_TASK`` entries.
  * Never fires for non-small models.
  * Never costs an LLM call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from infinidev.engine.guidance.detectors import detect_stuck_pattern
    from infinidev.engine.guidance.hooks import (
        drain_pending_guidance,
        maybe_queue_guidance,
    )
    from infinidev.engine.guidance.library import GuidanceEntry, get_entry
    from infinidev.engine.guidance.test_runners import (
        is_test_command,
        normalize_test_command,
        test_outcome_fingerprint,
    )

__all__ = [
    "GuidanceEntry",
    "get_entry",
    "is_test_command",
    "test_outcome_fingerprint",
    "normalize_test_command",
    "detect_stuck_pattern",
    "maybe_queue_guidance",
    "drain_pending_guidance",
]

_EXPORTS = {
    "GuidanceEntry": ("infinidev.engine.guidance.library", "GuidanceEntry"),
    "get_entry": ("infinidev.engine.guidance.library", "get_entry"),
    "is_test_command": ("infinidev.engine.guidance.test_runners", "is_test_command"),
    "normalize_test_command": (
        "infinidev.engine.guidance.test_runners",
        "normalize_test_command",
    ),
    "test_outcome_fingerprint": (
        "infinidev.engine.guidance.test_runners",
        "test_outcome_fingerprint",
    ),
    "detect_stuck_pattern": (
        "infinidev.engine.guidance.detectors",
        "detect_stuck_pattern",
    ),
    "maybe_queue_guidance": (
        "infinidev.engine.guidance.hooks",
        "maybe_queue_guidance",
    ),
    "drain_pending_guidance": (
        "infinidev.engine.guidance.hooks",
        "drain_pending_guidance",
    ),
}


def __getattr__(name: str) -> Any:
    """Load public guidance helpers without importing the engine eagerly."""
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    module_name, attribute = target
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
