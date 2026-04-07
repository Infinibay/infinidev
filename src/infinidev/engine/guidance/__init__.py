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

from infinidev.engine.guidance.library import GuidanceEntry, get_entry
from infinidev.engine.guidance.test_runners import (
    is_test_command,
    test_outcome_fingerprint,
    normalize_test_command,
)
from infinidev.engine.guidance.detectors import detect_stuck_pattern
from infinidev.engine.guidance.hooks import (
    maybe_queue_guidance,
    drain_pending_guidance,
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
