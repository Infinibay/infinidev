"""Public hook functions consumed by the engine and the prompt builder.

Two functions, two integration points:

  * :func:`maybe_queue_guidance` is called by the engine at the end of
    each step. It runs the detectors and, when one fires, queues the
    matching :class:`~library.GuidanceEntry` onto the LoopState for
    rendering on the *next* iteration.

  * :func:`drain_pending_guidance` is called by ``build_iteration_prompt``
    at the start of each iteration. It pops the queued text (if any)
    and clears the slot so the same entry isn't rendered twice.

Hard guarantees:
  * Never delivers the same entry twice in one task (``guidance_given``).
  * Never delivers more than ``LOOP_GUIDANCE_MAX_PER_TASK`` entries.
  * Never fires for non-small models.
  * Never costs an LLM call.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from infinidev.engine.guidance.detectors import detect_stuck_pattern
from infinidev.engine.guidance.library import get_entry
from infinidev.engine.guidance.similarity_detector import check_similarity_after_write

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState


def maybe_queue_guidance(
    state: "LoopState",
    messages: list[dict],
    *,
    is_small: bool,
    max_per_task: int,
) -> str | None:
    """Detect a stuck-pattern and queue the matching guidance entry.

    Returns the key that was queued (for logging) or None. Safe to
    call every step — short-circuits when:
      * the model is not classified as small,
      * a guidance is already queued for the next iteration,
      * the per-task quota is reached,
      * no pattern matches,
      * the matching pattern was already delivered earlier in the task.
    """
    from infinidev.engine.static_analysis_timer import measure
    with measure("guidance"):
        if not is_small:
            return None
        if state.pending_guidance:
            return None
        if len(state.guidance_given) >= max_per_task:
            return None

        # Dynamic detector first: similarity_after_write produces its
        # own rendered text (specific method names, file paths, line
        # ranges) and must run BEFORE the static dispatch. It
        # self-manages its de-dup via ``state.similarity_warned_files``
        # so we only gate it on the "already given this task?" check.
        if "similarity_after_write" not in state.guidance_given:
            try:
                dyn_text = check_similarity_after_write(state)
            except Exception:
                dyn_text = None
            if dyn_text:
                state.pending_guidance = dyn_text
                state.guidance_given.append("similarity_after_write")
                return "similarity_after_write"

        key = detect_stuck_pattern(messages, state)
        if not key or key in state.guidance_given:
            return None

        entry = get_entry(key)
        if not entry:
            return None

        state.pending_guidance = entry.render()
        state.guidance_given.append(key)
        return key


def drain_pending_guidance(state: "LoopState") -> str:
    """Pop and return the queued guidance text, or "" if none.

    Called by ``build_iteration_prompt`` exactly once per iteration.
    Idempotent: a second call returns "" until something queues again.
    """
    txt = state.pending_guidance
    state.pending_guidance = ""
    return txt


__all__ = ["maybe_queue_guidance", "drain_pending_guidance"]
