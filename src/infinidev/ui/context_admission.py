"""Admission policy for models too small to retain useful task context."""

from __future__ import annotations

from dataclasses import dataclass

from infinidev.engine.loop.context_manager import CONTEXT_COMPACTION_MIN_REMAINING
from infinidev.engine.loop.model_context import get_model_context_window

# A model with less than two fixed reserves has little room for both the
# initial task prompt and retained evidence before pressure compaction starts.
SMALL_CONTEXT_ADMISSION_MAX = CONTEXT_COMPACTION_MIN_REMAINING * 2
LARGE_CONTEXT_CANDIDATE_MIN = 200_000


@dataclass(frozen=True)
class ContextAdmission:
    """The explicit action needed before starting a small-window task."""

    active_window: int
    replacement_model: str | None


def find_context_admission(
    *,
    model: str,
    provider_id: str,
    llm_params: dict[str, object],
    candidates: list[str],
) -> ContextAdmission | None:
    """Return a same-provider large-model offer for an undersized window.

    Unknown windows deliberately do not block a task: no safe capacity claim
    can be made. Candidate windows are resolved through the engine's shared
    capacity source, so the UI cannot offer a model the loop would size
    differently.
    """
    active_window = get_model_context_window(llm_params, provider_id)
    if active_window is None or active_window >= SMALL_CONTEXT_ADMISSION_MAX:
        return None

    replacement_model = None
    for candidate in candidates:
        if candidate == model:
            continue
        candidate_params = {**llm_params, "model": candidate}
        candidate_window = get_model_context_window(candidate_params, provider_id)
        if candidate_window is not None and candidate_window > LARGE_CONTEXT_CANDIDATE_MIN:
            replacement_model = candidate
            break
    return ContextAdmission(active_window, replacement_model)
