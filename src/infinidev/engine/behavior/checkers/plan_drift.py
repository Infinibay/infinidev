"""Stochastic plan-drift detection: task-anchored change-point + EWMA persistence.

Design rationale
----------------
Detecting drift from a single (plan_step_title, step_summary) pair is
fundamentally noisy: short strings embed poorly, imperative titles and
past-tense summaries have vocabulary mismatch, and legitimate on-task
work for a given step is a *dispersed cluster*, not a single point you
can threshold against.

Instead, this checker combines two classic stochastic techniques:

1. **Task-anchored change-point detection (Option A).**
   Anchor the baseline on the *task* description — long, rich, stable —
   not on individual step titles. After each completed step, compute
   ``d_i = 1 - cosine(task, summary_i)`` and track a rolling (μ, σ)
   over the last N distances. The current step's "raw drift signal"
   is its z-score against that running baseline, so we're asking
   *"how unusual is this step relative to the agent's own prior
   on-task work?"* rather than comparing against a hardcoded threshold.

2. **EWMA persistence gate (Option B).**
   Drift is a temporal pattern, not an instantaneous event. Any single
   step can be a legitimate detour (exploration, scaffolding, context
   switch). We smooth the per-step signal into an exponentially
   weighted moving average ``D_i = α·s_i + (1-α)·D_{i-1}`` and only
   fire when ``D > 0.7`` for 2+ consecutive steps. Hysteresis latches
   the warning so one sustained drift episode fires *once*, not once
   per step while drifted; the latch releases when ``D < 0.3``.

Per-agent state lives in a module-level dict keyed by
``(project_id, agent_id)``. The scorer's ``reset()`` hook calls
``_reset_state()`` to clear it at task boundaries.
"""

from __future__ import annotations

import hashlib
import logging
import math
from dataclasses import dataclass, field
from typing import Any

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_MEDIUM,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import Confidence, cosine_sim, embed

logger = logging.getLogger(__name__)


# ── Tunables ──────────────────────────────────────────────────────────────
# These mirror their conceptual role in the docstring above. They are
# intentionally *not* settings-exposed for now — the whole point of this
# rewrite is that thresholds should be self-calibrated by the rolling
# statistics, not knobs users have to tune.

_ROLLING_WINDOW = 5            # N: distances kept for (μ, σ)
_MIN_BASELINE = 3              # cold start: need ≥3 priors before evaluating
_EWMA_ALPHA = 0.8              # D_i = α·s_i + (1-α)·D_{i-1}
_FIRE_THRESHOLD = 0.7          # D_i must exceed this
_FIRE_PERSISTENCE = 2          # …for this many consecutive steps
_RELEASE_THRESHOLD = 0.3       # D_i dropping below this rearms the latch
_Z_LOW = 1.0                   # z ≤ Z_LOW → s = 0
_Z_HIGH = 4.0                  # z ≥ Z_HIGH → s = 1
_SIGMA_FLOOR = 0.05            # prevent divide-by-zero on flat baselines
_MIN_SUMMARY_LEN = 30          # telegraphic summaries are too thin to judge
_WINSORIZE_K = 2.0             # cap appended distances at μ + K·σ (robust update)

_READ_ONLY_TOOLS: frozenset[str] = frozenset({
    "read_file", "partial_read", "list_directory", "glob", "code_search",
    "get_symbol_code", "list_symbols", "search_symbols", "find_references",
    "project_structure", "help", "git_status", "git_diff", "git_log",
    "read_findings", "search_findings",
})


# ── Per-agent rolling state ───────────────────────────────────────────────

@dataclass
class _DriftState:
    task_hash: str = ""                   # md5 of the task text
    task_vec: Any = None                  # cached task embedding (np.ndarray)
    distances: list[float] = field(default_factory=list)
    ewma: float = 0.0
    consecutive_high: int = 0
    latched: bool = False                 # already fired this episode?
    episode_counter: int = 0              # monotonic, bumps per fire


_state_by_agent: dict[tuple[int, str], _DriftState] = {}


def _reset_state() -> None:
    """Clear all per-agent rolling state. Called from ``BehaviorScorer.reset``."""
    _state_by_agent.clear()


def _get_state(project_id: int, agent_id: str) -> _DriftState:
    key = (project_id, agent_id)
    st = _state_by_agent.get(key)
    if st is None:
        st = _DriftState()
        _state_by_agent[key] = st
    return st


def _ensure_task_anchor(st: _DriftState, task: str) -> Any:
    """Embed the task once and cache it; re-embed if the task changed.

    Returns the cached embedding, or ``None`` if embedding failed.
    """
    import numpy as np

    task_hash = hashlib.md5(task.encode("utf-8", errors="ignore")).hexdigest()
    if st.task_hash != task_hash:
        # Task changed → start a fresh episode. This is the right
        # behaviour because "drift" is only defined relative to a
        # specific task.
        st.task_hash = task_hash
        st.distances.clear()
        st.ewma = 0.0
        st.consecutive_high = 0
        st.latched = False
        st.episode_counter += 1
        st.task_vec = embed(task)
    elif st.task_vec is None:
        st.task_vec = embed(task)
    return st.task_vec


def _cosine_with_vec(a_vec: Any, text: str) -> float:
    """Cosine between a pre-cached vector and a freshly embedded text."""
    if a_vec is None:
        return 0.0
    import numpy as np

    b_vec = embed(text)
    if b_vec is None:
        return 0.0
    try:
        from infinidev.tools.base.dedup import _cosine_similarity

        return float(_cosine_similarity(a_vec, b_vec))
    except Exception:
        return 0.0


def _z_to_signal(z: float) -> float:
    """Map a z-score into a drift signal in ``[0, 1]``.

    ``z <= Z_LOW`` → 0 (no signal), ``z >= Z_HIGH`` → 1 (strong signal),
    linear in between. Intentionally simple — the persistence gate
    downstream does the heavy lifting.
    """
    if z <= _Z_LOW:
        return 0.0
    if z >= _Z_HIGH:
        return 1.0
    return (z - _Z_LOW) / (_Z_HIGH - _Z_LOW)


def _rolling_stats(values: list[float]) -> tuple[float, float]:
    """Return ``(mean, std)`` of *values*, with std floored at ``_SIGMA_FLOOR``."""
    n = len(values)
    if n == 0:
        return 0.0, _SIGMA_FLOOR
    mean = sum(values) / n
    if n < 2:
        return mean, _SIGMA_FLOOR
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    std = math.sqrt(max(0.0, var))
    return mean, max(std, _SIGMA_FLOOR)


# ── The checker ───────────────────────────────────────────────────────────

class PlanDriftChecker(StochasticChecker):
    name = "plan_drift"
    description = "Punish persistent drift from the task via change-point detection"
    default_enabled = True
    delta_range = (-2, 0)
    ttl_steps = TTL_MEDIUM      # drift episodes fade as the agent recovers
    settings_message = "Plan drift — task-anchored change-point + EWMA persistence (-2..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        # Need both identity and a task anchor.
        if not ctx.task:
            return None
        summary = (ctx.step_summary or "").strip()
        if len(summary) < _MIN_SUMMARY_LEN:
            return None
        # Exploration-only steps are always legitimate.
        non_read = [c for c in ctx.tool_calls if c.name not in _READ_ONLY_TOOLS]
        if not non_read:
            return None

        st = _get_state(ctx.project_id, ctx.agent_id)
        task_vec = _ensure_task_anchor(st, ctx.task)
        if task_vec is None:
            # Embedder unavailable — silently skip rather than falsely fire.
            return None

        # d_i = 1 - cos(task, summary). Higher = further from task intent.
        cos = _cosine_with_vec(task_vec, summary)
        d_i = 1.0 - cos

        # We compute the z-score BEFORE appending d_i so the current
        # observation is compared against the *prior* baseline, not
        # against a window that already includes itself.
        baseline = st.distances[-_ROLLING_WINDOW:]
        mean, std = _rolling_stats(baseline)
        z = (d_i - mean) / std

        # Cold start: update state but don't emit a verdict until we
        # have enough history to form a meaningful baseline.
        cold_start = len(baseline) < _MIN_BASELINE

        # Raw per-step drift signal ∈ [0, 1].
        s_i = _z_to_signal(z) if not cold_start else 0.0

        # EWMA smoothing (B): a one-off high z doesn't count; sustained
        # high z accumulates.
        st.ewma = _EWMA_ALPHA * s_i + (1 - _EWMA_ALPHA) * st.ewma

        # Persistence counter.
        if st.ewma > _FIRE_THRESHOLD:
            st.consecutive_high += 1
        else:
            st.consecutive_high = 0

        # Hysteresis: once the EWMA drops clearly below the release
        # threshold, rearm the latch so a future drift episode can
        # fire again. On the release transition we also FLUSH the
        # rolling baseline — otherwise the distances collected during
        # the drift episode linger and desensitise detection of the
        # next episode ("drift fatigue"). Flushing forces a short cold
        # start on the post-recovery steps, so the new baseline
        # reflects the actual on-task work the agent is doing now.
        if st.ewma < _RELEASE_THRESHOLD:
            if st.latched:
                st.distances.clear()
            st.latched = False

        # Winsorized append: cap the value we add to the baseline at
        # ``μ + K·σ`` so a sustained outlier can't hijack the rolling
        # statistics. This keeps the baseline slowly evolving (so
        # legitimately more-varied on-task work still gets absorbed)
        # while preventing drift episodes from becoming "the new
        # normal" within a window-length of steps. In the cold-start
        # phase the cap would be meaningless, so skip it.
        if cold_start:
            st.distances.append(d_i)
        else:
            cap = mean + _WINSORIZE_K * std
            st.distances.append(min(d_i, cap))
        if len(st.distances) > _ROLLING_WINDOW:
            st.distances = st.distances[-_ROLLING_WINDOW:]

        if cold_start:
            return None
        if st.latched:
            return None
        if st.consecutive_high < _FIRE_PERSISTENCE:
            return None

        # All gates passed — fire once, then latch until the EWMA
        # recovers below _RELEASE_THRESHOLD.
        st.latched = True
        st.episode_counter += 1
        conf = Confidence(
            min(1.0, st.ewma),
            f"drift z={z:.2f}, D={st.ewma:.2f}, {st.consecutive_high} steps",
        )

        # Map EWMA directly to delta: D=0.7 → ~-1, D≥0.9 → -2.
        if st.ewma >= 0.9:
            delta = -2
        else:
            delta = -1

        # trigger_key is unique per episode so the scorer dedup doesn't
        # collapse genuinely distinct drift events, but the same episode
        # (if re-presented) will still be deduped.
        trigger_key = hashlib.md5(
            f"{st.task_hash}|episode={st.episode_counter}".encode("utf-8")
        ).hexdigest()[:16]
        return Verdict(
            delta=delta,
            reason=(
                f"plan_drift: EWMA={st.ewma:.2f} over "
                f"{st.consecutive_high} steps (z={z:.2f} vs "
                f"μ={mean:.2f} σ={std:.2f})"
            ),
            trigger_key=trigger_key,
        )
