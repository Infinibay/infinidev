"""Penalize repetitive thinking — same ideas restated without action."""

from __future__ import annotations

import hashlib

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_SHORT,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    Confidence,
    confidence_to_delta,
    max_cosine_sim,
)


class RepetitiveThinkingChecker(StochasticChecker):
    name = "repetitive_thinking"
    description = "Penalize repeated thinking that restates the same ideas without acting"
    default_enabled = True
    delta_range = (-3, 0)
    ttl_steps = TTL_SHORT       # thinking loop is a tactical problem, not structural
    settings_message = "Repetitive thinking — punishes loops that re-think instead of acting (-3..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        reasoning = (ctx.reasoning_content or "").strip()
        if len(reasoning) < 80:
            return None
        # If the agent acted this step, repetition is forgivable.
        if ctx.action_tool_calls >= 2:
            return None
        # Compare against prior step summaries (the best proxy for
        # "things the agent already said").
        prior = [s for s in ctx.prior_reasoning_blocks if s]
        if not prior:
            return None

        from infinidev.config.settings import settings

        threshold = float(
            getattr(settings, "BEHAVIOR_REPETITION_COSINE_THRESHOLD", 0.88)
        )
        sim = max_cosine_sim(reasoning, prior[-5:])
        if sim < threshold:
            return None
        # Scale (threshold..1.0) → (0.5..1.0)
        span = max(1e-9, 1.0 - threshold)
        conf = Confidence(
            0.5 + (sim - threshold) / span * 0.5,
            f"cosine={sim:.2f} vs prior reasoning",
        )
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        # Fingerprint on the reasoning block so the same thinking loop
        # isn't punished twice as it slides through the history window.
        trigger_key = hashlib.md5(
            reasoning.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(delta=delta, reason=conf.evidence, trigger_key=trigger_key)
