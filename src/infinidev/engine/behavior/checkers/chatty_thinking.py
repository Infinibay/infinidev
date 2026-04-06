"""Punish very long thinking on trivially simple tasks."""

from __future__ import annotations

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_EPHEMERAL,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import Confidence, confidence_to_delta


class ChattyThinkingChecker(StochasticChecker):
    name = "chatty_thinking"
    description = "Punish overly long thinking for trivial tasks"
    default_enabled = False
    delta_range = (-2, 0)
    ttl_steps = TTL_EPHEMERAL   # one-step nuisance; don't haunt the agent
    settings_message = "Chatty thinking — punishes huge reasoning blobs on simple tasks (-2..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        from infinidev.config.settings import settings

        threshold_chars = int(
            getattr(settings, "BEHAVIOR_CHATTY_CHAR_THRESHOLD", 2000)
        )
        n = len(ctx.reasoning_content or "")
        if n < threshold_chars:
            return None
        # If the agent actually acted, long thinking was probably justified.
        if ctx.action_tool_calls >= 2:
            return None
        # Scale length (threshold..3×threshold) → confidence (0.5..1.0)
        excess = (n - threshold_chars) / (2 * threshold_chars)
        conf = Confidence(
            min(1.0, 0.5 + excess * 0.5),
            f"{n} reasoning chars, {ctx.action_tool_calls} tool calls",
        )
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        return Verdict(delta=delta, reason=conf.evidence)
