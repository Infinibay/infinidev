"""Reward changing approach after a tool failure."""

from __future__ import annotations

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_LONG,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
import hashlib

from infinidev.engine.behavior.primitives import (
    Confidence,
    detect_error,
    was_acknowledged,
)


class GracefulRecoveryChecker(StochasticChecker):
    name = "graceful_recovery"
    description = "Reward switching strategy after a tool error instead of repeating it"
    default_enabled = True
    delta_range = (0, 2)
    ttl_steps = TTL_LONG        # reward — recovery deserves lasting credit
    settings_message = "Graceful recovery — rewards changing approach after a failed tool call (0..+2)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        # Look at the PRIOR step record for an error signal.
        if not ctx.prior_step_records:
            return None
        prior = ctx.prior_step_records[-1]
        prior_anti = getattr(prior, "anti_patterns", "") or ""
        prior_summary = getattr(prior, "summary", "") or ""
        prior_err = detect_error(f"{prior_anti}\n{prior_summary}")
        if not prior_err:
            return None
        # Did this step acknowledge the failure?
        ack = was_acknowledged(
            {
                "raw_content": ctx.latest_content,
                "reasoning_content": ctx.reasoning_content,
            }
        )
        if not ack:
            return None
        # Did this step use a *different* set of tools than the previous
        # one's anti-pattern hints at?
        prior_changes = getattr(prior, "changes_made", "") or ""
        current_tools = {c.name for c in ctx.tool_calls}
        different_tool = bool(current_tools) and not any(
            name in prior_changes for name in current_tools
        )
        if not different_tool and len(current_tools) < 2:
            return None
        value = 0.9 if different_tool and len(current_tools) >= 2 else 0.6
        conf = Confidence(value, "recovered after prior failure")
        # Reward-only range: reuse the helper.
        from infinidev.engine.behavior.primitives import confidence_to_delta

        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        trigger_key = hashlib.md5(
            f"{prior_anti}|{prior_summary}".encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(delta=delta, reason=conf.evidence, trigger_key=trigger_key)
