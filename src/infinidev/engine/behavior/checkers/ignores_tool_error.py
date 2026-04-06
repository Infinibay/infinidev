"""Punish silently ignoring a tool failure — only fires on the direct
response to a failed tool call (never re-fires on the same error)."""

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
    immediately_preceding_tool_error,
    step_complete_status,
    was_acknowledged,
)


class IgnoresToolErrorChecker(StochasticChecker):
    name = "ignores_tool_error"
    description = "Punish silently ignoring a tool failure"
    default_enabled = True
    delta_range = (-3, 0)
    ttl_steps = TTL_SHORT       # relevant for a few steps, then the agent moved on
    settings_message = "Ignores tool error — punishes carrying on as if a failed tool call worked (-3..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        err, evidence = immediately_preceding_tool_error(ctx.step_messages)
        if not err:
            return None
        # Acknowledgement clears the flag.
        if was_acknowledged(
            {
                "raw_content": ctx.latest_content,
                "reasoning_content": ctx.reasoning_content,
            }
        ):
            return None
        status = step_complete_status(ctx.tool_calls)
        boost = 0.2 if status == "done" else 0.0
        conf = Confidence(
            min(1.0, err.value + boost),
            f"unacknowledged error ({err.evidence})"
            + (" + status=done" if status == "done" else ""),
        )
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        # trigger_key fingerprints the specific tool error so the same
        # failure can never be punished twice by this checker.
        trigger_key = hashlib.md5(
            evidence.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(delta=delta, reason=conf.evidence, trigger_key=trigger_key)
