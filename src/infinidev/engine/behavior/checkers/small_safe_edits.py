"""Reward small, scoped edits over massive rewrites."""

from __future__ import annotations

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_LONG,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    Confidence,
    biggest_op,
    confidence_to_delta,
    parse_file_ops,
)


_SCOPED_TOOLS = {"edit_file"}


class SmallSafeEditsChecker(StochasticChecker):
    name = "small_safe_edits"
    description = "Reward scoped edit_file swaps vs full-file rewrites"
    default_enabled = True
    delta_range = (0, 2)
    ttl_steps = TTL_LONG        # reward — good hygiene accumulates
    settings_message = "Small safe edits — rewards scoped edit_file swaps (0..+2)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        ops = parse_file_ops(ctx.tool_calls)
        if not ops:
            return None
        biggest = biggest_op(ops)
        if biggest is None:
            return None
        max_delta = max(biggest.lines_added, biggest.lines_removed)

        # Reward scoped edits under 30 lines.
        if biggest.tool in _SCOPED_TOOLS and max_delta <= 30:
            conf = Confidence(0.9, f"{biggest.tool} ≤30 lines on {biggest.path}")
        # Reward small new files under 80 lines.
        elif biggest.tool == "create_file" and max_delta <= 80:
            conf = Confidence(0.6, f"create_file ≤80 lines ({biggest.path})")
        else:
            return None
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        return Verdict(delta=delta, reason=conf.evidence)
