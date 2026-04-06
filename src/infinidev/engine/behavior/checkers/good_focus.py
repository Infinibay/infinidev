"""Reward focused, on-plan progress (stochastic)."""

from __future__ import annotations

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_LONG,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    Confidence,
    confidence_to_delta,
    fuzzy_ratio,
)


class GoodFocusChecker(StochasticChecker):
    name = "good_focus"
    description = "Reward focused, on-plan progress with concrete actions"
    default_enabled = False
    delta_range = (0, 2)
    ttl_steps = TTL_LONG        # reward — credit should stick while weighing mistakes
    settings_message = "Good focus — rewards concrete, on-plan progress (0..+2)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        # Must have done real work this step.
        non_read = [
            c for c in ctx.tool_calls
            if c.name not in {"read_file", "partial_read", "list_directory",
                              "glob", "code_search", "get_symbol_code",
                              "list_symbols", "search_symbols", "find_references",
                              "project_structure", "help"}
        ]
        if len(non_read) < 2:
            return None
        # Summary should resemble the active step title.
        if not ctx.active_step_title or not ctx.step_summary:
            return None
        ratio = fuzzy_ratio(ctx.active_step_title, ctx.step_summary)
        if ratio < 0.35:
            return None
        # Scale ratio (0.35..1.0) into confidence (0.5..1.0)
        conf_value = 0.5 + (ratio - 0.35) / 0.65 * 0.5
        conf = Confidence(conf_value, f"on-plan progress (ratio={ratio:.2f})")
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        return Verdict(delta=delta, reason=conf.evidence)
