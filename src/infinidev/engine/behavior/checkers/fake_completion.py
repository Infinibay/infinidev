"""Punish marking a step done when recent tool results show errors."""

from __future__ import annotations

import hashlib

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_MEDIUM,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    Confidence,
    confidence_to_delta,
    detect_error,
    iterate_tool_results,
    step_complete_status,
)


class FakeCompletionChecker(StochasticChecker):
    name = "fake_completion"
    description = "Punish step_complete status=done when recent tool results show errors"
    default_enabled = True
    delta_range = (-3, 0)
    ttl_steps = TTL_MEDIUM      # serious but forgivable with recovery
    settings_message = "Fake completion — punishes status=done while recent errors are unresolved (-3..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        status = step_complete_status(ctx.tool_calls)
        if status != "done":
            return None
        # Scan the last 3 tool results within this step for errors.
        results = iterate_tool_results(ctx.step_messages)[-3:]
        worst = Confidence.none()
        worst_text = ""
        for r in results:
            content = r.get("content") or ""
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content if isinstance(b, dict)
                )
            err = detect_error(str(content))
            if err.value > worst.value:
                worst = err
                worst_text = str(content)[:200]
        if not worst:
            return None
        conf = Confidence(
            min(1.0, worst.value + 0.1),
            f"status=done with recent error ({worst.evidence})",
        )
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        trigger_key = hashlib.md5(
            worst_text.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(delta=delta, reason=conf.evidence, trigger_key=trigger_key)
