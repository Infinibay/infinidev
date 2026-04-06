"""Bidirectional: reward concrete plans, punish vague or bloated ones."""

from __future__ import annotations

import re

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_INFINITE,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext


_VAGUE_TITLES = re.compile(
    r"^(explore|understand|investigate|research|look\s+(at|into)|review|"
    r"figure\s+out|do\s+stuff|work\s+on|handle|deal\s+with|fix\s+(it|this)|"
    r"continue|finish|complete|things?\s+to\s+do)\b",
    re.IGNORECASE,
)

_CONCRETE_HINT = re.compile(
    r"(\.\w+|/|_|\b(add|create|delete|refactor|replace|rename|implement|"
    r"wire|register|import)\b\s+\w+)",
    re.IGNORECASE,
)


class PlanQualityChecker(StochasticChecker):
    name = "plan_quality"
    description = "Reward concrete plans (+1..+2); punish vague or bloated ones (-2..-1)"
    default_enabled = True
    delta_range = (-2, 2)
    ttl_steps = TTL_INFINITE    # plan quality is structural — the plan IS the plan
    settings_message = "Plan quality — bidirectional: rewards concrete plans, punishes vague/bloated (-2..+2)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        steps = ctx.plan_steps or []
        if not steps:
            return None
        # Only fire once shortly after planning — we look for a plan that
        # just took shape this step. A rough heuristic: only evaluate
        # during the first couple of iterations of the task.
        if ctx.iteration > 2:
            return None

        n = len(steps)
        titles = [str(s.get("title", "")).strip() for s in steps]
        avg_len = sum(len(t) for t in titles) / max(1, n)
        vague_count = sum(1 for t in titles if _VAGUE_TITLES.match(t) or len(t) < 8)
        concrete_count = sum(1 for t in titles if _CONCRETE_HINT.search(t))

        # Vague/bloated path
        if n >= 7 and concrete_count < n // 2:
            return Verdict(-2, f"bloated plan ({n} steps, few concrete)")
        if vague_count >= max(1, n // 2):
            return Verdict(-1, f"vague plan ({vague_count}/{n} vague titles)")

        # Concrete path
        if 2 <= n <= 4 and avg_len >= 15 and concrete_count >= n - 1:
            return Verdict(+2, f"concrete plan ({n} steps, avg={avg_len:.0f} chars)")
        if 2 <= n <= 5 and concrete_count >= 1:
            return Verdict(+1, f"decent plan ({n} steps)")
        return None
