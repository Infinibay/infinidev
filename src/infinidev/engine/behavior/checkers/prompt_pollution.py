"""Punish meta-instructional filler / prompt-template leakage."""

from __future__ import annotations

import re

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_EPHEMERAL,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    confidence_to_delta,
    regex_scan,
)


_POLLUTION_PATTERNS = {
    "as_an_ai": re.compile(r"\bas\s+an?\s+ai\b", re.IGNORECASE),
    "i_am_an_ai": re.compile(r"\bi\s+am\s+an?\s+ai\s+assistant\b", re.IGNORECASE),
    "i_will_now": re.compile(
        r"\bi\s+(will|am\s+going\s+to)\s+(now\s+)?proceed\b", re.IGNORECASE
    ),
    "step_by_step": re.compile(
        r"\blet\s+me\s+think\s+step[\s-]+by[\s-]+step\b", re.IGNORECASE
    ),
    "i_understand": re.compile(r"\bi\s+understand\s+your\s+request\b", re.IGNORECASE),
    "in_conclusion": re.compile(r"\bin\s+conclusion\b", re.IGNORECASE),
    "sures_heres": re.compile(r"\bsure[,!]?\s+here'?s\s+what\b", re.IGNORECASE),
    "template_leak": re.compile(r"<\|im_(start|end)\||\[INST\]|system:\s*$", re.IGNORECASE),
    "ignore_previous": re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
}


class PromptPollutionChecker(StochasticChecker):
    name = "prompt_pollution"
    description = "Punish meta filler like 'As an AI...' / 'I will now...'"
    default_enabled = False
    delta_range = (-1, 0)
    ttl_steps = TTL_EPHEMERAL   # filler phrase is a per-message thing
    settings_message = "Prompt pollution — punishes meta-instructional filler tokens (-1..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        conf = regex_scan(ctx.latest_content or "", _POLLUTION_PATTERNS)
        if not conf:
            return None
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        return Verdict(delta=delta, reason=f"prompt_pollution: {conf.evidence}")
