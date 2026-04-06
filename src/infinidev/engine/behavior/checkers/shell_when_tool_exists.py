"""Punish execute_command when a dedicated tool exists."""

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
    detect_shell_hack,
    filter_by_name,
)


class ShellWhenToolExistsChecker(StochasticChecker):
    name = "shell_when_tool_exists"
    description = "Punish execute_command when a dedicated tool exists"
    default_enabled = True
    delta_range = (-2, 0)
    ttl_steps = TTL_SHORT       # specific bad tool choice, pattern fades quickly
    settings_message = "Shell-when-tool-exists — punishes shell hacks for cat/grep/find/git status (-2..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        shell_calls = filter_by_name(ctx.tool_calls, "execute_command")
        if not shell_calls:
            return None
        hits: list[str] = []
        strongest = 0.0
        strongest_cmd = ""
        for c in shell_calls:
            cmd = c.args.get("command", "") if isinstance(c.args, dict) else ""
            conf = detect_shell_hack(str(cmd))
            if conf.value > strongest:
                strongest = conf.value
                strongest_cmd = str(cmd)
            if conf.evidence:
                hits.append(conf.evidence)
        if strongest == 0.0:
            return None
        conf = Confidence(strongest, hits[0] if hits else "shell hack detected")
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        trigger_key = hashlib.md5(
            strongest_cmd.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(delta=delta, reason=conf.evidence, trigger_key=trigger_key)
