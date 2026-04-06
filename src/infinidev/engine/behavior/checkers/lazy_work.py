"""Penalize evasive / lazy / non-committal model behavior (stochastic)."""

from __future__ import annotations

import hashlib
import re

from infinidev.engine.behavior.checker_base import (
    StochasticChecker,
    TTL_MEDIUM,
    Verdict,
)
from infinidev.engine.behavior.eval_context import StepEvalContext
from infinidev.engine.behavior.primitives import (
    Confidence,
    combine,
    confidence_to_delta,
    parse_file_ops,
    regex_scan,
    scan_for_todo_markers,
)


_LAZY_TEXT_PATTERNS = {
    "left_as_exercise": re.compile(
        r"(left\s+as\s+(an?\s+)?exercise|you\s+can\s+add|fill\s+in\s+the)",
        re.IGNORECASE,
    ),
    "not_implemented": re.compile(
        r"(not\s+implemented|we\s+won'?t\s+implement|skip(ping)?\s+implementation)",
        re.IGNORECASE,
    ),
    "vague_refusal": re.compile(
        r"(i\s+can'?t|cannot\s+help|unable\s+to\s+proceed|too\s+complex\s+to)",
        re.IGNORECASE,
    ),
}


class LazyWorkChecker(StochasticChecker):
    name = "lazy_work"
    description = "Penalize evasive answers, TODO placeholders, or refusal to do real work"
    default_enabled = True
    delta_range = (-3, 0)
    ttl_steps = TTL_MEDIUM      # a TODO marker is fixable within a few steps
    settings_message = "Lazy/evasive work — punishes TODOs, vague summaries, refusals (-3..0)"

    def evaluate(self, ctx: StepEvalContext) -> Verdict | None:
        # Scan every edited file's content for lazy markers.
        ops = parse_file_ops(ctx.tool_calls)
        diff_hits = [scan_for_todo_markers(op.content) for op in ops]
        text_hit = regex_scan(
            f"{ctx.latest_content}\n{ctx.reasoning_content}", _LAZY_TEXT_PATTERNS
        )
        conf = combine(*diff_hits, text_hit, mode="max")
        delta = confidence_to_delta(self.delta_range, conf, threshold=0.5)
        if delta == 0:
            return None
        # Fingerprint on the triggering content (file contents + current
        # thinking) so the same TODO marker is only punished once.
        fp_src = (
            "|".join(op.content for op in ops)
            + "|"
            + (ctx.latest_content or "")
            + "|"
            + (ctx.reasoning_content or "")
        )
        trigger_key = hashlib.md5(
            fp_src.encode("utf-8", errors="ignore")
        ).hexdigest()[:16]
        return Verdict(
            delta=delta,
            reason=f"lazy_work: {conf.evidence}",
            trigger_key=trigger_key,
        )
