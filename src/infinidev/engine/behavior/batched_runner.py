"""Batched runner — evaluate every PromptBehaviorChecker in one LLM call.

Instead of N independent LLM calls (one per checker), this builds a single
system prompt that lists all enabled criteria and asks the model to reply
with one JSON object keyed by checker name.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.behavior.checker_base import (
    BehaviorChecker,
    PromptBehaviorChecker,
    Verdict,
)
from infinidev.engine.behavior.llm_checker import _summarize_message

logger = logging.getLogger(__name__)


_BASE_SYSTEM = (
    "You are a strict reward-and-punishment judge for an AI coding agent. "
    "You will be given:\n"
    "- task: the original user instruction the agent is working on\n"
    "- plan: the agent's current step-by-step plan and which step is active\n"
    "- latest_message: the agent's most recent message (thinking + tool calls)\n"
    "- recent_history: the last few turns of the conversation, including "
    "  tool results (role=\"tool\") with their full output\n"
    "You must evaluate the latest_message against several CRITERIA listed below.\n\n"
    "For EACH criterion, decide an integer delta within the allowed range:\n"
    "  - Negative delta = punish (the agent is doing the bad thing).\n"
    "  - Positive delta = promote (the agent is doing the good thing).\n"
    "  - Zero            = neither, no signal.\n\n"
    "Be strict: only return a non-zero delta if there is clear evidence in "
    "the latest message. Do not punish twice for the same problem across "
    "different criteria.\n\n"
    "Reply with ONE JSON object keyed by criterion name:\n"
    '{\n'
    '  "<criterion_name>": {"delta": <int>, "reason": "<short string>"},\n'
    '  ...\n'
    '}\n'
    "Do not wrap it in markdown. Do not add extra keys."
)


def run_batched(
    checkers: list[PromptBehaviorChecker],
    message: dict[str, Any],
    history: list[dict[str, Any]],
    task: str = "",
    plan_snapshot: dict[str, Any] | None = None,
) -> dict[str, Verdict]:
    """Run a single LLM call covering all *checkers* and return verdicts by name."""
    if not checkers:
        return {}
    try:
        from infinidev.config.llm import get_litellm_params_for_behavior
        from infinidev.engine.llm_client import call_llm

        params = get_litellm_params_for_behavior()

        criteria_block = _format_criteria(checkers)
        payload: dict[str, Any] = {
            "task": (task or "")[:1500],
            "plan": plan_snapshot or {},
            "latest_message": _summarize_message(message),
            "recent_history": [_summarize_message(m) for m in history],
        }
        messages = [
            {"role": "system", "content": _BASE_SYSTEM + "\n\n" + criteria_block},
            {
                "role": "user",
                "content": (
                    "Score the latest assistant message against every criterion "
                    "above. Reply ONLY with the JSON object described.\n\n"
                    + json.dumps(payload, ensure_ascii=False)[:8000]
                ),
            },
        ]
        response = call_llm(params, messages)
        content = (response.choices[0].message.content or "").strip()
        return _parse_batched_verdicts(content, checkers)
    except Exception:
        logger.debug("Batched behavior runner failed", exc_info=True)
        return {}


def _format_criteria(checkers: list[PromptBehaviorChecker]) -> str:
    lines = ["CRITERIA:"]
    for c in checkers:
        lo, hi = c.delta_range
        criteria_text = (c.criteria or c.description or c.name).strip()
        lines.append(
            f'- name="{c.name}" range=[{lo}..{hi}]\n'
            f'  rule: {criteria_text}'
        )
    return "\n".join(lines)


def _parse_batched_verdicts(
    text: str,
    checkers: list[PromptBehaviorChecker],
) -> dict[str, Verdict]:
    if not text:
        return {}
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        data = json.loads(cleaned[start : end + 1])
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}

    by_name = {c.name: c for c in checkers}
    out: dict[str, Verdict] = {}
    for name, entry in data.items():
        checker = by_name.get(name)
        if checker is None or not isinstance(entry, dict):
            continue
        try:
            delta = int(entry.get("delta", 0) or 0)
        except (TypeError, ValueError):
            delta = 0
        lo, hi = checker.delta_range
        delta = max(lo, min(hi, delta))
        reason = str(entry.get("reason", "") or "")[:200]
        if delta == 0 and not reason:
            continue
        out[name] = Verdict(delta, reason)
    return out
