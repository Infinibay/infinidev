"""Base class for checkers that delegate the judgement to a small LLM call."""

from __future__ import annotations

import json
import logging
from typing import Any

from infinidev.engine.behavior.checker_base import BehaviorChecker, Verdict

logger = logging.getLogger(__name__)


class LLMBehaviorChecker(BehaviorChecker):
    """Behavior checker that asks an LLM to score the message.

    Subclasses set :attr:`system_prompt` (instructions describing what to
    punish or promote) and may override :attr:`max_delta` to clamp output.
    The LLM must reply with a JSON object: ``{"delta": int, "reason": str}``.
    """

    system_prompt: str = ""
    max_delta: int = 5

    def check(
        self,
        message: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> Verdict:
        from infinidev.config.llm import get_litellm_params_for_behavior
        from infinidev.engine.llm_client import call_llm

        try:
            params = get_litellm_params_for_behavior()

            user_payload = {
                "latest_message": _summarize_message(message),
                "recent_history": [_summarize_message(m) for m in history],
            }
            messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": (
                        "Score the latest assistant message. Reply ONLY with "
                        "JSON: {\"delta\": <int>, \"reason\": <short string>}.\n\n"
                        + json.dumps(user_payload, ensure_ascii=False)[:6000]
                    ),
                },
            ]
            response = call_llm(params, messages)
            content = (response.choices[0].message.content or "").strip()
            return _parse_verdict(content, self.max_delta)
        except Exception:
            logger.debug("LLMBehaviorChecker %s failed", self.name, exc_info=True)
            return Verdict(0, "")


def _summarize_message(msg: dict[str, Any]) -> dict[str, Any]:
    """Truncate a message dict to a compact, JSON-safe form.

    Preserves the parts checkers actually need:
    - role
    - content (truncated)
    - reasoning_content (truncated, for the latest assistant message)
    - tool_calls with both name AND a short preview of arguments
    - tool result content (when role == "tool")
    """
    out: dict[str, Any] = {}
    role = msg.get("role")
    if role:
        out["role"] = role

    raw = msg.get("raw_content") or msg.get("content") or ""
    if isinstance(raw, str) and raw:
        out["content"] = raw[:1500]
    elif isinstance(raw, list):
        # Multimodal content blocks — flatten text parts
        text_parts = [b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"]
        joined = " ".join(p for p in text_parts if p)
        if joined:
            out["content"] = joined[:1500]

    reasoning = msg.get("reasoning_content") or ""
    if reasoning:
        out["reasoning"] = reasoning[:800]
        out["reasoning_chars"] = len(reasoning)

    tcs = msg.get("tool_calls")
    if tcs:
        calls = []
        for t in tcs:
            name = (
                getattr(t, "name", None)
                or getattr(getattr(t, "function", None), "name", None)
            )
            args = (
                getattr(t, "arguments", None)
                or getattr(getattr(t, "function", None), "arguments", None)
                or ""
            )
            if isinstance(args, dict):
                import json as _json
                try:
                    args = _json.dumps(args)
                except Exception:
                    args = str(args)
            if name:
                calls.append({"name": name, "args": str(args)[:500]})
        if calls:
            out["tool_calls"] = calls

    # Preserve tool-result messages so checkers can see errors / outputs
    if role == "tool":
        tool_name = msg.get("name") or msg.get("tool_name") or ""
        if tool_name:
            out["tool_name"] = tool_name
    return out


def _parse_verdict(text: str, max_delta: int) -> Verdict:
    """Parse the LLM's JSON response into a Verdict, with safe fallback."""
    if not text:
        return Verdict(0, "")
    # Strip code fences if present
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    # Find first {...} block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return Verdict(0, "")
    try:
        data = json.loads(cleaned[start : end + 1])
    except Exception:
        return Verdict(0, "")
    try:
        delta = int(data.get("delta", 0) or 0)
    except (TypeError, ValueError):
        delta = 0
    delta = max(-max_delta, min(max_delta, delta))
    reason = str(data.get("reason", "") or "")[:200]
    return Verdict(delta, reason)
