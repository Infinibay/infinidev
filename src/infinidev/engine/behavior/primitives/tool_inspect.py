"""Tool-call inspection primitives.

Normalizes the two tool-call shapes that appear in the engine:

1. Objects from the litellm response (``tc.function.name`` /
   ``tc.function.arguments`` as a JSON string).
2. Plain dicts the engine builds in manual-tool-call mode
   (``{"name": ..., "arguments": {...}}`` or
   ``{"function": {"name": ..., "arguments": "..."}}``).

Checkers should always go through :func:`normalize_tool_calls` rather
than poking at the raw shapes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Iterator

from infinidev.engine.behavior.primitives.scoring import Confidence


@dataclass
class NormalizedCall:
    name: str
    args: dict[str, Any]
    raw_args: str = ""   # verbatim string (useful for regex)


def _coerce_args(raw: Any) -> tuple[dict[str, Any], str]:
    if raw is None:
        return {}, ""
    if isinstance(raw, dict):
        try:
            return raw, json.dumps(raw, ensure_ascii=False)
        except Exception:
            return raw, str(raw)
    if isinstance(raw, str):
        s = raw
        try:
            parsed = json.loads(s)
            if isinstance(parsed, dict):
                return parsed, s
        except Exception:
            pass
        return {}, s
    return {}, str(raw)


def normalize_tool_calls(tool_calls: Any) -> list[NormalizedCall]:
    """Return a list of :class:`NormalizedCall` regardless of input shape."""
    if not tool_calls:
        return []
    out: list[NormalizedCall] = []
    for tc in tool_calls:
        name = (
            getattr(tc, "name", None)
            or getattr(getattr(tc, "function", None), "name", None)
        )
        raw_args = (
            getattr(tc, "arguments", None)
            or getattr(getattr(tc, "function", None), "arguments", None)
        )
        # Dict form
        if name is None and isinstance(tc, dict):
            name = tc.get("name") or (tc.get("function") or {}).get("name")
            raw_args = (
                tc.get("arguments")
                if tc.get("arguments") is not None
                else (tc.get("function") or {}).get("arguments")
            )
        if not name:
            continue
        args, raw_str = _coerce_args(raw_args)
        out.append(NormalizedCall(name=str(name), args=args, raw_args=raw_str))
    return out


def iterate_messages_tool_calls(
    messages: Iterable[dict[str, Any]],
) -> Iterator[NormalizedCall]:
    """Yield every tool call across a list of messages (assistant role only)."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        for call in normalize_tool_calls(m.get("tool_calls")):
            yield call


def filter_by_name(
    calls: Iterable[NormalizedCall], *names: str
) -> list[NormalizedCall]:
    name_set = set(names)
    return [c for c in calls if c.name in name_set]


# Pre-compiled patterns for shell-hack detection.
_SHELL_HACK_PATTERNS: dict[str, re.Pattern[str]] = {
    "read": re.compile(r"\b(cat|head|tail|less|more)\b\s+[^|;&]*\.", re.IGNORECASE),
    "search": re.compile(r"\b(grep|rg|egrep|fgrep)\b", re.IGNORECASE),
    "list": re.compile(r"\b(find|ls)\b", re.IGNORECASE),
    "git_read": re.compile(
        r"\bgit\s+(status|diff|log|branch|show)\b", re.IGNORECASE
    ),
    "edit": re.compile(r"\b(sed|awk)\b\s+-i", re.IGNORECASE),
}


def detect_shell_hack(command: str) -> Confidence:
    """Return a Confidence if *command* looks like a shell hack for a
    task with a dedicated tool (cat/grep/find/git status/sed…)."""
    if not command:
        return Confidence.none()
    hits: list[str] = []
    for name, pat in _SHELL_HACK_PATTERNS.items():
        if pat.search(command):
            hits.append(name)
    if not hits:
        return Confidence.none()
    # Even one clear match is high confidence.
    return Confidence(0.9, f"shell hack ({'/'.join(hits)}): {command[:60]}")


def step_complete_status(calls: Iterable[NormalizedCall]) -> str | None:
    """Return the ``status`` argument of any ``step_complete`` call in
    *calls*, or ``None`` if not present."""
    for c in calls:
        if c.name == "step_complete":
            status = c.args.get("status")
            if isinstance(status, str):
                return status
    return None
