"""Tool-result inspection: error detection and acknowledgement checks."""

from __future__ import annotations

import re
from typing import Any, Iterable

from infinidev.engine.behavior.primitives.scoring import Confidence
from infinidev.engine.behavior.primitives.text import keyword_presence


_ERROR_PATTERNS: dict[str, re.Pattern[str]] = {
    "traceback": re.compile(r"traceback \(most recent call last\)", re.IGNORECASE),
    "exception": re.compile(
        r"\b(error|exception|failed|failure)\b[:\s]", re.IGNORECASE
    ),
    "not_found": re.compile(
        r"\b(no such file|not found|does not exist|cannot find)\b", re.IGNORECASE
    ),
    "permission": re.compile(r"permission denied", re.IGNORECASE),
    "exit_nonzero": re.compile(r"exit(ed)?\s+(code\s+)?[1-9]", re.IGNORECASE),
    "x_prefix": re.compile(r"^\s*(x |✗|Error:)", re.IGNORECASE | re.MULTILINE),
    "stderr": re.compile(r"syntax\s*error|undefined|cannot import", re.IGNORECASE),
}


def detect_error(tool_result: str) -> Confidence:
    """Return a Confidence if *tool_result* looks like a tool failure."""
    if not tool_result:
        return Confidence.none()
    hits: list[str] = []
    for name, pat in _ERROR_PATTERNS.items():
        if pat.search(tool_result):
            hits.append(name)
    if not hits:
        return Confidence.none()
    # Multiple hit types → very high confidence.
    value = min(1.0, 0.6 + 0.15 * (len(hits) - 1))
    return Confidence(value, f"error signals: {', '.join(hits[:3])}")


def iterate_tool_results(
    messages: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return just the tool-role messages in order."""
    return [m for m in messages if m.get("role") == "tool"]


def _message_text(msg: dict[str, Any]) -> str:
    content = msg.get("content") or ""
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content if isinstance(b, dict)
        )
    return str(content)


def last_tool_error(messages: Iterable[dict[str, Any]]) -> Confidence:
    """Return the error-confidence of the *most recent* tool result.

    Returns :meth:`Confidence.none` if the last tool result is clean or
    there are no tool results at all.
    """
    results = iterate_tool_results(messages)
    if not results:
        return Confidence.none()
    return detect_error(_message_text(results[-1]))


def immediately_preceding_tool_error(
    messages: list[dict[str, Any]],
) -> tuple[Confidence, str]:
    """Return the error confidence for the tool result *immediately*
    preceding the latest assistant message, plus the triggering content.

    Unlike :func:`last_tool_error`, this only fires when the current
    assistant message is the direct response to a failed tool call — so
    in a sliding-window evaluation the same error is never observed
    twice. If the latest assistant message is not the direct continuation
    of a tool result, returns ``(Confidence.none(), "")``.
    """
    if not messages:
        return Confidence.none(), ""
    msgs = list(messages)
    # Find the latest assistant message.
    assistant_idx = -1
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "assistant":
            assistant_idx = i
            break
    if assistant_idx <= 0:
        return Confidence.none(), ""
    # The message that caused the latest assistant turn to run. We walk
    # backward through tool messages (there may be several in a row when
    # a step issued multiple tool calls) until we hit something else.
    worst = Confidence.none()
    evidence_text = ""
    i = assistant_idx - 1
    saw_any_tool = False
    while i >= 0 and msgs[i].get("role") == "tool":
        saw_any_tool = True
        text = _message_text(msgs[i])
        conf = detect_error(text)
        if conf.value > worst.value:
            worst = conf
            evidence_text = text[:200]
        i -= 1
    if not saw_any_tool:
        return Confidence.none(), ""
    return worst, evidence_text


_ACK_KEYWORDS: set[str] = {
    "error", "failed", "retry", "fix", "instead", "try",
    "oops", "sorry", "let me", "that didn", "issue", "problem",
}


def was_acknowledged(next_assistant_msg: dict[str, Any]) -> bool:
    """True if *next_assistant_msg* contains an acknowledgement of a failure."""
    if not next_assistant_msg:
        return False
    text = (
        next_assistant_msg.get("raw_content")
        or next_assistant_msg.get("content")
        or ""
    )
    reasoning = next_assistant_msg.get("reasoning_content") or ""
    combined = f"{text}\n{reasoning}"
    return bool(keyword_presence(combined, _ACK_KEYWORDS))
