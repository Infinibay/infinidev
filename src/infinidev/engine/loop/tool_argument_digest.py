"""Shrink the tool-call arguments the loop re-sends for already-closed Steps.

The measurement that motivates this: on a task that writes files, the serialized
``tool_calls`` of the assistant turns are **up to 33 % of the request payload**
(21 127 characters in one ``complex-plan`` run, against a 63 918-character
request). The body of a written file *is* the argument, so from the round that
writes it onward, every remaining request replays the file.

Nothing in the protocol requires those bytes to be the original ones. A provider
needs the ``id`` and the ``name`` so that the ``tool`` result has a call to
answer; the argument is the model's own previous output. And the loop already
has three cheaper records of the same fact: the tool *result* message, the step
summary that replaces the transcript when the step closes, and the archive that
``recall_context`` searches.

What this does is narrower than dropping the arguments. It keeps every key and
the whole JSON shape, and replaces only *long string values* with a placeholder
that says how long the elided body was. A provider that validates the shape
still sees the shape; the model still sees what it called and where; it no
longer sees a file it already wrote.

Only closed turns are touched — every assistant turn except the newest, whose
tool results travel in the same request. That is the same boundary the reasoning
trim uses, and it is the boundary the loop already treats as "resolved": the
step summary takes over from the transcript there.
"""

from __future__ import annotations

import json
from typing import Any

#: String values longer than this inside a closed call are elided. Chosen above
#: the size of a path, a command line or a short pattern, and well below a file
#: body, so a call keeps its identity and loses its payload.
_ELIDE_OVER_CHARS = 400

#: The marker a model reads in place of a body. It deliberately names the size,
#: so a model that wants the content back knows whether re-reading is worth a
#: round, and names the tool that has it.
_ELIDE_TEMPLATE = "<elided by infinidev: {size} chars; re-read the file or use recall_context>"


def _elide_value(value: Any) -> tuple[Any, int]:
    """Replace long strings in *value*, returning the new value and chars saved."""
    if isinstance(value, str):
        if len(value) <= _ELIDE_OVER_CHARS:
            return value, 0
        marker = _ELIDE_TEMPLATE.format(size=len(value))
        return marker, len(value) - len(marker)
    if isinstance(value, list):
        saved = 0
        result = []
        for item in value:
            replaced, delta = _elide_value(item)
            result.append(replaced)
            saved += delta
        return result, saved
    if isinstance(value, dict):
        saved = 0
        result: dict[str, Any] = {}
        for key, item in value.items():
            replaced, delta = _elide_value(item)
            result[key] = replaced
            saved += delta
        return result, saved
    return value, 0


def digest_arguments(arguments: Any) -> tuple[str, int]:
    """Return *arguments* with long string values elided, and the chars saved.

    Anything that is not parseable JSON is left exactly as it is: a malformed
    call is a fact about the run, and rewriting it would both hide that and risk
    turning a diagnosable failure into a different one.
    """
    if not isinstance(arguments, str):
        return arguments, 0
    try:
        parsed = json.loads(arguments)
    except (TypeError, ValueError):
        return arguments, 0
    digested, saved = _elide_value(parsed)
    if not saved:
        return arguments, 0
    return json.dumps(digested, ensure_ascii=False, separators=(",", ":")), saved


def trim_superseded_tool_arguments(messages: list[dict[str, Any]]) -> int:
    """Elide the arguments of every tool call whose Step has already closed.

    Returns the number of characters removed. The newest assistant turn is left
    alone, because its tool results are about to be sent alongside it and a
    provider is entitled to see the call it is answering in full.
    """
    newest = -1
    for index, message in enumerate(messages):
        if isinstance(message, dict) and message.get("role") == "assistant":
            newest = index
    if newest < 0:
        return 0

    saved = 0
    for message in messages[:newest]:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        calls = message.get("tool_calls")
        if not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, dict):
                continue
            function = call.get("function")
            if not isinstance(function, dict) or "arguments" not in function:
                continue
            digested, delta = digest_arguments(function["arguments"])
            if delta:
                function["arguments"] = digested
                saved += delta
    return saved


__all__ = [
    "digest_arguments",
    "trim_superseded_tool_arguments",
]
