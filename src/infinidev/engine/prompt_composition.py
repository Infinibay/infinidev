"""Content-free measurements of the prompt sections sent by an agent loop."""

from __future__ import annotations

import json
import re
from typing import Any


_TOP_LEVEL_OPEN = re.compile(
    r"(?:\A|\n\n)<([a-z][a-z0-9-]*)(?:\s[^>]*)?>\n",
    re.IGNORECASE,
)


def user_section_chars(prompt: str) -> dict[str, int]:
    """Measure top-level XML-like blocks without retaining their contents."""
    counts: dict[str, int] = {}
    covered = 0
    for match in _TOP_LEVEL_OPEN.finditer(prompt):
        tag = match.group(1).lower()
        start = match.start()
        if prompt.startswith("\n\n", start):
            start += 2
        close = re.compile(rf"\n</{re.escape(tag)}>(?=\n\n|\Z)", re.IGNORECASE)
        closing = close.search(prompt, match.end())
        if closing is None:
            continue
        end = closing.end()
        counts[tag] = counts.get(tag, 0) + end - start
        covered += end - start
    counts["unclassified"] = max(0, len(prompt) - covered)
    return dict(sorted(counts.items()))


def measure_prompt_composition(
    system_prompt: str,
    user_prompt: str,
    tool_schemas: list[dict[str, Any]] | None,
    *,
    iteration: int,
) -> dict[str, Any]:
    """Return exact character counts for one request's static components."""
    encoded_tools = json.dumps(
        tool_schemas or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    user_sections = user_section_chars(user_prompt)
    from infinidev.engine.prompt_layers import classify_user_section

    layer_chars: dict[str, int] = {}
    for tag, count in user_sections.items():
        if tag == "unclassified":
            continue
        layer = classify_user_section(tag).value
        layer_chars[layer] = layer_chars.get(layer, 0) + count
    return {
        "iteration": iteration,
        "system_chars": len(system_prompt),
        "user_chars": len(user_prompt),
        "tool_schema_chars": len(encoded_tools),
        "request_static_chars": len(system_prompt) + len(user_prompt) + len(encoded_tools),
        "user_sections": user_sections,
        "user_layer_chars": dict(sorted(layer_chars.items())),
    }


def measure_request_payload(
    messages: list[dict[str, Any]],
    tool_schemas: list[dict[str, Any]] | None,
    *,
    mode: str,
    sequence: int,
) -> dict[str, Any]:
    """Measure the complete message transcript immediately before dispatch."""
    encoded_messages = json.dumps(
        messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    encoded_tools = json.dumps(
        tool_schemas or [], ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    role_chars: dict[str, int] = {}
    for message in messages:
        role = str(message.get("role", "unknown"))
        content = json.dumps(
            message.get("content"), ensure_ascii=False, separators=(",", ":"), default=str
        )
        role_chars[role] = role_chars.get(role, 0) + len(content)
    return {
        "sequence": sequence,
        "mode": mode,
        "message_count": len(messages),
        "message_payload_chars": len(encoded_messages),
        "tool_schema_chars": len(encoded_tools),
        "request_payload_chars": len(encoded_messages) + len(encoded_tools),
        "message_content_chars_by_role": dict(sorted(role_chars.items())),
    }
