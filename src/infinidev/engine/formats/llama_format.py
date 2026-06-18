"""Llama tool-call format: ``<|python_tag|>`` followed by JSON."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats._normalize import _extract_calls_from_fragments
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class LlamaFormat(ToolCallFormat):
    """Llama: <|python_tag|> followed by JSON"""

    name = "llama"
    priority = 40

    # Capture the payload after each <|python_tag|>, bounded at the first
    # turn/segment terminator (<|eom_id|> / <|eot_id|>) or the next
    # <|python_tag|>. A bare greedy ``(.*)`` with DOTALL would swallow
    # everything to EOF — pulling in trailing prose and merging what should
    # be sequential, independent tool calls into one un-parseable blob.
    _RE = re.compile(
        r"<\|python_tag\|>\s*(.*?)\s*(?=<\|eom_id\|>|<\|eot_id\|>|<\|python_tag\|>|$)",
        re.DOTALL,
    )

    def detect(self, text: str) -> bool:
        return "<|python_tag|>" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        fragments = [m for m in self._RE.findall(text) if m.strip()]
        return _extract_calls_from_fragments(fragments) if fragments else None
