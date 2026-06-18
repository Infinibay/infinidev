"""Mistral tool-call format: ``[TOOL_CALLS] [{...}, ...]``."""

from __future__ import annotations

from typing import Any

from infinidev.engine.formats._normalize import _extract_calls_from_array
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class MistralFormat(ToolCallFormat):
    """Mistral: [TOOL_CALLS] [{...}, ...]"""

    name = "mistral"
    priority = 30

    _MARKER = "[TOOL_CALLS]"

    def detect(self, text: str) -> bool:
        return self._MARKER in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        marker = text.find(self._MARKER)
        if marker == -1:
            return None
        rest = text[marker + len(self._MARKER):]

        bracket_start = rest.find("[")
        if bracket_start == -1:
            return None

        # Bracket-depth scanner: capture the FULL balanced [...] so that a
        # ``]`` inside a string or a nested array value does not truncate the
        # call (a non-greedy ``\[.*?\]`` regex would stop at the first ``]``).
        depth = 0
        end_idx = -1
        in_sq = in_dq = False
        for i, ch in enumerate(rest[bracket_start:]):
            abs_i = bracket_start + i
            if in_sq:
                if ch == "'" and rest[abs_i - 1] != "\\":
                    in_sq = False
                continue
            if in_dq:
                if ch == '"' and rest[abs_i - 1] != "\\":
                    in_dq = False
                continue
            if ch == "'":
                in_sq = True
            elif ch == '"':
                in_dq = True
            elif ch in "[{":
                depth += 1
            elif ch in "]}":
                depth -= 1
                if depth == 0:
                    end_idx = abs_i
                    break

        if end_idx == -1:
            arr_str = rest[bracket_start:]
        else:
            arr_str = rest[bracket_start : end_idx + 1]

        return _extract_calls_from_array(arr_str)
