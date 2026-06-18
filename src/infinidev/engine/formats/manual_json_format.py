"""Manual-mode JSON format: ``{"tool_calls": [...]}``, bare JSON, or markdown blocks."""

from __future__ import annotations

import json
import re
from typing import Any

from infinidev.engine.formats._normalize import _normalize_call_list, safe_json_loads
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class ManualJsonFormat(ToolCallFormat):
    """Manual-mode JSON: {"tool_calls": [...]}, bare JSON, or markdown blocks."""

    name = "manual_json"
    priority = 100

    def detect(self, text: str) -> bool:
        return "{" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        candidates: list[str] = []

        # Markdown code blocks
        code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        candidates.extend(code_blocks)
        candidates.append(text.strip())

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            brace_start = candidate.find("{")
            if brace_start == -1:
                continue

            depth = 0
            balanced_parsed = False
            for i, ch in enumerate(candidate[brace_start:], start=brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = candidate[brace_start : i + 1]
                        try:
                            parsed = safe_json_loads(json_str)
                            if isinstance(parsed, dict):
                                if "tool_calls" in parsed:
                                    calls = _normalize_call_list(parsed["tool_calls"])
                                    if calls:
                                        return calls
                                if "name" in parsed:
                                    calls = _normalize_call_list([parsed])
                                    if calls:
                                        return calls
                        except (json.JSONDecodeError, TypeError):
                            pass
                        balanced_parsed = True
                        continue

            # Truncated JSON recovery: if we never reached a balanced
            # object (depth stayed > 0 through EOF), the model was cut
            # off mid-emission. json_repair can usually reconstruct the
            # call from partial input. Worth one final attempt.
            if not balanced_parsed and depth > 0:
                try:
                    parsed = safe_json_loads(candidate[brace_start:])
                    if isinstance(parsed, dict):
                        if "tool_calls" in parsed:
                            calls = _normalize_call_list(parsed["tool_calls"])
                            if calls:
                                return calls
                        if "name" in parsed:
                            calls = _normalize_call_list([parsed])
                            if calls:
                                return calls
                except (json.JSONDecodeError, TypeError):
                    pass

        return None
