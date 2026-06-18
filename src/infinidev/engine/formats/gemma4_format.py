"""Gemma 4 tool-call format: ``<|tool_call>call:func_name{args}<tool_call|>``."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats._normalize import _fix_jslike_json
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class Gemma4Format(ToolCallFormat):
    """Gemma 4: <|tool_call>call:func_name{args}<tool_call|>"""

    name = "gemma4"
    priority = 10

    _PREFIX = re.compile(r"<\|tool_call>call:([a-z_]\w*)")

    def detect(self, text: str) -> bool:
        return "<|tool_call>call:" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []

        for m in self._PREFIX.finditer(text):
            name = m.group(1)
            rest = text[m.end():]

            brace_start = rest.find("{")
            if brace_start == -1:
                calls.append({"name": name, "arguments": {}})
                continue

            # Brace-depth match (handles nested objects/arrays)
            depth = 0
            end_idx = -1
            in_sq = in_dq = False
            for i, ch in enumerate(rest[brace_start:]):
                if in_sq:
                    if ch == "'" and (i == 0 or rest[brace_start + i - 1] != "\\"):
                        in_sq = False
                    continue
                if in_dq:
                    if ch == '"' and (i == 0 or rest[brace_start + i - 1] != "\\"):
                        in_dq = False
                    continue
                if ch == "'":
                    in_sq = True
                elif ch == '"':
                    in_dq = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        end_idx = brace_start + i
                        break

            if end_idx == -1:
                args_str = rest[brace_start:]
                cut = args_str.find("<tool_call|>")
                if cut != -1:
                    args_str = args_str[:cut]
            else:
                args_str = rest[brace_start : end_idx + 1]

            calls.append({"name": name, "arguments": _fix_jslike_json(args_str)})

        return calls if calls else None
