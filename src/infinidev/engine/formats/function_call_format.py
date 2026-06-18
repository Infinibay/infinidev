"""Generic function-call format: ``<function_call>{...}</function_call>``."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats._normalize import _extract_calls_from_fragments
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class FunctionCallFormat(ToolCallFormat):
    """Generic: <function_call>{...}</function_call>"""

    name = "function_call"
    priority = 50

    _RE = re.compile(r"<function_?call>\s*(.*?)\s*</function_?call>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        low = text.lower()
        return "<functioncall>" in low or "<function_call>" in low

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None
