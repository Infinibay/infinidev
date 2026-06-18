"""Qwen / GLM tool-call format: ``<tool_call>{...}</tool_call>``."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats._normalize import _extract_calls_from_fragments
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class QwenFormat(ToolCallFormat):
    """Qwen / GLM: <tool_call>{...}</tool_call>"""

    name = "qwen"
    priority = 20

    _RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def detect(self, text: str) -> bool:
        return "<tool_call>" in text and "</tool_call>" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None
