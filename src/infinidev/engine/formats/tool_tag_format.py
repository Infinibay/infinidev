"""Small-model tool-tag format: ``<tool>{...}</tool>`` / ``<tools>{...}</tools>``."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats._normalize import _extract_calls_from_fragments
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class ToolTagFormat(ToolCallFormat):
    """Fine-tuned / small models: <tool>{...}</tool> or <tools>{...}</tools>"""

    name = "tool_tag"
    priority = 51

    _RE = re.compile(r"<tools?>\s*(.*?)\s*</tools?>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        low = text.lower()
        return ("<tool>" in low or "<tools>") and ("</tool>" in low or "</tools>" in low)

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None
