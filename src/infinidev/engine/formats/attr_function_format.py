"""Attribute-style tool-call format: ``<function=name>{...}</function>``."""

from __future__ import annotations

import json
import re
from typing import Any

from infinidev.engine.formats._normalize import safe_json_loads
from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class AttrFunctionFormat(ToolCallFormat):
    """Attribute-style: <function=tool_name>{"arg": "val"}</function>"""

    name = "attr_function"
    priority = 52

    _RE = re.compile(r"<function=([a-z_]\w*)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        return "<function=" in text.lower()

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        if not matches:
            return None
        calls = []
        for name, args_str in matches:
            try:
                args = safe_json_loads(args_str)
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            except (json.JSONDecodeError, TypeError):
                calls.append({"name": name, "arguments": {}})
        return calls if calls else None
