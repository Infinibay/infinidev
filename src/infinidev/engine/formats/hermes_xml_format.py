"""Hermes / OpenHands nested XML tool-call format.

Qwen3 Hermes models (and derivatives like Qwen3.6-A3B) occasionally emit
tool calls as nested XML instead of the JSON-in-<tool_call> form the
QwenFormat handles. Example:

    <tool_call>
    <function=execute_command>
    <parameter=command>
    cd /tmp && ls -la
    </parameter>
    </function>
    </tool_call>

Also appears without the outer <tool_call> wrapper, and with multiple
<parameter=...> children for multi-arg tools. Must run BEFORE QwenFormat
and AttrFunctionFormat, which would otherwise try to parse the inner
text as JSON and silently drop the call.
"""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats.tool_call_format import (
    ToolCallFormat,
    register_format,
)


_FUNC_RE = re.compile(
    r"<function=([a-zA-Z_]\w*)>(.*?)</function>",
    re.DOTALL,
)
_PARAM_RE = re.compile(
    r"<parameter=([a-zA-Z_]\w*)>(.*?)</parameter>",
    re.DOTALL,
)


@register_format
class HermesXmlFormat(ToolCallFormat):
    """Hermes-style nested XML: <function=NAME><parameter=KEY>VAL</parameter></function>."""

    name = "hermes_xml"
    priority = 15

    def detect(self, text: str) -> bool:
        return "<function=" in text and "<parameter=" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for name, body in _FUNC_RE.findall(text):
            args: dict[str, Any] = {}
            for key, raw in _PARAM_RE.findall(body):
                args[key] = raw.strip("\n")
            calls.append({"name": name, "arguments": args})
        return calls if calls else None
