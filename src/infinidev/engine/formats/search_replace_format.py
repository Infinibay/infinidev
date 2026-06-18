"""Aider-style SEARCH/REPLACE block format → ``edit_file`` tool calls."""

from __future__ import annotations

import re
from typing import Any

from infinidev.engine.formats.tool_call_format import ToolCallFormat, register_format


@register_format
class SearchReplaceFormat(ToolCallFormat):
    """Aider-style SEARCH/REPLACE blocks → edit_file tool calls."""

    name = "search_replace"
    priority = 200

    _RE = re.compile(
        r"(?:^([^\n<>]+\.[\w]+)\n)?"
        r"<{4,}\s*SEARCH"
        r"(?:[@\s]+([^\n]*\.[\w]+))?"
        r"(?:[@\s]*(\d+)(?:-\d+)?)?\s*\n"
        r"(.*?)\n"
        r"={4,}\s*\n"
        r"(.*?)\n"
        r">{4,}\s*REPLACE",
        re.DOTALL | re.MULTILINE,
    )

    def detect(self, text: str) -> bool:
        return "SEARCH" in text and "REPLACE" in text and "=====" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for m in self._RE.finditer(text):
            path = m.group(1) or m.group(2) or ""
            old_string = m.group(4)
            new_string = m.group(5)
            if old_string is not None and new_string is not None:
                args: dict[str, Any] = {"old_string": old_string, "new_string": new_string}
                if path:
                    args["path"] = path.strip()
                calls.append({"name": "edit_file", "arguments": args})
        return calls if calls else None
