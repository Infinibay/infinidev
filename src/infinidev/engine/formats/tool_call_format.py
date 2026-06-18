"""Modular tool call parsing for multiple LLM output formats.

Each format is a ToolCallFormat subclass with a `detect()` and `parse()` method.
Formats are tried in priority order (lowest number first). To add a new format,
subclass ToolCallFormat and register it with @register_format.

Built-in formats:
- Gemma4Format:        <|tool_call>call:name{args}<tool_call|>
- QwenFormat:          <tool_call>{...}</tool_call>
- QwenPipeFormat:      <|tool_call|>{...}<|/tool_call|>
- MistralFormat:       [TOOL_CALLS] [...]
- LlamaFormat:         <|python_tag|> {...}
- FunctionCallFormat:  <function_call>{...}</function_call>
- ToolTagFormat:       <tool>{...}</tool>
- AttrFunctionFormat:  <function=name>{...}</function>
- ManualJsonFormat:    {"tool_calls": [...]}  / bare JSON / markdown blocks
- SearchReplaceFormat: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE

Also provides:
- safe_json_loads() with automatic JSON repair
- ManualToolCall wrapper for uniform tool call handling
- parse_step_complete_args() for step_complete tool results
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Re-exported for backward compatibility — historically this module owned a
# local copy of safe_json_loads. The canonical definition now lives in the
# dependency-free ``_normalize`` leaf so there is a single source of truth.
from infinidev.engine.formats._normalize import safe_json_loads  # noqa: F401

# Registry for format classes — populated by @register_format decorator
_FORMATS: list[type["ToolCallFormat"]] = []


def register_format(cls: type["ToolCallFormat"]) -> type["ToolCallFormat"]:
    """Class decorator to register a tool call format."""
    _FORMATS.append(cls)
    _FORMATS.sort(key=lambda c: c.priority)
    return cls


def get_registered_formats() -> list[type["ToolCallFormat"]]:
    """Return all registered formats in priority order."""
    return list(_FORMATS)


class ToolCallFormat(ABC):
    """Base class for tool call format parsers.

    Subclass and decorate with @register_format to add a new format.

    Attributes:
        name:     Human-readable format name (for logging).
        priority: Lower = tried first. Native token formats should be 10-50,
                  generic JSON formats 100+, fallbacks 200+.
    """

    name: str = "unknown"
    priority: int = 100

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Return True if this format's markers are present in the text."""
        ...

    @abstractmethod
    def parse(self, text: str) -> list[dict[str, Any]] | None:
        """Extract tool calls from text. Return None if parsing fails."""
        ...


