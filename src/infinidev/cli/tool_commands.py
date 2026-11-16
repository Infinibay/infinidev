"""Atomic tool-selection commands with an opaque task body."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from infinidev.engine.turns.catalog import RUNTIME_TOOLS, effective_tools


@dataclass(frozen=True)
class ToolRequest:
    """One literal task and its names, never a sticky next-turn setting."""

    names: tuple[str, ...]
    body: str


def is_tools_command(text: str) -> bool:
    """Recognize only the explicit control-command token."""
    parts = text.split(maxsplit=1)
    return bool(parts and parts[0].lower() == '/tools')


def parse_tool_request(text: str) -> ToolRequest:
    """Parse control syntax only; never interpret or strip the task body."""
    header, delimiter, body = text.partition(' -- ')
    parts = header.split()
    if (not delimiter or not body.strip() or [part.lower() for part in parts[:2]] != ['/tools', 'run']
            or len(parts) not in (2, 3)):
        raise ValueError('Usage: /tools run [name1,name2] -- <task>')
    names = tuple(parts[2].split(',')) if len(parts) == 3 else ()
    if any(not name for name in names):
        raise ValueError('Tool names must not be empty')
    return ToolRequest(names, body)


def describe_tools(tools: list[Any]) -> str:
    """Render the effective catalog without a model request."""
    names = ', '.join(tool.name for tool in effective_tools(tools)) or '(none)'
    return (f'Available tools: {names}\nAlways included: {", ".join(sorted(RUNTIME_TOOLS))}'
            '\nUsage: /tools run [name1,name2] -- <task>')


def describe_selection(tools: tuple[Any, ...]) -> str:
    """Show the selected and mandatory tool names."""
    names = [tool.name for tool in tools] + sorted(RUNTIME_TOOLS)
    return 'Tools for this task: ' + ', '.join(names)
