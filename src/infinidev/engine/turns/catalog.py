"""Pure principal tool selection and immutable per-request membership."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

LEGACY_TOOLS = frozenset({
    'step_complete', 'add_step', 'modify_step', 'remove_step', 'escalate',
    'respond', 'continue_investigation',
})
RUNTIME_TOOLS = frozenset({'finish_turn', 'update_plan'})


def effective_tools(tools: Sequence[Any]) -> list[Any]:
    """Keep the existing principal catalog order and exclusion behavior."""
    return [tool for tool in tools
            if tool.name not in LEGACY_TOOLS and tool.name not in RUNTIME_TOOLS]


@dataclass(frozen=True)
class ToolGrant(Sequence[Any]):
    """A parent-owned membership snapshot, not a filesystem sandbox."""

    tools: tuple[Any, ...]
    names: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.tools)

    def __getitem__(self, index: int | slice) -> Any:
        return self.tools[index]

    def validate(self) -> None:
        """Reject renaming of a previously accepted member."""
        if tuple(tool.name for tool in self.tools) != self.names:
            raise ValueError('Selected tool names changed during this request')


def freeze_task_tools(
    catalog: Sequence[Any], selected: Sequence[Any] | None,
) -> ToolGrant | None:
    """Validate canonical identity before any hook or provider execution."""
    if selected is None:
        return None
    if isinstance(selected, ToolGrant):
        selected.validate()
    if isinstance(selected, (str, bytes)) or not isinstance(selected, Sequence):
        raise ValueError('task_tools must be a sequence of canonical tool instances')
    available = effective_tools(catalog)
    chosen = tuple(selected)
    names = [getattr(tool, 'name', None) for tool in chosen]
    if any(not isinstance(name, str) for name in names) or len(names) != len(set(names)):
        raise ValueError('Tool selection contains invalid or duplicate names')
    chosen = tuple(tool for tool in chosen if tool.name not in RUNTIME_TOOLS)
    for tool in chosen:
        matches = [candidate for candidate in available if candidate.name == tool.name]
        if len(matches) != 1 or matches[0] is not tool:
            raise ValueError(f'Tool {tool.name!r} is not a canonical available principal tool')
    return ToolGrant(chosen, tuple(tool.name for tool in chosen))


def select_tool_names(catalog: Sequence[Any], names: Sequence[str]) -> tuple[Any, ...]:
    """Resolve exact names without aliases, capability changes or replacement tools."""
    if isinstance(names, (str, bytes)) or any(not isinstance(name, str) for name in names):
        raise ValueError('Tool names must be a sequence of exact names')
    if len(names) != len(set(names)):
        raise ValueError('Duplicate tool names')
    available = effective_tools(catalog)
    selected = []
    for name in names:
        if name in RUNTIME_TOOLS:
            continue
        matches = [tool for tool in available if tool.name == name]
        if len(matches) != 1:
            raise ValueError(f'Tool {name!r} is not an unambiguous available principal tool')
        selected.append(matches[0])
    return tuple(selected)
