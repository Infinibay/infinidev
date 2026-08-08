"""Runtime validation at the final tool-dispatch boundary."""

from __future__ import annotations

import json

from pydantic import BaseModel, Field

from infinidev.engine.tool_dispatch import execute_tool_call
from infinidev.tools.base.base_tool import InfinibayBaseTool


class _ConstrainedInput(BaseModel):
    value: str = Field(min_length=5)


class _ConstrainedTool(InfinibayBaseTool):
    name: str = "constrained"
    description: str = "Test a constrained argument."
    args_schema: type[BaseModel] = _ConstrainedInput

    def _run(self, value: str) -> str:
        return value


class _RecallLikeTool(InfinibayBaseTool):
    name: str = "recall_context"
    description: str = "Test recall query aliases."

    def _run(self, query: str) -> str:
        return query


class _ShellLikeTool(InfinibayBaseTool):
    name: str = "execute_command"
    description: str = "Record a shell command without running it."

    def _run(self, command: str, cwd: str | None = None) -> str:
        return json.dumps({"command": command, "cwd": cwd})


def test_dispatch_enforces_pydantic_constraints() -> None:
    tool = _ConstrainedTool()

    result = json.loads(
        execute_tool_call({tool.name: tool}, tool.name, {"value": "no"})
    )

    assert "validation failed" in result["error"]
    assert "at least 5 characters" in result["error"]


def test_dispatch_runs_after_successful_validation() -> None:
    tool = _ConstrainedTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"value": "valid value"}
    )

    assert result == "valid value"


def test_dispatch_maps_recall_context_to_query() -> None:
    tool = _RecallLikeTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"context": "package metadata"}
    )

    assert result == "package metadata"


def test_dispatch_maps_minimax_shell_aliases_to_execute_command() -> None:
    """Live M3 naming misses must execute instead of becoming false blockers."""
    tool = _ShellLikeTool()

    for alias in ("shell_command", "shell_exec"):
        result = execute_tool_call(
            {tool.name: tool},
            alias,
            {"command": "pwd", "cwd": "/tmp/project"},
        )

        assert json.loads(result) == {"command": "pwd", "cwd": "/tmp/project"}
