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


class _SearchLikeTool(InfinibayBaseTool):
    name: str = "code_search"
    description: str = "Record normalized search arguments."

    def _run(self, pattern: str, context_lines: int = 0) -> str:
        return json.dumps({"pattern": pattern, "context_lines": context_lines})


class _EditLikeTool(InfinibayBaseTool):
    name: str = "edit_file"
    description: str = "Record normalized edit arguments."

    def _run(self, file_path: str, old_string: str, new_string: str) -> str:
        return json.dumps({
            "file_path": file_path,
            "old_string": old_string,
            "new_string": new_string,
        })


class _ReadPathTool:
    name = "read_file"

    @staticmethod
    def _resolve_path(path: str) -> str:
        return path

    def _run(self, file_path: str, offset: int = 1) -> str:
        return json.dumps({"read": file_path, "offset": offset})


class _ListPathTool:
    name = "list_directory"

    def _run(self, file_path: str = ".") -> str:
        return json.dumps({"listed": file_path})


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


def test_dispatch_normalizes_minimax_code_search_context() -> None:
    tool = _SearchLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {"pattern": "Widget", "context": 20},
    )

    assert json.loads(result) == {"pattern": "Widget", "context_lines": 5}


def test_dispatch_expands_minimax_structured_edit_replacement() -> None:
    tool = _EditLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {
            "file_path": "src/widget.py",
            "replace": {"old": "before", "new": "after"},
        },
    )

    assert json.loads(result) == {
        "file_path": "src/widget.py",
        "old_string": "before",
        "new_string": "after",
    }


def test_dispatch_routes_read_file_on_directory_to_list_directory(tmp_path) -> None:
    read = _ReadPathTool()
    listing = _ListPathTool()

    result = execute_tool_call(
        {read.name: read, listing.name: listing},
        "read_file",
        {"file_path": str(tmp_path), "offset": 50},
    )

    assert json.loads(result) == {"listed": str(tmp_path)}
