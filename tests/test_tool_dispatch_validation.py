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
