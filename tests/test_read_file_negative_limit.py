"""A negative line count must not become a Python slice from the file's end."""

from __future__ import annotations

import json
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from infinidev.tools.file.read_file_input import ReadFileInput
from infinidev.tools.file.read_file_tool import ReadFileTool
from tests.test_turn_runtime import reply, runtime, tc


@pytest.mark.parametrize("limit", [-1, -2])
def test_schema_rejects_negative_line_counts(limit):
    with pytest.raises(ValidationError):
        ReadFileInput.model_validate({"file_path": "synthetic.txt", "limit": limit})


def test_direct_negative_limit_returns_error_before_local_or_pod_read(bound_tool, monkeypatch):
    tool = bound_tool(ReadFileTool)
    monkeypatch.setattr(ReadFileTool, "_is_pod_mode",
                        Mock(side_effect=AssertionError("must reject before execution")))
    result = json.loads(tool._run(file_path="synthetic.txt", offset=1, limit=-2))
    assert "limit" in result["error"] and "non-negative" in result["error"]


def test_negative_limit_recovery_never_dispatches_or_exposes_source(runtime):
    engine, agent, script, seen = runtime
    script(reply("", tc("read_file", file_path="sample.txt", offset=1, limit=-2)),
           reply("", tc("finish_turn", message="Invalid range; no file read.")))
    result = engine.run_turn(agent, "read a bounded range", task_tools=[ReadFileTool()])
    assert result.status == "answered"
    assert engine._last_turn_receipt["dispatches"] == 0
    feedback = next(m["content"] for m in seen[1][0] if m["role"] == "tool")
    assert json.loads(feedback)["executed"] is False
    assert "greater_than_equal" in feedback
