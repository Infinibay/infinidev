"""Empty workspaces are ordinary tool results, not failed exploration."""

from __future__ import annotations

import json
import os

import pytest

from infinidev.tools.code_intel.project_structure_tool import ProjectStructureTool


@pytest.mark.parametrize("hidden_state", [False, True])
def test_empty_workspace_has_successful_structure_result(
    bound_tool, workspace_dir, sandbox_disabled, hidden_state,
):
    root = workspace_dir / "empty"
    root.mkdir()
    if hidden_state:
        (root / ".infinidev").mkdir()
    tool = bound_tool(ProjectStructureTool)

    result = tool._run(file_path=str(root))

    assert result.startswith(f"Project structure of {root}")
    assert "no visible files or directories" in result.lower()


def test_unreadable_root_is_still_an_error(
    bound_tool, workspace_dir, sandbox_disabled, monkeypatch,
):
    root = workspace_dir / "unreadable"
    root.mkdir()
    original = os.listdir

    def listdir(path):
        if os.fspath(path) == str(root):
            raise PermissionError("Access denied")
        return original(path)

    monkeypatch.setattr(os, "listdir", listdir)
    result = bound_tool(ProjectStructureTool)._run(file_path=str(root))

    assert "error" in json.loads(result)
