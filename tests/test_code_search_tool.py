"""Regression tests for bounded code-search subprocess execution."""

from __future__ import annotations

import json

from infinidev.tools.file.code_search_tool import CodeSearchTool


def test_max_depth_search_uses_safe_shell_quoting(bound_tool, workspace_dir) -> None:
    marker = workspace_dir / "shell-injection-marker"
    (workspace_dir / "source.py").write_text("needle = True\n")
    tool = bound_tool(CodeSearchTool)

    result = json.loads(
        tool._run(
            pattern=f"needle; touch {marker}",
            file_path=str(workspace_dir),
            max_depth=2,
        )
    )

    assert result["match_count"] == 0
    assert not marker.exists()


def test_max_depth_search_returns_matches(bound_tool, workspace_dir) -> None:
    nested = workspace_dir / "nested"
    nested.mkdir()
    (nested / "source.py").write_text("needle = True\n")
    tool = bound_tool(CodeSearchTool)

    result = json.loads(
        tool._run(
            pattern="needle",
            file_path=str(workspace_dir),
            file_extensions=["py"],
            max_depth=2,
        )
    )

    assert result["match_count"] == 1
    assert result["matches"][0]["file"].endswith("nested/source.py")
