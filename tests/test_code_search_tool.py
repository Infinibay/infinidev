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


def test_search_accepts_a_single_file_path(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("before\nneedle = True\nafter\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        context_lines=1,
    ))

    assert result["match_count"] == 1
    assert result["matches"][0]["line"] == 2
    assert [row["line"] for row in result["matches"][0]["context"]] == [1, 2, 3]


def test_single_file_exact_limit_is_not_truncated(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("needle\nneedle\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        max_results=2,
    ))

    assert result["match_count"] == 2
    assert result["truncated"] is False


def test_single_file_more_than_limit_is_truncated(bound_tool, workspace_dir) -> None:
    source = workspace_dir / "source.py"
    source.write_text("needle\nneedle\nneedle\n")

    result = json.loads(bound_tool(CodeSearchTool)._run(
        pattern="needle",
        file_path=str(source),
        max_results=2,
    ))

    assert result["match_count"] == 2
    assert result["truncated"] is True
