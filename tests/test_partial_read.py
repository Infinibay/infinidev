"""Tests for ReadFileTool's partial-read behaviour (start_line/end_line).

Originally these tested PartialReadTool, a 6-line wrapper that just
delegated to ReadFileTool with offset/limit. The wrapper was removed
because read_file accepts start_line/end_line natively as aliases —
the wrapper added a tool name to the registry but no behaviour. The
tests were rewritten to exercise the same range-read code path
through ReadFileTool directly. The legacy tool name still routes
here through the alias map in ``engine.loop.tools._TOOL_ALIASES``.
"""

import json

import pytest

from infinidev.tools.file.read_file import ReadFileTool


class TestReadFilePartial:
    """Range reads via ReadFileTool's start_line/end_line parameters."""

    def test_read_line_range(self, bound_tool, workspace_dir):
        """Reads the specified line range."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            file_path=str(workspace_dir / "sample.txt"), start_line=2, end_line=4,
        )
        assert "line two" in result
        assert "line three" in result
        assert "line four" in result
        assert "line one" not in result
        assert "line five" not in result

    def test_read_single_line(self, bound_tool, workspace_dir):
        """Reads a single line."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            file_path=str(workspace_dir / "sample.txt"), start_line=3, end_line=3,
        )
        assert "line three" in result
        assert "line two" not in result
        assert "line four" not in result

    def test_read_first_line(self, bound_tool, workspace_dir):
        """Reads the first line only."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(
            file_path=str(workspace_dir / "sample.txt"), start_line=1, end_line=1,
        )
        assert "line one" in result
        assert "line two" not in result

    def test_file_not_found(self, bound_tool):
        """Returns error for missing file."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path="/nonexistent.txt", start_line=1, end_line=1)
        data = json.loads(result)
        assert "error" in data
