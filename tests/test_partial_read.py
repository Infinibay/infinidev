"""Tests for PartialReadTool."""

import json

import pytest

from infinidev.tools.file.partial_read import PartialReadTool


class TestPartialRead:
    """Tests for PartialReadTool."""

    def test_read_line_range(self, bound_tool, workspace_dir):
        """Reads the specified line range."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"), start_line=2, end_line=4,
        )
        assert "line two" in result
        assert "line three" in result
        assert "line four" in result
        assert "line one" not in result
        assert "line five" not in result

    def test_read_single_line(self, bound_tool, workspace_dir):
        """Reads a single line."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"), start_line=3, end_line=3,
        )
        assert "line three" in result
        assert "line two" not in result
        assert "line four" not in result

    def test_read_first_line(self, bound_tool, workspace_dir):
        """Reads the first line only."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"), start_line=1, end_line=1,
        )
        assert "line one" in result
        assert "line two" not in result

    def test_invalid_start_line(self, bound_tool, workspace_dir):
        """Rejects start_line < 1."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"), start_line=0, end_line=3,
        )
        data = json.loads(result)
        assert "error" in data

    def test_end_before_start(self, bound_tool, workspace_dir):
        """Rejects end_line < start_line."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"), start_line=3, end_line=1,
        )
        data = json.loads(result)
        assert "error" in data

    def test_file_not_found(self, bound_tool):
        """Returns error for missing file."""
        tool = bound_tool(PartialReadTool)
        result = tool._run(path="/nonexistent.txt", start_line=1, end_line=1)
        data = json.loads(result)
        assert "error" in data
