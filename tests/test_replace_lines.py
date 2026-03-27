"""Tests for ReplaceLinesTool."""

import json
import os
from unittest.mock import patch

import pytest

from infinidev.tools.file.replace_lines import ReplaceLinesTool


class TestReplaceLines:
    """Tests for ReplaceLinesTool."""

    def test_basic_replacement(self, bound_tool, workspace_dir):
        """Replaces a range of lines with new content."""
        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        # sample.txt has: line one, line two, line three, line four, line five
        result = tool._run(
            file_path=path,
            content="replaced two\nreplaced three\n",
            start_line=2,
            end_line=3,
        )
        data = json.loads(result)
        assert "error" not in data
        assert data["lines_removed"] == 2
        assert data["lines_added"] == 2

        with open(path) as f:
            lines = f.readlines()
        assert lines[0] == "line one\n"
        assert lines[1] == "replaced two\n"
        assert lines[2] == "replaced three\n"
        assert lines[3] == "line four\n"

    def test_delete_lines(self, bound_tool, workspace_dir):
        """Empty content deletes the line range."""
        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(
            file_path=path, content="", start_line=2, end_line=3,
        )
        data = json.loads(result)
        assert "error" not in data
        assert data["lines_removed"] == 2
        assert data["lines_added"] == 0

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 3  # 5 - 2

    def test_insert_lines(self, bound_tool, workspace_dir):
        """start_line == end_line + 1 inserts without removing."""
        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(
            file_path=path,
            content="inserted line\n",
            start_line=3,
            end_line=2,  # end < start = insert before line 3
        )
        data = json.loads(result)
        assert "error" not in data
        assert data["lines_removed"] == 0
        assert data["lines_added"] == 1

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 6  # 5 + 1
        assert lines[2] == "inserted line\n"
        assert lines[3] == "line three\n"

    def test_replace_single_line(self, bound_tool, workspace_dir):
        """Replaces a single line."""
        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(
            file_path=path, content="new line one\n", start_line=1, end_line=1,
        )
        data = json.loads(result)
        assert "error" not in data

        with open(path) as f:
            first_line = f.readline()
        assert first_line == "new line one\n"

    def test_invalid_range(self, bound_tool, workspace_dir):
        """Rejects invalid line ranges."""
        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")

        # start_line < 1
        result = tool._run(file_path=path, content="x", start_line=0, end_line=1)
        data = json.loads(result)
        assert "error" in data

        # start_line beyond file
        result = tool._run(file_path=path, content="x", start_line=100, end_line=100)
        data = json.loads(result)
        assert "error" in data

    def test_file_not_found(self, bound_tool, workspace_dir):
        """Returns error for nonexistent file."""
        tool = bound_tool(ReplaceLinesTool)
        result = tool._run(
            file_path=str(workspace_dir / "nope.txt"),
            content="x", start_line=1, end_line=1,
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_audit_trail(self, bound_tool, workspace_dir, temp_db):
        """Records modification in artifact_changes."""
        from infinidev.tools.base.db import execute_with_retry

        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        tool._run(file_path=path, content="new\n", start_line=1, end_line=1)

        def _check(conn):
            row = conn.execute(
                "SELECT action FROM artifact_changes WHERE file_path = ?",
                (path,),
            ).fetchone()
            return row

        row = execute_with_retry(_check)
        assert row is not None
        assert row[0] == "modified"

    def test_preserves_permissions(self, bound_tool, workspace_dir):
        """Preserves original file permissions."""
        import stat

        tool = bound_tool(ReplaceLinesTool)
        path = str(workspace_dir / "sample.txt")
        os.chmod(path, 0o755)
        tool._run(file_path=path, content="new\n", start_line=1, end_line=1)
        mode = os.stat(path).st_mode
        assert stat.S_IMODE(mode) == 0o755

    def test_sandbox_blocking(self, bound_tool, workspace_dir, sandbox_enabled):
        """Sandbox blocks edits outside allowed dirs."""
        tool = bound_tool(ReplaceLinesTool)
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(
                file_path="/tmp/outside.txt", content="x", start_line=1, end_line=1,
            )
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()
