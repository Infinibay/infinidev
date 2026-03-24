"""Tests for MultiEditFileTool."""

import json
import os

import pytest

from infinidev.tools.file.multi_edit_file import MultiEditFileTool


class TestMultiEdit:
    """Tests for MultiEditFileTool."""

    def test_basic_multi_edit(self, bound_tool, workspace_dir):
        """Apply two edits to the same file."""
        tool = bound_tool(MultiEditFileTool)
        fpath = str(workspace_dir / "sample.txt")
        result = tool._run(
            path=fpath,
            edits=[
                {"old_string": "line one", "new_string": "LINE ONE"},
                {"old_string": "line five", "new_string": "LINE FIVE"},
            ],
        )
        data = json.loads(result)
        assert data["edits_applied"] == 2

        content = open(fpath).read()
        assert "LINE ONE" in content
        assert "LINE FIVE" in content
        assert "line two" in content  # unchanged

    def test_atomicity_all_fail_on_missing(self, bound_tool, workspace_dir):
        """If one old_string doesn't exist, no edits apply."""
        tool = bound_tool(MultiEditFileTool)
        fpath = str(workspace_dir / "sample.txt")
        original = open(fpath).read()

        result = tool._run(
            path=fpath,
            edits=[
                {"old_string": "line one", "new_string": "CHANGED"},
                {"old_string": "NONEXISTENT", "new_string": "WHATEVER"},
            ],
        )
        data = json.loads(result)
        assert "error" in data

        # File should be unchanged
        assert open(fpath).read() == original

    def test_atomicity_all_fail_on_duplicate(self, bound_tool, workspace_dir):
        """If old_string appears more than once, edits fail."""
        tool = bound_tool(MultiEditFileTool)
        fpath = str(workspace_dir / "dupes.txt")
        with open(fpath, "w") as f:
            f.write("hello world\nhello world\n")

        result = tool._run(
            path=fpath,
            edits=[{"old_string": "hello world", "new_string": "goodbye"}],
        )
        data = json.loads(result)
        assert "error" in data
        assert "2 times" in data["error"]

    def test_overlapping_edits_rejected(self, bound_tool, workspace_dir):
        """Overlapping edits are detected and rejected."""
        tool = bound_tool(MultiEditFileTool)
        fpath = str(workspace_dir / "overlap.txt")
        with open(fpath, "w") as f:
            f.write("abcdefgh\n")

        result = tool._run(
            path=fpath,
            edits=[
                {"old_string": "abcdef", "new_string": "ABCDEF"},
                {"old_string": "defgh", "new_string": "DEFGH"},
            ],
        )
        data = json.loads(result)
        assert "error" in data
        assert "overlap" in data["error"].lower()

    def test_empty_edits(self, bound_tool, workspace_dir):
        """Empty edits list returns error."""
        tool = bound_tool(MultiEditFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            edits=[],
        )
        data = json.loads(result)
        assert "error" in data

    def test_identical_old_new_rejected(self, bound_tool, workspace_dir):
        """old_string == new_string is rejected."""
        tool = bound_tool(MultiEditFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            edits=[{"old_string": "line one", "new_string": "line one"}],
        )
        data = json.loads(result)
        assert "error" in data
        assert "identical" in data["error"].lower()

    def test_nonexistent_file(self, bound_tool):
        """Returns error for nonexistent file."""
        tool = bound_tool(MultiEditFileTool)
        result = tool._run(
            path="/nonexistent/file.txt",
            edits=[{"old_string": "a", "new_string": "b"}],
        )
        data = json.loads(result)
        assert "error" in data

    def test_edits_dont_interfere(self, bound_tool, workspace_dir):
        """Edit B's old_string shouldn't be affected by edit A's new_string."""
        tool = bound_tool(MultiEditFileTool)
        fpath = str(workspace_dir / "nointerfer.txt")
        with open(fpath, "w") as f:
            f.write("aaa\nbbb\nccc\n")

        result = tool._run(
            path=fpath,
            edits=[
                {"old_string": "aaa", "new_string": "bbb"},  # Now file has "bbb\nbbb\n..."
                {"old_string": "ccc", "new_string": "ddd"},
            ],
        )
        data = json.loads(result)
        assert data["edits_applied"] == 2

        content = open(fpath).read()
        assert content == "bbb\nbbb\nddd\n"
