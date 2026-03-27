"""Tests for add_content_after_line and add_content_before_line tools."""

import json
import os

import pytest

from infinidev.tools.file.insert_lines import AddContentAfterLineTool, AddContentBeforeLineTool


class TestAddContentAfterLine:
    def test_insert_after_line(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentAfterLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=2, content="inserted\n")
        data = json.loads(result)
        assert data["lines_added"] == 1
        assert data["inserted_at"] == 3  # after line 2 = position 3

        with open(path) as f:
            lines = f.readlines()
        assert lines[2] == "inserted\n"
        assert lines[1] == "line two\n"  # original line 2 unchanged
        assert len(lines) == 6  # 5 + 1

    def test_insert_at_beginning(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentAfterLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=0, content="first\n")
        data = json.loads(result)
        assert data["inserted_at"] == 1

        with open(path) as f:
            first = f.readline()
        assert first == "first\n"

    def test_insert_at_end(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentAfterLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=5, content="last\n")
        data = json.loads(result)

        with open(path) as f:
            lines = f.readlines()
        assert lines[-1] == "last\n"
        assert len(lines) == 6

    def test_multiline_insert(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentAfterLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=1, content="a\nb\nc\n")
        data = json.loads(result)
        assert data["lines_added"] == 3

        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 8  # 5 + 3

    def test_file_not_found(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentAfterLineTool)
        result = tool._run(file_path=str(workspace_dir / "nope.txt"), line_number=1, content="x")
        data = json.loads(result)
        assert "error" in data


class TestAddContentBeforeLine:
    def test_insert_before_line(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentBeforeLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=3, content="before three\n")
        data = json.loads(result)
        assert data["lines_added"] == 1
        assert data["inserted_at"] == 3  # before line 3 = position 3

        with open(path) as f:
            lines = f.readlines()
        assert lines[2] == "before three\n"
        assert lines[3] == "line three\n"  # pushed down
        assert len(lines) == 6

    def test_insert_before_first_line(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentBeforeLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=1, content="header\n")
        data = json.loads(result)

        with open(path) as f:
            first = f.readline()
        assert first == "header\n"

    def test_invalid_line_number(self, bound_tool, workspace_dir):
        tool = bound_tool(AddContentBeforeLineTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=path, line_number=0, content="x")
        data = json.loads(result)
        assert "error" in data
