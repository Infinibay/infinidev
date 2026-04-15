"""Tests for file tools: ReadFile, WriteFile."""

import json
import os
import stat
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file.read_file import ReadFileTool
from infinidev.tools.file.write_file import WriteFileTool


# ── ReadFile ─────────────────────────────────────────────────────────────────


class TestReadFile:
    """Tests for ReadFileTool."""

    def test_read_existing_file(self, bound_tool, workspace_dir):
        """Reads content with line numbers."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir / "sample.txt"))
        assert "line one" in result
        assert "line five" in result
        # Check line numbering format
        assert "     1\t" in result

    def test_read_nonexistent_file(self, bound_tool):
        """Returns JSON error for missing file."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path="/nonexistent/file_path/file.txt")
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_read_directory_not_file(self, bound_tool, workspace_dir):
        """Returns error when file_path is a directory."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir))
        data = json.loads(result)
        assert "error" in data
        assert "Not a file" in data["error"]

    def test_read_with_offset_and_limit(self, bound_tool, workspace_dir):
        """Reads specific line range."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir / "sample.txt"), offset=2, limit=2)
        assert "line two" in result
        assert "line three" in result
        assert "line one" not in result
        assert "line four" not in result

    def test_read_offset_only(self, bound_tool, workspace_dir):
        """Offset without limit reads to end."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir / "sample.txt"), offset=4)
        assert "line four" in result
        assert "line five" in result
        assert "line one" not in result

    def test_read_limit_only(self, bound_tool, workspace_dir):
        """Limit without offset reads from beginning."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir / "sample.txt"), limit=2)
        assert "line one" in result
        assert "line two" in result
        assert "line three" not in result

    def test_read_oversized_file(self, bound_tool, workspace_dir):
        """File larger than MAX_FILE_SIZE_BYTES returns error."""
        big = workspace_dir / "big.txt"
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 50  # Very small limit for testing
        try:
            big.write_text("x" * 100)
            tool = bound_tool(ReadFileTool)
            result = tool._run(file_path=str(big))
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_read_line_numbering_format(self, bound_tool, workspace_dir):
        """Verifies {N:>6}\\t{content} format."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(workspace_dir / "sample.txt"))
        lines = result.split("\n")
        first_line = lines[0]
        # Format: "     1\tline one"
        assert first_line.startswith("     1\t")

    def test_read_binary_file_rejected(self, bound_tool, workspace_dir):
        """Binary files (with null bytes) are rejected."""
        binary = workspace_dir / "image.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
        tool = bound_tool(ReadFileTool)
        result = tool._run(file_path=str(binary))
        data = json.loads(result)
        assert "error" in data
        assert "binary" in data["error"].lower()

    def test_read_sandbox_blocked(self, bound_tool, sandbox_enabled):
        """With sandbox enabled, file_path outside allowed dirs is blocked."""
        tool = bound_tool(ReadFileTool)
        # Disable pod mode so sandbox validation runs locally
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(file_path="/etc/passwd")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()


# ── WriteFile ────────────────────────────────────────────────────────────────


class TestWriteFile:
    """Tests for WriteFileTool."""

    def test_write_new_file(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Creates a new file and returns action=created."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "new_file.txt")
        result = tool._run(file_path=file_path, content="hello world")
        data = json.loads(result)
        assert data["action"] == "created"
        assert os.path.exists(file_path)
        with open(file_path) as f:
            assert f.read() == "hello world"

    def test_write_overwrites_existing(self, bound_tool, workspace_dir, auto_approve_permissions):
        """mode='w' replaces content."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "sample.txt")
        result = tool._run(file_path=file_path, content="replaced")
        data = json.loads(result)
        assert data["action"] == "modified"
        with open(file_path) as f:
            assert f.read() == "replaced"

    def test_write_append_mode(self, bound_tool, workspace_dir, auto_approve_permissions):
        """mode='a' appends content."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "sample.txt")
        tool._run(file_path=file_path, content="appended", mode="a")
        with open(file_path) as f:
            content = f.read()
        assert content.endswith("appended")
        assert "line one" in content

    def test_write_creates_parent_directories(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Nested file_path creates all parent dirs."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "a" / "b" / "c.txt")
        tool._run(file_path=file_path, content="deep")
        assert os.path.exists(file_path)
        with open(file_path) as f:
            assert f.read() == "deep"

    def test_write_preserves_permissions(self, bound_tool, workspace_dir, auto_approve_permissions):
        """File permissions are preserved after overwrite."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "exec.sh")
        with open(file_path, "w") as f:
            f.write("#!/bin/bash\necho hi")
        os.chmod(file_path, 0o755)
        original_mode = os.stat(file_path).st_mode

        tool._run(file_path=file_path, content="#!/bin/bash\necho bye")
        new_mode = os.stat(file_path).st_mode
        assert stat.S_IMODE(new_mode) == stat.S_IMODE(original_mode)

    def test_write_size_limit_enforced(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Content exceeding MAX_FILE_SIZE_BYTES returns error."""
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 50
        try:
            tool = bound_tool(WriteFileTool)
            result = tool._run(file_path=str(workspace_dir / "big.txt"), content="x" * 100)
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_write_records_artifact_change(self, bound_tool, workspace_dir, auto_approve_permissions):
        """After write, artifact_changes table has a row."""
        tool = bound_tool(WriteFileTool)
        file_path = str(workspace_dir / "audited.txt")
        tool._run(file_path=file_path, content="audited content")

        rows = execute_with_retry(
            lambda c: c.execute("SELECT * FROM artifact_changes WHERE file_path = ?", (file_path,)).fetchall()
        )
        assert len(rows) >= 1
        row = dict(rows[0])
        assert row["action"] == "created"

    def test_write_sandbox_blocked(self, bound_tool, sandbox_enabled, auto_approve_permissions):
        """Sandbox check rejects out-of-bounds path."""
        tool = bound_tool(WriteFileTool)
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(file_path="/etc/shadow", content="nope")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()

    def test_write_append_size_limit(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Existing + appended content exceeding limit returns error."""
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 100
        try:
            tool = bound_tool(WriteFileTool)
            file_path = str(workspace_dir / "sample.txt")  # already ~50 bytes
            result = tool._run(file_path=file_path, content="x" * 80, mode="a")
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

