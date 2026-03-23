"""Tests for file tools: ReadFile, WriteFile, EditFile."""

import json
import os
import stat
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.file.read_file import ReadFileTool
from infinidev.tools.file.write_file import WriteFileTool
from infinidev.tools.file.edit_file import EditFileTool


# ── ReadFile ─────────────────────────────────────────────────────────────────


class TestReadFile:
    """Tests for ReadFileTool."""

    def test_read_existing_file(self, bound_tool, workspace_dir):
        """Reads content with line numbers."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"))
        assert "line one" in result
        assert "line five" in result
        # Check line numbering format
        assert "     1\t" in result

    def test_read_nonexistent_file(self, bound_tool):
        """Returns JSON error for missing file."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path="/nonexistent/path/file.txt")
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_read_directory_not_file(self, bound_tool, workspace_dir):
        """Returns error when path is a directory."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir))
        data = json.loads(result)
        assert "error" in data
        assert "Not a file" in data["error"]

    def test_read_with_offset_and_limit(self, bound_tool, workspace_dir):
        """Reads specific line range."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"), offset=2, limit=2)
        assert "line two" in result
        assert "line three" in result
        assert "line one" not in result
        assert "line four" not in result

    def test_read_offset_only(self, bound_tool, workspace_dir):
        """Offset without limit reads to end."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"), offset=4)
        assert "line four" in result
        assert "line five" in result
        assert "line one" not in result

    def test_read_limit_only(self, bound_tool, workspace_dir):
        """Limit without offset reads from beginning."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"), limit=2)
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
            result = tool._run(path=str(big))
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_read_line_numbering_format(self, bound_tool, workspace_dir):
        """Verifies {N:>6}\\t{content} format."""
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(workspace_dir / "sample.txt"))
        lines = result.split("\n")
        first_line = lines[0]
        # Format: "     1\tline one"
        assert first_line.startswith("     1\t")

    def test_read_binary_file_rejected(self, bound_tool, workspace_dir):
        """Binary files (with null bytes) are rejected."""
        binary = workspace_dir / "image.bin"
        binary.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00")
        tool = bound_tool(ReadFileTool)
        result = tool._run(path=str(binary))
        data = json.loads(result)
        assert "error" in data
        assert "binary" in data["error"].lower()

    def test_read_sandbox_blocked(self, bound_tool, sandbox_enabled):
        """With sandbox enabled, path outside allowed dirs is blocked."""
        tool = bound_tool(ReadFileTool)
        # Disable pod mode so sandbox validation runs locally
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(path="/etc/passwd")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()


# ── WriteFile ────────────────────────────────────────────────────────────────


class TestWriteFile:
    """Tests for WriteFileTool."""

    def test_write_new_file(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Creates a new file and returns action=created."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "new_file.txt")
        result = tool._run(path=path, content="hello world")
        data = json.loads(result)
        assert data["action"] == "created"
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "hello world"

    def test_write_overwrites_existing(self, bound_tool, workspace_dir, auto_approve_permissions):
        """mode='w' replaces content."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(path=path, content="replaced")
        data = json.loads(result)
        assert data["action"] == "modified"
        with open(path) as f:
            assert f.read() == "replaced"

    def test_write_append_mode(self, bound_tool, workspace_dir, auto_approve_permissions):
        """mode='a' appends content."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "sample.txt")
        tool._run(path=path, content="appended", mode="a")
        with open(path) as f:
            content = f.read()
        assert content.endswith("appended")
        assert "line one" in content

    def test_write_creates_parent_directories(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Nested path creates all parent dirs."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "a" / "b" / "c.txt")
        tool._run(path=path, content="deep")
        assert os.path.exists(path)
        with open(path) as f:
            assert f.read() == "deep"

    def test_write_preserves_permissions(self, bound_tool, workspace_dir, auto_approve_permissions):
        """File permissions are preserved after overwrite."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "exec.sh")
        with open(path, "w") as f:
            f.write("#!/bin/bash\necho hi")
        os.chmod(path, 0o755)
        original_mode = os.stat(path).st_mode

        tool._run(path=path, content="#!/bin/bash\necho bye")
        new_mode = os.stat(path).st_mode
        assert stat.S_IMODE(new_mode) == stat.S_IMODE(original_mode)

    def test_write_size_limit_enforced(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Content exceeding MAX_FILE_SIZE_BYTES returns error."""
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 50
        try:
            tool = bound_tool(WriteFileTool)
            result = tool._run(path=str(workspace_dir / "big.txt"), content="x" * 100)
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_write_records_artifact_change(self, bound_tool, workspace_dir, auto_approve_permissions):
        """After write, artifact_changes table has a row."""
        tool = bound_tool(WriteFileTool)
        path = str(workspace_dir / "audited.txt")
        tool._run(path=path, content="audited content")

        rows = execute_with_retry(
            lambda c: c.execute("SELECT * FROM artifact_changes WHERE file_path = ?", (path,)).fetchall()
        )
        assert len(rows) >= 1
        row = dict(rows[0])
        assert row["action"] == "created"

    def test_write_sandbox_blocked(self, bound_tool, sandbox_enabled, auto_approve_permissions):
        """Sandbox check rejects out-of-bounds path."""
        tool = bound_tool(WriteFileTool)
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(path="/etc/shadow", content="nope")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()

    def test_write_append_size_limit(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Existing + appended content exceeding limit returns error."""
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 100
        try:
            tool = bound_tool(WriteFileTool)
            path = str(workspace_dir / "sample.txt")  # already ~50 bytes
            result = tool._run(path=path, content="x" * 80, mode="a")
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original


# ── EditFile ─────────────────────────────────────────────────────────────────


class TestEditFile:
    """Tests for EditFileTool."""

    def test_edit_single_replacement(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Replaces one occurrence."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(path=path, old_string="line two", new_string="LINE TWO")
        data = json.loads(result)
        assert data["action"] == "modified"
        assert data["replacements"] == 1
        with open(path) as f:
            assert "LINE TWO" in f.read()

    def test_edit_identical_strings_rejected(self, bound_tool, workspace_dir, auto_approve_permissions):
        """old_string == new_string returns error."""
        tool = bound_tool(EditFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            old_string="line one",
            new_string="line one",
        )
        data = json.loads(result)
        assert "error" in data
        assert "identical" in data["error"].lower()

    def test_edit_nonexistent_file(self, bound_tool, auto_approve_permissions):
        """Returns error for missing file."""
        tool = bound_tool(EditFileTool)
        result = tool._run(
            path="/nonexistent/file.py",
            old_string="x",
            new_string="y",
        )
        data = json.loads(result)
        assert "error" in data

    def test_edit_old_string_not_found(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Returns error when old_string not in file."""
        tool = bound_tool(EditFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            old_string="this does not exist",
            new_string="replacement",
        )
        data = json.loads(result)
        assert "error" in data
        assert "not found" in data["error"].lower()

    def test_edit_close_match_suggestion(self, bound_tool, workspace_dir, auto_approve_permissions):
        """When not found, suggests close match."""
        tool = bound_tool(EditFileTool)
        result = tool._run(
            path=str(workspace_dir / "sample.txt"),
            old_string="lien one",  # Typo: "lien" vs "line"
            new_string="replacement",
        )
        data = json.loads(result)
        assert "error" in data
        assert "closest match" in data["error"].lower() or "not found" in data["error"].lower()

    def test_edit_multiple_occurrences_without_replace_all(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Multiple occurrences without replace_all returns error."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "dupes.txt")
        with open(path, "w") as f:
            f.write("hello world\nhello again\nhello third\n")
        result = tool._run(path=path, old_string="hello", new_string="bye")
        data = json.loads(result)
        assert "error" in data
        assert "3 times" in data["error"]

    def test_edit_replace_all(self, bound_tool, workspace_dir, auto_approve_permissions):
        """replace_all=True replaces all occurrences."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "dupes.txt")
        with open(path, "w") as f:
            f.write("hello world\nhello again\n")
        result = tool._run(path=path, old_string="hello", new_string="bye", replace_all=True)
        data = json.loads(result)
        assert data["replacements"] == 2
        with open(path) as f:
            content = f.read()
        assert "hello" not in content
        assert content.count("bye") == 2

    def test_edit_resulting_file_too_large(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Replacement that exceeds size limit returns error."""
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 100
        try:
            tool = bound_tool(EditFileTool)
            path = str(workspace_dir / "sample.txt")
            result = tool._run(
                path=path,
                old_string="line one",
                new_string="x" * 200,
            )
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_edit_preserves_permissions(self, bound_tool, workspace_dir, auto_approve_permissions):
        """File permissions retained after edit."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "sample.txt")
        os.chmod(path, 0o755)
        original_mode = os.stat(path).st_mode

        tool._run(path=path, old_string="line one", new_string="LINE ONE")
        new_mode = os.stat(path).st_mode
        assert stat.S_IMODE(new_mode) == stat.S_IMODE(original_mode)

    def test_edit_records_artifact_change(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Audit record created in artifact_changes."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "sample.txt")
        tool._run(path=path, old_string="line one", new_string="LINE ONE")

        rows = execute_with_retry(
            lambda c: c.execute("SELECT * FROM artifact_changes WHERE file_path = ?", (path,)).fetchall()
        )
        assert len(rows) >= 1

    def test_edit_unicode_content(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Handles UTF-8 correctly."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "unicode.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write("café résumé naïve\n")
        result = tool._run(path=path, old_string="café", new_string="coffee")
        data = json.loads(result)
        assert data["replacements"] == 1
        with open(path, encoding="utf-8") as f:
            assert "coffee" in f.read()

    def test_edit_multiline_replacement(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Multi-line old_string and new_string work."""
        tool = bound_tool(EditFileTool)
        path = str(workspace_dir / "sample.txt")
        result = tool._run(
            path=path,
            old_string="line two\nline three",
            new_string="LINE 2\nLINE 3",
        )
        data = json.loads(result)
        assert data["replacements"] == 1
        with open(path) as f:
            content = f.read()
        assert "LINE 2\nLINE 3" in content

    def test_edit_sandbox_blocked(self, bound_tool, sandbox_enabled, auto_approve_permissions):
        """Sandbox check blocks out-of-bounds edit."""
        tool = bound_tool(EditFileTool)
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(path="/etc/hosts", old_string="x", new_string="y")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()

    def test_edit_not_a_file(self, bound_tool, workspace_dir, auto_approve_permissions):
        """Directory path returns error."""
        tool = bound_tool(EditFileTool)
        result = tool._run(
            path=str(workspace_dir),
            old_string="x",
            new_string="y",
        )
        data = json.loads(result)
        assert "error" in data
