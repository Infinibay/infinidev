"""Tests for the read-file tool."""

import json
from unittest.mock import patch

from infinidev.config.settings import settings
from infinidev.tools.file.read_file import ReadFileTool


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
