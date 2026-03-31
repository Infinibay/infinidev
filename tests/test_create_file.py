"""Tests for CreateFileTool."""

import json
import os
from unittest.mock import patch

import pytest

from infinidev.tools.file.create_file import CreateFileTool


class TestCreateFile:
    """Tests for CreateFileTool."""

    def test_create_new_file(self, bound_tool, workspace_dir):
        """Creates a new file successfully."""
        tool = bound_tool(CreateFileTool)
        file_path = str(workspace_dir / "new_file.py")
        result = tool._run(file_path=file_path, content="print('hello')\n")
        data = json.loads(result)
        assert data["action"] == "created"
        assert os.path.exists(file_path)
        with open(file_path) as f:
            assert f.read() == "print('hello')\n"

    def test_fail_if_exists(self, bound_tool, workspace_dir):
        """Fails when file already exists."""
        tool = bound_tool(CreateFileTool)
        file_path = str(workspace_dir / "sample.txt")  # already exists from fixture
        result = tool._run(file_path=file_path, content="overwrite")
        data = json.loads(result)
        assert "error" in data
        assert "already exists" in data["error"]

    def test_creates_parent_directories(self, bound_tool, workspace_dir):
        """Creates parent directories as needed."""
        tool = bound_tool(CreateFileTool)
        file_path = str(workspace_dir / "deep" / "nested" / "dir" / "file.txt")
        result = tool._run(file_path=file_path, content="nested content\n")
        data = json.loads(result)
        assert data["action"] == "created"
        assert os.path.exists(file_path)

    def test_audit_trail(self, bound_tool, workspace_dir, temp_db):
        """Records creation in artifact_changes."""
        from infinidev.tools.base.db import execute_with_retry

        tool = bound_tool(CreateFileTool)
        file_path = str(workspace_dir / "audited.txt")
        tool._run(file_path=file_path, content="audit me\n")

        def _check(conn):
            row = conn.execute(
                "SELECT action, file_path FROM artifact_changes WHERE file_path = ?",
                (file_path,),
            ).fetchone()
            return row

        row = execute_with_retry(_check)
        assert row is not None
        assert row[0] == "created"

    def test_content_size_limit(self, bound_tool, workspace_dir):
        """Rejects content exceeding MAX_FILE_SIZE_BYTES."""
        from infinidev.config.settings import settings

        tool = bound_tool(CreateFileTool)
        original = settings.MAX_FILE_SIZE_BYTES
        settings.MAX_FILE_SIZE_BYTES = 100
        try:
            file_path = str(workspace_dir / "big.txt")
            result = tool._run(file_path=file_path, content="x" * 200)
            data = json.loads(result)
            assert "error" in data
            assert "too large" in data["error"].lower()
        finally:
            settings.MAX_FILE_SIZE_BYTES = original

    def test_sandbox_blocking(self, bound_tool, workspace_dir, sandbox_enabled):
        """Sandbox blocks writes outside allowed dirs."""
        tool = bound_tool(CreateFileTool)
        with patch.object(tool, "_is_pod_mode", return_value=False):
            result = tool._run(file_path="/tmp/outside_sandbox.txt", content="blocked")
        data = json.loads(result)
        assert "error" in data
        assert "denied" in data["error"].lower()
