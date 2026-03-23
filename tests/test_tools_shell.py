"""Tests for ExecuteCommandTool."""

import json
from unittest.mock import MagicMock, patch

import pytest

from infinidev.config.settings import settings
from infinidev.tools.shell.execute_command import ExecuteCommandTool


class TestExecuteCommand:
    """Tests for shell command execution."""

    def test_execute_simple_command(self, bound_tool, auto_approve_permissions):
        """echo hello returns stdout."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="echo hello")
        data = json.loads(result)
        assert data["success"] is True
        assert "hello" in data["stdout"]

    def test_execute_empty_command(self, bound_tool, auto_approve_permissions):
        """Empty string returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="")
        data = json.loads(result)
        assert "error" in data
        assert "empty" in data["error"].lower()

    def test_execute_whitespace_only(self, bound_tool, auto_approve_permissions):
        """Whitespace-only command returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="   ")
        data = json.loads(result)
        assert "error" in data

    def test_execute_nonzero_exit_code(self, bound_tool, auto_approve_permissions):
        """exit 1 returns success=false with exit_code=1."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="exit 1")
        data = json.loads(result)
        assert data["success"] is False
        assert data["exit_code"] == 1

    def test_execute_timeout(self, bound_tool, auto_approve_permissions):
        """Command that exceeds timeout returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="sleep 10", timeout=1)
        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"].lower()

    def test_execute_stdout_truncation(self, bound_tool, auto_approve_permissions):
        """Output longer than 10K is truncated to last 10K chars."""
        tool = bound_tool(ExecuteCommandTool)
        # Generate >10K of output
        result = tool._run(command="python3 -c \"print('x' * 15000)\"")
        data = json.loads(result)
        assert len(data["stdout"]) <= 10001  # 10K + possible newline

    def test_execute_custom_cwd(self, bound_tool, auto_approve_permissions, workspace_dir):
        """cwd parameter is respected."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="pwd", cwd=str(workspace_dir))
        data = json.loads(result)
        assert str(workspace_dir) in data["stdout"]

    def test_execute_custom_env(self, bound_tool, auto_approve_permissions):
        """env parameter adds environment variables."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(
            command="echo $MY_TEST_VAR",
            env={"MY_TEST_VAR": "test_value_123"},
        )
        data = json.loads(result)
        assert "test_value_123" in data["stdout"]

    def test_permission_auto_approve(self, bound_tool):
        """auto_approve mode allows any command."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "auto_approve"
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo allowed")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_permission_allowed_list_allows(self, bound_tool):
        """Command in allowed list runs."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = ["echo", "ls"]
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo ok")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_allowed_list_blocks(self, bound_tool):
        """Command not in allowed list is denied."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = ["echo"]
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="rm -rf /")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_allowed_list_empty(self, bound_tool):
        """Empty allowed list denies everything."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = []
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo blocked")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_ask_approved(self, bound_tool):
        """When ask mode and permission granted, command runs."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "ask"
        try:
            tool = bound_tool(ExecuteCommandTool)
            with patch("infinidev.tools.permission.request_permission", return_value=True):
                result = tool._run(command="echo approved")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_permission_ask_denied(self, bound_tool):
        """When ask mode and permission denied, command blocked."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "ask"
        try:
            tool = bound_tool(ExecuteCommandTool)
            with patch("infinidev.tools.permission.request_permission", return_value=False):
                result = tool._run(command="echo denied")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_cwd_defaults_to_workspace(self, bound_tool, auto_approve_permissions):
        """When no cwd given, uses workspace_path."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="pwd")
        data = json.loads(result)
        ws = tool.workspace_path
        assert ws in data["stdout"]
