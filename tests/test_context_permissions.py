"""Tests for tool context management and permissions."""

import os
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.tools.base.context import (
    ToolContext,
    bind_tools_to_agent,
    clear_agent_context,
    get_context,
    get_context_for_agent,
    set_context,
)
from infinidev.tools.base.permissions import (
    check_command_permission,
    check_file_permission,
)


# ── ToolContext ──────────────────────────────────────────────────────────────


class TestToolContext:
    """Tests for context management."""

    def test_set_and_get_context(self):
        """Round-trip set/get works."""
        ctx = set_context(project_id=42, agent_id="ctx-test", session_id="s1")
        try:
            assert ctx.project_id == 42
            assert ctx.agent_id == "ctx-test"
            assert ctx.session_id == "s1"
        finally:
            clear_agent_context("ctx-test")

    def test_get_context_for_agent(self):
        """Process-global storage keyed by agent_id."""
        set_context(agent_id="agent-a", project_id=1)
        set_context(agent_id="agent-b", project_id=2)
        try:
            ctx_a = get_context_for_agent("agent-a")
            ctx_b = get_context_for_agent("agent-b")
            assert ctx_a.project_id == 1
            assert ctx_b.project_id == 2
        finally:
            clear_agent_context("agent-a")
            clear_agent_context("agent-b")

    def test_clear_agent_context(self):
        """Clearing removes entry."""
        set_context(agent_id="to-clear", project_id=99)
        clear_agent_context("to-clear")
        ctx = get_context_for_agent("to-clear")
        assert ctx.project_id is None

    def test_bind_tools_to_agent(self):
        """Tools get _bound_agent_id attribute."""
        class FakeTool:
            name = "fake"

        tool = FakeTool()
        bind_tools_to_agent([tool], "bind-test")
        assert getattr(tool, "_bound_agent_id") == "bind-test"

    def test_nonexistent_agent_returns_empty_context(self):
        """Getting context for unknown agent returns empty ToolContext."""
        ctx = get_context_for_agent("nonexistent-agent-xyz")
        assert ctx.project_id is None
        assert ctx.agent_id is None


# ── Permissions ──────────────────────────────────────────────────────────────


class TestCommandPermission:
    """Tests for command permission checking."""

    def test_sandbox_disabled_allows_all(self):
        """When sandbox off, any command allowed."""
        orig = settings.SANDBOX_ENABLED
        settings.SANDBOX_ENABLED = False
        try:
            assert check_command_permission("rm -rf /") is True
        finally:
            settings.SANDBOX_ENABLED = orig

    def test_sandbox_enabled_with_allowed_commands(self):
        """When sandbox on and command in allowed list, returns True."""
        orig_sandbox = settings.SANDBOX_ENABLED
        settings.SANDBOX_ENABLED = True
        settings.ALLOWED_COMMANDS = ["echo", "ls"]
        try:
            assert check_command_permission("echo hello") is True
            assert check_command_permission("rm /tmp/x") is False
        finally:
            settings.SANDBOX_ENABLED = orig_sandbox
            settings.ALLOWED_COMMANDS = []


class TestFilePermission:
    """Tests for file permission checking."""

    def test_auto_approve_allows_all(self):
        """auto_approve mode allows any path."""
        orig = settings.FILE_OPERATIONS_PERMISSION
        settings.FILE_OPERATIONS_PERMISSION = "auto_approve"
        try:
            assert check_file_permission("write_file", "/etc/passwd") is None
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig

    def test_allowed_paths_match(self):
        """Path within allowed paths returns None."""
        orig_mode = settings.FILE_OPERATIONS_PERMISSION
        orig_paths = settings.ALLOWED_FILE_PATHS
        settings.FILE_OPERATIONS_PERMISSION = "allowed_paths"
        settings.ALLOWED_FILE_PATHS = ["/home/user/project"]
        try:
            result = check_file_permission("write_file", "/home/user/project/src/file.py")
            assert result is None
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig_mode
            settings.ALLOWED_FILE_PATHS = orig_paths

    def test_allowed_paths_no_match(self):
        """Path outside allowed paths returns error."""
        orig_mode = settings.FILE_OPERATIONS_PERMISSION
        orig_paths = settings.ALLOWED_FILE_PATHS
        settings.FILE_OPERATIONS_PERMISSION = "allowed_paths"
        settings.ALLOWED_FILE_PATHS = ["/home/user/project"]
        try:
            result = check_file_permission("write_file", "/etc/shadow")
            assert result is not None
            assert "denied" in result.lower()
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig_mode
            settings.ALLOWED_FILE_PATHS = orig_paths

    def test_allowed_paths_empty(self):
        """Empty allowed paths denies everything."""
        orig_mode = settings.FILE_OPERATIONS_PERMISSION
        orig_paths = settings.ALLOWED_FILE_PATHS
        settings.FILE_OPERATIONS_PERMISSION = "allowed_paths"
        settings.ALLOWED_FILE_PATHS = []
        try:
            result = check_file_permission("edit_file", "/any/path")
            assert result is not None
            assert "denied" in result.lower()
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig_mode
            settings.ALLOWED_FILE_PATHS = orig_paths

    def test_ask_mode_approved(self):
        """Ask mode with user approval returns None."""
        orig = settings.FILE_OPERATIONS_PERMISSION
        settings.FILE_OPERATIONS_PERMISSION = "ask"
        try:
            with patch("infinidev.tools.permission.request_permission", return_value=True):
                result = check_file_permission("write_file", "/some/path")
            assert result is None
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig

    def test_ask_mode_denied(self):
        """Ask mode with user denial returns error."""
        orig = settings.FILE_OPERATIONS_PERMISSION
        settings.FILE_OPERATIONS_PERMISSION = "ask"
        try:
            with patch("infinidev.tools.permission.request_permission", return_value=False):
                result = check_file_permission("write_file", "/some/path")
            assert result is not None
            assert "denied" in result.lower()
        finally:
            settings.FILE_OPERATIONS_PERMISSION = orig
