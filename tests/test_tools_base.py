"""Tests for InfinibayBaseTool base class."""

import json
import os
from unittest.mock import patch

import pytest

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.context import (
    bind_tools_to_agent,
    clear_agent_context,
    get_context_for_agent,
    set_context,
)


# ── Concrete tool for testing ────────────────────────────────────────────────


class _DummyTool(InfinibayBaseTool):
    """Minimal concrete tool for testing the base class."""

    name: str = "dummy_tool"
    description: str = "A dummy tool for testing"

    def _run(self, message: str = "hello") -> str:
        return self._success({"message": message})


class _DummyToolVarKw(InfinibayBaseTool):
    """Concrete tool whose _run accepts **kwargs."""

    name: str = "dummy_varkw"
    description: str = "A dummy tool with **kwargs"

    def _run(self, message: str = "hello", **kwargs) -> str:
        return self._success({"message": message, "extra": kwargs})


# ── TestKwargsStripping ──────────────────────────────────────────────────────


class TestKwargsStripping:
    """run() should strip kwargs not accepted by _run()."""

    def test_strips_unexpected_kwargs(self, tool_context):
        """Extra kwargs like project_id are silently removed."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        result = tool.run(message="hi", project_id="fake", agent_id="fake")
        data = json.loads(result)
        assert data["message"] == "hi"

    def test_keeps_valid_kwargs(self, tool_context):
        """Valid kwargs reach _run unchanged."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        result = tool.run(message="world")
        data = json.loads(result)
        assert data["message"] == "world"

    def test_allows_var_keyword_run(self, tool_context):
        """If _run has **kwargs, nothing is stripped."""
        tool = _DummyToolVarKw()
        bind_tools_to_agent([tool], "test-agent")
        result = tool.run(message="hi", extra_param="kept")
        data = json.loads(result)
        # CrewAI's super().run() may or may not forward **kwargs;
        # the key assertion is that no TypeError is raised.
        assert data["message"] == "hi"

    def test_strips_multiple_extra(self, tool_context):
        """Multiple hallucinated params stripped at once."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        result = tool.run(message="ok", foo="bar", baz=42, qux=True)
        data = json.loads(result)
        assert data["message"] == "ok"

    def test_no_kwargs_at_all(self, tool_context):
        """Tool called with no kwargs uses default."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        result = tool.run()
        data = json.loads(result)
        assert data["message"] == "hello"


# ── TestResolvePath ──────────────────────────────────────────────────────────


class TestResolvePath:
    """_resolve_path() resolves relative paths against workspace."""

    def test_relative_resolved_against_workspace(self, tool_context, workspace_dir):
        """Relative path is joined with workspace_path."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        resolved = tool._resolve_path("foo.py")
        assert resolved == os.path.normpath(os.path.join(str(workspace_dir), "foo.py"))

    def test_absolute_returned_unchanged(self, tool_context):
        """Absolute path stays absolute."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        resolved = tool._resolve_path("/etc/hosts")
        assert resolved == "/etc/hosts"

    def test_resolve_with_no_workspace_uses_cwd(self):
        """When workspace_path is None, uses os.getcwd()."""
        # Patch workspace_path to None so fallback to cwd is used
        tool = _DummyTool()
        with patch.object(type(tool), "workspace_path", new_callable=lambda: property(lambda self: None)):
            resolved = tool._resolve_path("test.py")
            expected = os.path.normpath(os.path.join(os.getcwd(), "test.py"))
            assert resolved == expected

    def test_dot_slash_normalised(self, tool_context, workspace_dir):
        """./foo.py normalised to workspace/foo.py."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        resolved = tool._resolve_path("./foo.py")
        assert resolved == os.path.normpath(os.path.join(str(workspace_dir), "foo.py"))

    def test_parent_traversal(self, tool_context, workspace_dir):
        """../foo.py resolved correctly."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        resolved = tool._resolve_path("../foo.py")
        parent = os.path.dirname(str(workspace_dir))
        assert resolved == os.path.normpath(os.path.join(parent, "foo.py"))


# ── TestValidateSandboxPath ──────────────────────────────────────────────────


class TestValidateSandboxPath:
    """_validate_sandbox_path() enforces directory boundaries."""

    def test_sandbox_disabled_allows_everything(self, sandbox_disabled):
        """When sandbox is off, any path is allowed."""
        result = InfinibayBaseTool._validate_sandbox_path("/etc/passwd")
        assert result is None

    def test_sandbox_allows_within_allowed_dirs(self, sandbox_enabled, workspace_dir):
        """Path inside allowed dir returns None."""
        path = os.path.join(str(workspace_dir), "subdir", "file.txt")
        result = InfinibayBaseTool._validate_sandbox_path(path)
        assert result is None

    def test_sandbox_blocks_outside_dirs(self, sandbox_enabled):
        """Path outside allowed dirs returns error string."""
        result = InfinibayBaseTool._validate_sandbox_path("/etc/passwd")
        assert result is not None
        assert "Access denied" in result

    def test_sandbox_checks_directory_boundary(self, tmp_path):
        """'/tmp/foo' should NOT match when only '/tmp/foobar' is allowed."""
        orig_enabled = settings.SANDBOX_ENABLED
        orig_dirs = settings.ALLOWED_BASE_DIRS
        foobar = tmp_path / "foobar"
        foobar.mkdir()
        settings.SANDBOX_ENABLED = True
        settings.ALLOWED_BASE_DIRS = [str(foobar)]
        try:
            # "/tmp/.../foo" is NOT inside "/tmp/.../foobar"
            foo = tmp_path / "foo"
            foo.mkdir()
            test_file = foo / "test.txt"
            test_file.write_text("x")
            result = InfinibayBaseTool._validate_sandbox_path(str(test_file))
            assert result is not None
            assert "Access denied" in result
        finally:
            settings.SANDBOX_ENABLED = orig_enabled
            settings.ALLOWED_BASE_DIRS = orig_dirs

    def test_sandbox_exact_match_allowed(self, sandbox_enabled, workspace_dir):
        """Exact match of allowed dir is allowed."""
        result = InfinibayBaseTool._validate_sandbox_path(str(workspace_dir))
        assert result is None


# ── TestErrorSuccess ─────────────────────────────────────────────────────────


class TestErrorSuccess:
    """_error() and _success() produce consistent JSON."""

    def test_error_returns_json_with_error_key(self):
        """_error produces {"error": ...}."""
        tool = _DummyTool()
        result = tool._error("something went wrong")
        data = json.loads(result)
        assert data == {"error": "something went wrong"}

    def test_success_returns_string_directly(self):
        """_success with a plain string returns it as-is."""
        tool = _DummyTool()
        assert tool._success("plain text") == "plain text"

    def test_success_serialises_dict(self):
        """_success with a dict returns JSON."""
        tool = _DummyTool()
        result = tool._success({"key": "value", "num": 42})
        data = json.loads(result)
        assert data == {"key": "value", "num": 42}


# ── TestContextResolution ────────────────────────────────────────────────────


class TestContextResolution:
    """Context resolution from bound agent and fallbacks."""

    def test_bound_context_takes_precedence(self, tool_context):
        """Bound tool resolves context from process-global dict."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        assert tool.project_id == 1
        assert tool.agent_id == "test-agent"
        assert tool.session_id == "test-session"

    def test_unbound_tool_falls_back(self, tool_context):
        """Unbound tool falls back to thread-local context."""
        tool = _DummyTool()
        # Not bound, but thread-local is set by tool_context fixture
        assert tool.project_id == 1

    def test_validate_project_context_raises_on_none(self):
        """_validate_project_context raises ValueError when no project_id."""
        tool = _DummyTool()
        # Patch all context sources to return None
        with patch("infinidev.tools.base.base_tool.get_current_project_id", return_value=None):
            with pytest.raises(ValueError, match="No project_id"):
                tool._validate_project_context()

    def test_validate_agent_context_raises_on_none(self):
        """_validate_agent_context raises ValueError when no agent_id."""
        tool = _DummyTool()
        with patch("infinidev.tools.base.base_tool.get_current_agent_id", return_value=None):
            with pytest.raises(ValueError, match="No agent_id"):
                tool._validate_agent_context()

    def test_bind_delegate_propagates_agent_id(self, tool_context):
        """_bind_delegate stamps child tool with parent's agent_id."""
        parent = _DummyTool()
        bind_tools_to_agent([parent], "test-agent")
        child = _DummyTool()
        parent._bind_delegate(child)
        assert getattr(child, "_bound_agent_id") == "test-agent"


# ── TestGitCwd ───────────────────────────────────────────────────────────────


class TestGitCwd:
    """_git_cwd property returns workspace or None."""

    def test_returns_workspace_when_exists(self, tool_context, workspace_dir):
        """Returns workspace_path when directory exists."""
        tool = _DummyTool()
        bind_tools_to_agent([tool], "test-agent")
        assert tool._git_cwd == str(workspace_dir)

    def test_returns_none_when_no_workspace(self):
        """Returns None when workspace not set."""
        tool = _DummyTool()
        with patch.object(type(tool), "workspace_path", new_callable=lambda: property(lambda self: None)):
            assert tool._git_cwd is None
