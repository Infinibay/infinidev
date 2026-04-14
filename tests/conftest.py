"""Shared test fixtures for Infinidev test suite."""

import os
import tempfile

import pytest

from infinidev.config.settings import settings
from infinidev.db.service import init_db
from infinidev.tools.base.context import (
    ToolContext,
    _agent_contexts,
    _agent_contexts_lock,
    bind_tools_to_agent,
    clear_agent_context,
    set_context,
)
from infinidev.tools.base.db import execute_with_retry


# ── Database fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def temp_db_path():
    """Create a temporary SQLite database file."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    yield path
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


@pytest.fixture
def temp_db(temp_db_path):
    """Initialise a temporary database and patch settings.DB_PATH."""
    original = settings.DB_PATH
    settings.DB_PATH = temp_db_path
    init_db()
    yield temp_db_path
    settings.DB_PATH = original


# ── Workspace fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def workspace_dir(tmp_path):
    """Create a temporary workspace directory with sample files."""
    sample = tmp_path / "sample.txt"
    sample.write_text("line one\nline two\nline three\nline four\nline five\n")
    return tmp_path


# ── Tool context fixtures ────────────────────────────────────────────────────


@pytest.fixture
def tool_context(workspace_dir, temp_db):
    """Set up a full agent context for tool testing."""
    agent_id = "test-agent"
    ctx = set_context(
        project_id=1,
        agent_id=agent_id,
        agent_run_id="run-1",
        session_id="test-session",
        workspace_path=str(workspace_dir),
    )
    yield ctx
    clear_agent_context(agent_id)


@pytest.fixture
def bound_tool(tool_context):
    """Factory fixture: instantiate a tool and bind it to the test agent.

    Usage::

        def test_something(bound_tool):
            tool = bound_tool(ReadFileTool)
            result = tool._run(path="sample.txt")
    """

    def _factory(tool_cls, **kwargs):
        tool = tool_cls(**kwargs)
        bind_tools_to_agent([tool], "test-agent")
        return tool

    return _factory


# ── Settings override helpers ────────────────────────────────────────────────


@pytest.fixture
def sandbox_disabled():
    """Ensure sandbox is disabled for the duration of the test."""
    original = settings.SANDBOX_ENABLED
    settings.SANDBOX_ENABLED = False
    yield
    settings.SANDBOX_ENABLED = original


@pytest.fixture
def sandbox_enabled(workspace_dir):
    """Enable sandbox, allowing only the workspace directory."""
    orig_enabled = settings.SANDBOX_ENABLED
    orig_dirs = settings.ALLOWED_BASE_DIRS
    settings.SANDBOX_ENABLED = True
    settings.ALLOWED_BASE_DIRS = [str(workspace_dir)]
    yield
    settings.SANDBOX_ENABLED = orig_enabled
    settings.ALLOWED_BASE_DIRS = orig_dirs


@pytest.fixture
def auto_approve_permissions():
    """Set all permission modes to auto_approve."""
    orig_exec = settings.EXECUTE_COMMANDS_PERMISSION
    orig_file = settings.FILE_OPERATIONS_PERMISSION
    settings.EXECUTE_COMMANDS_PERMISSION = "auto_approve"
    settings.FILE_OPERATIONS_PERMISSION = "auto_approve"
    yield
    settings.EXECUTE_COMMANDS_PERMISSION = orig_exec
    settings.FILE_OPERATIONS_PERMISSION = orig_file


def pytest_ignore_collect(collection_path, config):
    """Ignore interactive tests by default unless INFINIDEV_RUN_INTERACTIVE_TESTS=1."""
    import os
    if collection_path.name == "interactive":
        run_interactive = os.environ.get("INFINIDEV_RUN_INTERACTIVE_TESTS", "0") in ("1", "true", "True")
        if not run_interactive:
            return True
