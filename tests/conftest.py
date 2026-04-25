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


@pytest.fixture(autouse=True)
def _reset_db_conn_cache():
    """Drop the process-wide pooled SQLite connection between tests.

    The connection cache in ``infinidev.code_intel._db`` is keyed on
    ``settings.DB_PATH`` and *should* re-open when the path changes,
    but several tests mutate ``DB_PATH`` only inside a fixture and
    leave a stale connection bound to a temp file that gets unlinked
    on teardown. The next test then talks to a closed-over file
    descriptor or, worse, to a recycled inode that holds rows from a
    previous run. Forcing a clean slate here avoids that whole class
    of leak.
    """
    yield
    try:
        from infinidev.code_intel._db import _conn_cache
        cached = getattr(_conn_cache, "conn", None)
        if cached is not None:
            try:
                cached.close()
            except Exception:
                pass
        _conn_cache.conn = None
        _conn_cache.path = None
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_tool_context_thread_locals():
    """Clear the tool-context thread-local + ContextVar storage between tests.

    ``clear_agent_context`` removes the process-global dict entry but
    leaves the ``threading.local`` (``_tls``) and the ``ContextVar``
    backing store populated. Tests that patch
    ``get_current_project_id`` / ``get_current_agent_id`` to return
    None therefore see leftover values from earlier tests that ran
    ``set_context``, breaking assertions like "raises when no project_id".
    """
    yield
    try:
        from infinidev.tools.base.context import (
            _tls, _project_id_var, _agent_id_var, _agent_run_id_var,
            _session_id_var, _workspace_path_var, _agent_contexts,
            _agent_contexts_lock,
        )
        for attr in ("project_id", "agent_id", "agent_run_id",
                     "session_id", "workspace_path"):
            if hasattr(_tls, attr):
                delattr(_tls, attr)
        _project_id_var.set(None)
        _agent_id_var.set(None)
        _agent_run_id_var.set(None)
        _session_id_var.set(None)
        _workspace_path_var.set(None)
        with _agent_contexts_lock:
            _agent_contexts.clear()
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _reset_capabilities_singleton():
    """Reset the model-capabilities singleton between tests.

    ``probe_model`` mutates a module-level ``_capabilities`` object;
    tests that probe with mocked litellm leave that singleton in a
    state that causes downstream tests asserting against a fresh
    capabilities object to see stale flags (e.g. ``probed=True``,
    ``supports_function_calling=False``).
    """
    yield
    try:
        import infinidev.config.model_capabilities as _mc
        _mc._reset_capabilities()
    except Exception:
        pass


def pytest_ignore_collect(collection_path, config):
    """Ignore interactive tests by default unless INFINIDEV_RUN_INTERACTIVE_TESTS=1."""
    import os
    if collection_path.name == "interactive":
        run_interactive = os.environ.get("INFINIDEV_RUN_INTERACTIVE_TESTS", "0") in ("1", "true", "True")
        if not run_interactive:
            return True
