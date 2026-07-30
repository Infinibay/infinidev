"""Ken-backed tools must degrade, never break, when MCP is unavailable.

The protocol itself is covered by ``test_mcp_client.py``. These tests are
about the promise made to the agent: a workspace with no Ken index, a dead
server, or a hung server still produces useful tool output.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from infinidev.engine import mcp_client as mcp_module
from infinidev.engine.ken_client import reset_ken_client
from infinidev.engine.mcp_client import reset_default_mcp_manager

FAKE_SERVER = str(Path(__file__).parent / "mcp_fake_server.py")


def server_config(*flags: str, **extra) -> dict:
    return {"command": sys.executable, "args": [FAKE_SERVER, *flags], **extra}


@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_default_mcp_manager()
    reset_ken_client()
    yield
    reset_default_mcp_manager()
    reset_ken_client()


@pytest.fixture
def no_ken(monkeypatch):
    """Simulate a workspace where Ken is not installed at all."""
    monkeypatch.setattr(mcp_module, "resolve_mcp_servers", lambda: {})


@pytest.fixture
def dead_ken(monkeypatch):
    """Simulate a Ken that starts but dies mid-session."""
    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": server_config("--crash-after", "1")},
    )


@pytest.fixture
def live_ken(monkeypatch):
    monkeypatch.setattr(
        mcp_module, "resolve_mcp_servers", lambda: {"ken": server_config()}
    )


def _tool(cls, tmp_path):
    """Instantiate a tool bound to *tmp_path* as its workspace."""
    from infinidev.tools.base.context import set_context

    set_context(project_id=1, workspace_path=str(tmp_path))
    return cls()


# ── code_search / glob stay deterministic regardless of Ken ───────────────


def test_code_search_is_independent_of_ken(no_ken, tmp_path):
    """The literal search path must never depend on an index being present."""
    from infinidev.tools.file.code_search_tool import CodeSearchTool

    (tmp_path / "auth.py").write_text("def verify_token(tok):\n    return True\n")
    result = json.loads(
        _tool(CodeSearchTool, tmp_path)._run(
            pattern="verify_token", file_path=str(tmp_path)
        )
    )
    assert result["match_count"] == 1
    assert result["matches"][0]["line"] == 1
    assert "verify_token" in result["matches"][0]["content"]


def test_glob_is_independent_of_ken(no_ken, tmp_path):
    """Glob must return real paths matching the pattern, with real stats."""
    from infinidev.tools.file.glob_tool import GlobTool

    (tmp_path / "a.py").write_text("x = 1\n")
    (tmp_path / "b.txt").write_text("nope\n")
    result = json.loads(
        _tool(GlobTool, tmp_path)._run(pattern="*.py", file_path=str(tmp_path))
    )
    assert [m["file_path"] for m in result["matches"]] == ["a.py"]
    assert result["matches"][0]["size"] > 0


def test_code_search_still_deterministic_with_ken_running(live_ken, tmp_path):
    """A live Ken must not change literal search results."""
    from infinidev.tools.file.code_search_tool import CodeSearchTool

    (tmp_path / "auth.py").write_text("def verify_token(tok):\n    return True\n")
    result = json.loads(
        _tool(CodeSearchTool, tmp_path)._run(
            pattern="verify_token", file_path=str(tmp_path)
        )
    )
    assert result["match_count"] == 1
    assert result["matches"][0]["file"].endswith("auth.py")


# ── Bridged MCP tools degrade instead of raising ──────────────────────────
#
# The tools themselves are generated from the server's ``tools/list``, so
# "Ken is unavailable" means the tool does not exist rather than that it
# returns a fallback. What must never happen is a crash, a hang, or a
# discovery that blocks the caller.


@pytest.fixture(autouse=True)
def _reset_bridge_cache():
    from infinidev.tools.mcp_bridge import reset_discovery_cache

    reset_discovery_cache()
    yield
    reset_discovery_cache()


def _bridged(tmp_path) -> dict:
    from infinidev.tools.base.context import set_context
    from infinidev.tools.mcp_bridge import discover_mcp_tool_classes

    set_context(project_id=1, workspace_path=str(tmp_path))
    return {cls().name: cls() for cls in discover_mcp_tool_classes(block=True)}


def test_no_server_means_no_tools_not_a_crash(no_ken, tmp_path):
    assert _bridged(tmp_path) == {}


def test_live_server_tools_keep_the_names_the_server_published(live_ken, tmp_path):
    """The whole point of the bridge: ``ken_rank`` is called ``ken_rank``."""
    tools = _bridged(tmp_path)
    assert {"ken_find", "ken_read", "ken_rank", "ken_recall", "ken_remember"} <= set(tools)


def test_a_bridged_tool_returns_the_servers_payload(live_ken, tmp_path):
    output = _bridged(tmp_path)["ken_find"].run(query="auth", scope="files")
    assert "src/auth.py" in output
    assert "verify_token" in output


def test_writers_are_kept_out_of_the_read_only_tiers(live_ken, tmp_path):
    """``is_read_only`` is a security boundary — ``ken_remember`` writes."""
    tools = _bridged(tmp_path)
    assert tools["ken_find"].is_read_only is True
    assert tools["ken_recall"].is_read_only is True
    assert tools["ken_remember"].is_read_only is False


def test_dead_server_does_not_propagate_to_tools(dead_ken, tmp_path):
    # Discovery survives a server that dies mid-session; whatever it hands
    # back, calling it must return an error string rather than raise.
    for tool in _bridged(tmp_path).values():
        assert isinstance(tool.run(query="anything"), str)


def test_discovery_never_blocks_the_caller_on_a_hung_server(monkeypatch, tmp_path):
    """``get_tools_for_role`` runs once per turn; it must not wait on a
    subprocess. The non-blocking path only reads listings a *running*
    server already produced."""
    import time

    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": server_config("--hang", timeout=1, startup_timeout=5)},
    )
    from infinidev.tools.mcp_bridge import discover_mcp_tool_classes

    started = time.monotonic()
    assert discover_mcp_tool_classes() == []
    assert time.monotonic() - started < 1.0


def test_hung_server_times_out_within_the_configured_budget(monkeypatch, tmp_path):
    import time

    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": server_config("--hang", timeout=1, startup_timeout=5)},
    )
    started = time.monotonic()
    assert _bridged(tmp_path) == {}
    assert time.monotonic() - started < 15
