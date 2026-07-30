"""Tests for the MCP client against a protocol-conformant fake server."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from infinidev.engine import mcp_client as mcp_module
from infinidev.engine.ken_client import KenClient, get_ken_client, reset_ken_client
from infinidev.engine.mcp_client import (
    McpManager,
    McpUnavailable,
    keyword_search,
    load_mcp_config,
    parse_tool_result,
    reset_default_mcp_manager,
    resolve_mcp_servers,
)

FAKE_SERVER = str(Path(__file__).parent / "mcp_fake_server.py")


def server_config(*flags: str, **extra) -> dict:
    return {"command": sys.executable, "args": [FAKE_SERVER, *flags], **extra}


@pytest.fixture
def manager():
    mgr = McpManager({"ken": server_config()})
    yield mgr
    mgr.close()


@pytest.fixture(autouse=True)
def _reset_singletons():
    reset_default_mcp_manager()
    reset_ken_client()
    yield
    reset_default_mcp_manager()
    reset_ken_client()


# ── protocol ──────────────────────────────────────────────────────────────


def test_handshake_precedes_tool_listing(manager):
    tools = manager.get("ken").list_tools()
    assert [tool.name for tool in tools][:2] == ["ken_find", "ken_read"]
    assert manager.get("ken")._initialized is True


def test_response_is_matched_by_id_not_line_order(manager):
    # The fake emits a notification before every answer; a client that
    # returned the first line it read would get {} here.
    result = manager.call("ken", "ken_recall", {"query": "jwt"})
    assert result.is_error is False
    assert result.rows()[0]["topic"] == "jwt-clock-skew"


def test_structured_and_text_content_agree(manager):
    result = manager.call(
        "ken", "ken_find", {"query": "verify", "scope": "symbols"},
    )
    assert result.data[0]["qualname"] == "verify_token"
    assert "verify_token" in result.text


def test_tool_error_is_reported_not_swallowed(manager):
    result = manager.call("ken", "does_not_exist", {})
    assert result.is_error is True
    assert "Unknown tool" in result.text


def test_failed_handshake_marks_server_unavailable():
    mgr = McpManager({"ken": server_config("--no-handshake")})
    with pytest.raises(McpUnavailable):
        mgr.get("ken").list_tools()
    status = mgr.status()["ken"]
    assert status["initialized"] is False
    assert "initialize refused" in status["reason"]
    mgr.close()


def test_hung_server_times_out_instead_of_blocking():
    mgr = McpManager({"ken": server_config("--hang", timeout=1)})
    with pytest.raises(McpUnavailable, match="timed out"):
        mgr.get("ken").list_tools()
    mgr.close()


def test_non_json_stdout_noise_is_ignored():
    mgr = McpManager({"ken": server_config("--noise")})
    assert mgr.get("ken").list_tools()
    assert mgr.get("ken").stderr_tail()
    mgr.close()


def test_crashed_server_surfaces_and_backs_off():
    mgr = McpManager({"ken": server_config("--crash-after", "1")})
    client = mgr.get("ken")
    with pytest.raises(McpUnavailable):
        client.list_tools()
    assert client._failure_count >= 1
    assert client._next_retry_at is not None
    mgr.close()


def test_try_call_returns_none_instead_of_raising():
    mgr = McpManager({"ken": {"command": "definitely-missing-binary"}})
    assert mgr.try_call("ken", "ken_recall", {"query": "x"}) is None
    assert mgr.names() == []
    mgr.close()


def test_backoff_caps_at_8_seconds():
    from infinidev.engine.mcp_client import McpServerClient

    client = McpServerClient("missing", "definitely-missing-ken")
    client._failure_count = 10
    assert client._backoff_delay() == 8.0


def test_parse_tool_result_handles_plain_text():
    result = parse_tool_result({"content": [{"type": "text", "text": "hello"}]})
    assert result.text == "hello"
    assert result.data is None
    assert result.rows() == []


def test_parse_tool_result_handles_multi_block_lists():
    result = parse_tool_result(
        {
            "content": [
                {"type": "text", "text": json.dumps({"path": "a.py"})},
                {"type": "text", "text": json.dumps({"path": "b.py"})},
            ]
        }
    )
    assert [row["path"] for row in result.rows()] == ["a.py", "b.py"]


# ── configuration ─────────────────────────────────────────────────────────


def test_load_mcp_config_parses_dot_mcp_json(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"alpha": {"command": "echo", "args": []}}})
    )
    config = load_mcp_config()
    assert config["alpha"]["command"] == "echo"


def test_ken_is_the_default_server_without_config(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mcp_module, "get_base_dir", lambda: tmp_path / ".infinidev")
    servers = resolve_mcp_servers()
    assert servers["ken"]["command"] == "ken"
    assert servers["ken"]["args"] == ["mcp"]


def test_declared_servers_keep_ken_alongside(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".mcp.json").write_text(
        json.dumps({"mcpServers": {"docs": {"command": "docs-server"}}})
    )
    monkeypatch.setattr(mcp_module, "get_base_dir", lambda: tmp_path / ".infinidev")
    servers = resolve_mcp_servers()
    assert set(servers) == {"docs", "ken"}


def test_mcp_disabled_yields_no_servers(monkeypatch):
    monkeypatch.setattr(mcp_module.settings, "MCP_ENABLED", False)
    assert resolve_mcp_servers() == {}


def test_per_server_timeout_overrides_global(monkeypatch):
    monkeypatch.setattr(mcp_module.settings, "MCP_REQUEST_TIMEOUT", 99)
    mgr = McpManager({"ken": server_config(timeout=7)})
    assert mgr.get("ken")._timeout == 7
    mgr.close()


def test_tools_cache_is_reused_and_invalidated(manager):
    client = manager.get("ken")
    first = client.list_tools()
    assert client.list_tools() is first  # cached instance
    client.invalidate_tools_cache()
    assert client.list_tools() is not first


def test_manager_start_stop_restart(manager):
    assert manager.start("ken") is True
    first_pid = manager.get("ken")._process.pid
    assert manager.restart("ken") is True
    assert manager.get("ken")._process.pid != first_pid
    assert manager.stop("ken") is True
    assert manager.get("ken").running is False


def test_manager_start_returns_false_for_unknown_server():
    mgr = McpManager()
    assert mgr.start("missing") is False
    assert mgr.stop("missing") is False


def test_manager_emits_lifecycle_events():
    events: list[dict] = []
    mgr = McpManager({"ken": server_config()}, on_event=events.append)
    mgr.call("ken", "ken_recall", {"query": "x"})
    kinds = [event["event"] for event in events]
    assert "started" in kinds and "ready" in kinds
    assert "tool_call" in kinds and "tool_result" in kinds
    mgr.close()


# ── Ken facade ────────────────────────────────────────────────────────────


def test_ken_facade_maps_every_result_shape(monkeypatch):
    monkeypatch.setattr(
        mcp_module, "resolve_mcp_servers", lambda: {"ken": server_config()}
    )
    client = get_ken_client()
    assert client.available is True

    files = client.search_files("auth")
    assert files[0].target == "src/auth.py"
    assert "verify_token" in files[0].snippet

    symbols = client.search_symbols("auth")
    assert symbols[0].qualname == "verify_token" and symbols[0].line == 12

    matches = client.grep("verify_token")
    assert matches[0].path == "src/auth.py" and matches[0].line == 12

    memories = client.recall("jwt")
    assert memories[0].topic == "jwt-clock-skew"

    assert "<context-rank>" in client.rank("auth")
    assert client.remember("topic", "content", ["tag"]) is True


def test_ken_facade_falls_back_to_keyword_search(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_module, "resolve_mcp_servers", lambda: {})
    memory_dir = tmp_path / ".infinidev" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "x.md").write_text("password rotation happens quarterly")
    monkeypatch.setattr(mcp_module, "get_base_dir", lambda: tmp_path / ".infinidev")

    client = get_ken_client()
    assert client.available is False
    hits = client.recall("password rotation")
    assert hits and "rotation" in hits[0].content


def test_ken_facade_never_raises_when_server_dies(monkeypatch):
    monkeypatch.setattr(
        mcp_module,
        "resolve_mcp_servers",
        lambda: {"ken": server_config("--crash-after", "1")},
    )
    client = get_ken_client()
    # Crashes mid-session; the facade degrades instead of propagating.
    assert client.search_symbols("anything") == []
    assert client.grep("anything") == []
    assert client.rank("anything") == ""


def test_keyword_search_returns_memory_hits(monkeypatch, tmp_path):
    memory_dir = tmp_path / ".infinidev" / "memory"
    memory_dir.mkdir(parents=True)
    (memory_dir / "feature.md").write_text(
        "Feature highlights: auth tokens are short lived."
    )
    monkeypatch.setattr(mcp_module, "get_base_dir", lambda: tmp_path / ".infinidev")
    hits = keyword_search("auth tokens", limit=5, kinds={"memory"})
    assert hits and hits[0].source == "fallback"
    assert hits[0].line > 0


def test_ken_client_status_reports_reason(monkeypatch):
    monkeypatch.setattr(mcp_module, "resolve_mcp_servers", lambda: {})
    status = KenClient().status()
    assert status["fallback"] is True
    assert "no 'ken' server" in status["reason"]
