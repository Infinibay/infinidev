"""Provider-boundary checks for verified code excerpts in a principal turn."""

from __future__ import annotations

import hashlib
import json
from unittest.mock import Mock

import pytest

from infinidev.config.settings import settings
from infinidev.engine.context_recording import recording
from infinidev.engine.ken_client import KenClient
from infinidev.tools.file.read_file_tool import ReadFileTool
from tests.test_ken_code_search_contract import SearchManager
from tests.test_turn_runtime import reply, runtime


@pytest.mark.parametrize("case", [
    "current", "disabled", "no_read", "pod", "stale", "changed", "escape",
    "bad_range", "lookup_error", "larger_than_budget",
])
def test_principal_uses_only_verified_local_ken_source(
    case, runtime, workspace_dir, tmp_path_factory, monkeypatch, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    monkeypatch.setattr(settings, "TASK_KEN_SOURCE_CONTEXT_ENABLED", case != "disabled")
    monkeypatch.setattr(settings, "KEN_SESSION_ENABLED", True)
    monkeypatch.setattr(settings, "MCP_ENABLED", True)
    monkeypatch.setattr(settings, "SANDBOX_ENABLED", case == "pod")
    tool = ReadFileTool()
    agent.tools = [tool]
    literal = "Use the current helper API."
    lines = ["def current_helper():", "    # local helper", "    return '<fresh>'"]
    if case == "larger_than_budget":
        lines[-1] = "    return " + repr("x" * 7000)
    text = "\n".join(lines) + "\n"
    if case == "escape":
        outside = tmp_path_factory.mktemp("other-workspace") / "source.py"
        outside.write_text(text)
        (workspace_dir / "source.py").symlink_to(outside)
    else:
        (workspace_dir / "source.py").write_text(text)

    row = {
        "path": "source.py",
        "entity": {"kind": "function", "qualname": "current_helper"},
        "entity_span": {"start_line": 1, "end_line": 3},
        "source_hash": hashlib.sha256(text.encode()).hexdigest(),
        "stale": case == "stale",
        "truncated": True,
        "evidence": [
            {"start_line": 1, "end_line": 1, "content": "INDEX COPY SHOULD NOT BE SENT"},
            {"start_line": 3, "end_line": 3, "content": "INDEX COPY SHOULD NOT BE SENT"},
        ],
    }
    if case == "changed":
        (workspace_dir / "source.py").write_text(text.replace("fresh", "changed"))
    if case == "bad_range":
        row["evidence"] = [{"start_line": 0, "end_line": 4, "content": "Invalid range"}]
    if case == "larger_than_budget":
        row["evidence"] = row["evidence"][1:]

    client = KenClient(SearchManager({"results": [row], "truncated": False, "omitted_count": 0}))
    search = Mock(wraps=client.search_symbols)
    if case == "lookup_error":
        search.side_effect = ValueError("unrecognized search result")
    monkeypatch.setattr(client, "search_symbols", search)
    monkeypatch.setattr("infinidev.engine.ken_client.get_ken_client", lambda: client)
    script(reply("Ready."))
    capture_dir = tmp_path_factory.mktemp("record-source") / "capture"
    with recording(capture_dir, workspace=workspace_dir):
        result = engine.run_turn(agent, literal, task_tools=[] if case == "no_read" else [tool])
    assert result.status == "answered"
    assert result.metrics["observed_tool_calls"] == 0
    assert seen[0][0][-1]["content"] == literal
    if case in {"disabled", "no_read", "pod"}:
        search.assert_not_called()
    else:
        search.assert_called_once_with(literal, limit=2)
    blocks = [message["content"] for message in seen[0][0]
              if message["content"].startswith("<verified-source-context")]
    if case != "current":
        assert blocks == []
        return
    block, = blocks
    assert "<fresh>" not in block
    evidence = json.loads(block.splitlines()[1])
    assert evidence["partial"] is True
    assert [(part["start_line"], part["end_line"], part["content"])
            for part in evidence["excerpts"]] == [(1, 1, lines[0]), (3, 3, lines[2])]
    assert all(part["path"] == "source.py" and part["source_sha256"] == row["source_hash"]
               for part in evidence["excerpts"])
    assert "INDEX COPY SHOULD NOT BE SENT" not in block
    captured = (capture_dir / "metadata.jsonl").read_text()
    assert "current_helper" not in captured and row["source_hash"] not in captured
