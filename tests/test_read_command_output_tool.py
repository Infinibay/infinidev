"""Bounded, scope-safe reads of private command output."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from infinidev.engine.command_output_store import (
    COMMAND_OUTPUT_MAX_READ_BYTES,
    CommandOutputStore,
)
from infinidev.tools.base.context import bind_tools_to_agent, clear_agent_context
from infinidev.tools.knowledge.read_command_output import ReadCommandOutputTool


def _arguments(handle, **overrides) -> dict:
    values = {
        "artifact_id": handle.artifact_id,
        "type": handle.artifact_type,
        "stream": handle.stream,
        "char_count": handle.char_count,
        "byte_count": handle.byte_count,
        "offset": 0,
        "limit": 16_384,
    }
    values.update(overrides)
    return values


def _bound_reader(bound_tool) -> ReadCommandOutputTool:
    return bound_tool(ReadCommandOutputTool)


def _store(tmp_path, temp_db) -> CommandOutputStore:
    return CommandOutputStore(
        root=tmp_path / "private" / "command_output",
        db_path=temp_db,
    )


def _bind(tool: ReadCommandOutputTool, agent_id: str) -> None:
    bind_tools_to_agent([tool], agent_id)


def _storage_id(db_path: str, artifact_id: int) -> str:
    conn = sqlite3.connect(db_path)
    try:
        reference = conn.execute(
            "SELECT file_path FROM artifacts WHERE id = ?", (artifact_id,)
        ).fetchone()[0]
    finally:
        conn.close()
    return str(reference).split(".", 1)[0]


def test_registered_reader_round_trips_exact_bounded_ranges(
    bound_tool, tmp_path, temp_db, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    text = "prefix-π-middle-λ-suffix"
    handle = CommandOutputStore().store_streams(
        project_id=1,
        session_id="test-session",
        streams={"stdout": text},
    )["stdout"]
    tool = _bound_reader(bound_tool)

    chunks: list[str] = []
    offset = 0
    while True:
        result = json.loads(tool._run(**_arguments(handle, offset=offset, limit=8)))
        assert "file_path" not in result
        assert result["returned_bytes"] <= 8
        chunks.append(result["content"])
        if not result["has_more"]:
            assert result["next_offset"] is None
            break
        assert result["next_offset"] > offset
        offset = result["next_offset"]

    assert "".join(chunks) == text


def test_reader_rejects_missing_context(tmp_path, temp_db):
    store = _store(tmp_path, temp_db)
    handle = store.store_streams(
        project_id=1, session_id="owner", streams={"stdout": "private"}
    )["stdout"]

    result = json.loads(ReadCommandOutputTool()._run(**_arguments(handle)))

    assert "error" in result
    assert "bound project and session" in result["error"]


def test_reader_rejects_cross_session_and_project(
    tmp_path, temp_db, tool_context
):
    store = _store(tmp_path, temp_db)
    handle = store.store_streams(
        project_id=1, session_id="owner", streams={"stdout": "private"}
    )["stdout"]
    tool = ReadCommandOutputTool()
    _bind(tool, "test-agent")

    wrong_session = json.loads(tool._run(**_arguments(handle)))
    assert "scope" in wrong_session["error"]

    from infinidev.tools.base.context import set_context

    set_context(
        project_id=2,
        agent_id="other-project",
        agent_run_id="run-2",
        session_id="owner",
        workspace_path=str(tmp_path),
    )
    other = ReadCommandOutputTool()
    _bind(other, "other-project")
    try:
        wrong_project = json.loads(other._run(**_arguments(handle)))
    finally:
        clear_agent_context("other-project")
    assert "scope" in wrong_project["error"]


def test_reader_requires_exact_type_stream_and_lengths(
    bound_tool, tmp_path, temp_db
):
    store = _store(tmp_path, temp_db)
    handle = store.store_streams(
        project_id=1,
        session_id="test-session",
        streams={"stdout": "private"},
    )["stdout"]
    tool = _bound_reader(bound_tool)

    for changed in (
        {"type": "report"},
        {"stream": "stderr"},
        {"char_count": handle.char_count + 1},
        {"byte_count": handle.byte_count + 1},
    ):
        result = json.loads(tool._run(**_arguments(handle, **changed)))
        assert "error" in result


def test_reader_enforces_range_bounds_and_utf8_boundaries(
    bound_tool, tmp_path, temp_db
):
    store = _store(tmp_path, temp_db)
    handle = store.store_streams(
        project_id=1,
        session_id="test-session",
        streams={"stdout": "πtail"},
    )["stdout"]
    tool = _bound_reader(bound_tool)

    cases = (
        {"offset": -1},
        {"offset": handle.byte_count + 1},
        {"limit": 0},
        {"limit": COMMAND_OUTPUT_MAX_READ_BYTES + 1},
        {"offset": 1},
        {"limit": 1},
    )
    for changed in cases:
        result = json.loads(tool._run(**_arguments(handle, **changed)))
        assert "error" in result


@pytest.mark.parametrize("attack", ["traversal", "symlink", "corrupt", "substitute"])
def test_reader_fails_closed_for_private_store_attacks(
    attack, bound_tool, tmp_path, temp_db, monkeypatch
):
    # Use the production root rather than injecting a test-only root: this proves
    # the registered tool resolves only the fixed cwd-relative private store.
    monkeypatch.chdir(tmp_path)
    store = CommandOutputStore()
    handle = store.store_streams(
        project_id=1,
        session_id="test-session",
        streams={"stdout": "original"},
    )["stdout"]
    storage_id = _storage_id(temp_db, handle.artifact_id)

    if attack == "traversal":
        conn = sqlite3.connect(temp_db)
        conn.execute(
            "UPDATE artifacts SET file_path = '../../etc/passwd' WHERE id = ?",
            (handle.artifact_id,),
        )
        conn.commit()
        conn.close()
    elif attack == "symlink":
        outside = tmp_path / "outside-secret"
        outside.write_text("do-not-read")
        blob = store.root / f"{storage_id}.blob"
        blob.unlink()
        blob.symlink_to(outside)
    elif attack == "corrupt":
        blob = store.root / f"{storage_id}.blob"
        blob.write_text("same-len")
        os.chmod(blob, 0o600)
    else:
        sidecar = store.root / f"{storage_id}.json"
        payload = json.loads(sidecar.read_text())
        payload["created_at"] += 1
        sidecar.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":"))
        )
        os.chmod(sidecar, 0o600)

    result = json.loads(_bound_reader(bound_tool)._run(**_arguments(handle)))
    assert "error" in result
    assert "do-not-read" not in result["error"]
    assert str(store.root) not in result["error"]


def test_reader_registration_is_gated_without_changing_disabled_prompt(monkeypatch):
    from infinidev.config.settings import settings
    from infinidev.prompts.tool_hints import TOOL_DESCRIPTIONS
    from infinidev.tools import get_tools_for_role

    def names(*, small_model: bool = False) -> set[str]:
        return {
            tool.name
            for tool in get_tools_for_role(
                "developer", small_model=small_model, supports_vision=False
            )
            if getattr(tool, "mcp_server", None) is None
        }

    monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", False)
    assert "read_command_output" not in names()
    assert "read_command_output" not in names(small_model=True)

    monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
    assert "read_command_output" in names()
    assert "read_command_output" in names(small_model=True)
    assert "read_command_output" in TOOL_DESCRIPTIONS
