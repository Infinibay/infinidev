"""Recoverable file lifecycle and atomic patch tools."""

from __future__ import annotations

import json

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.workspace_baseline import WorkspaceBaseline
from infinidev.tools.base.context import set_file_tracker
from infinidev.tools.file.safe_file_tools import (
    ApplyFilePatchTool,
    DeleteFileTool,
    MoveFileTool,
    PreviewChangesTool,
    RollbackTaskChangesTool,
)


def _payload(result: str) -> dict:
    parsed = json.loads(result)
    assert "error" not in parsed, parsed
    return parsed


def test_delete_file_is_recoverable(bound_tool, workspace_dir, auto_approve_permissions) -> None:
    path = workspace_dir / "remove.txt"
    path.write_text("keep me\n")
    tool = bound_tool(DeleteFileTool)

    payload = _payload(tool.run(file_path="remove.txt", rationale="obsolete"))

    assert not path.exists()
    recovery = workspace_dir / ".infinidev" / "trash"
    assert str(recovery) in payload["recovery_path"]
    with open(payload["recovery_path"], encoding="utf-8") as handle:
        assert handle.read() == "keep me\n"


def test_move_refuses_implicit_overwrite(bound_tool, workspace_dir, auto_approve_permissions) -> None:
    source = workspace_dir / "source.txt"
    destination = workspace_dir / "destination.txt"
    source.write_text("source\n")
    destination.write_text("user content\n")
    tool = bound_tool(MoveFileTool)

    result = json.loads(tool.run(
        source_path="source.txt",
        destination_path="destination.txt",
    ))

    assert "error" in result
    assert source.read_text() == "source\n"
    assert destination.read_text() == "user content\n"


def test_patch_validates_every_replacement_before_writing(
    bound_tool,
    workspace_dir,
    auto_approve_permissions,
) -> None:
    path = workspace_dir / "sample.txt"
    original = path.read_text()
    tool = bound_tool(ApplyFilePatchTool)

    result = json.loads(tool.run(
        file_path="sample.txt",
        replacements=[
            {"old_string": "line one", "new_string": "changed"},
            {"old_string": "missing", "new_string": "never written"},
        ],
    ))

    assert "error" in result
    assert path.read_text() == original


def test_preview_uses_patch_validation_without_writing(
    bound_tool,
    workspace_dir,
) -> None:
    path = workspace_dir / "sample.txt"
    original = path.read_text()
    tool = bound_tool(PreviewChangesTool)

    result = tool.run(
        file_path="sample.txt",
        replacements=[{"old_string": "line two", "new_string": "second line"}],
    )

    assert "-line two" in result
    assert "+second line" in result
    assert path.read_text() == original


def test_rollback_restores_task_start_state_and_removes_new_files(
    bound_tool,
    workspace_dir,
    auto_approve_permissions,
) -> None:
    existing = workspace_dir / "sample.txt"
    user_state = existing.read_text()
    baseline = WorkspaceBaseline.capture(str(workspace_dir))
    tracker = FileChangeTracker(baseline)
    set_file_tracker("test-agent", tracker)

    existing.write_text("agent state\n")
    created = workspace_dir / "created.txt"
    created.write_text("new\n")
    tool = bound_tool(RollbackTaskChangesTool)

    payload = _payload(tool.run(rationale="revert this task"))

    assert existing.read_text() == user_state
    assert not created.exists()
    assert str(existing) in payload["restored"]
    assert str(created) in payload["removed_new_files"]
    assert tracker.get_all_paths() == []


def test_change_fingerprint_returns_to_entry_after_edit_is_reverted(
    workspace_dir,
) -> None:
    path = workspace_dir / "sample.txt"
    original = path.read_text()
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(workspace_dir)))
    entry = tracker.change_fingerprint(reconcile=True)

    path.write_text("temporary agent edit\n")
    tracker.record(str(path), original, path.read_text())
    changed = tracker.change_fingerprint()
    path.write_text(original)
    tracker.record(str(path), "temporary agent edit\n", original)

    assert changed != entry
    assert tracker.change_fingerprint(reconcile=True) == entry
