"""Tests for deterministic tracking of successful workspace effects."""

from __future__ import annotations

import json

from infinidev.engine.loop.behavior_tracker import BehaviorTracker


def _args(path: str = "src/app.py") -> str:
    return json.dumps({"file_path": path})


def test_apply_file_patch_counts_as_successful_edit():
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("apply_file_patch", _args(), had_error=False)

    assert tracker.task_has_edits is True
    assert tracker.files_edited == {"src/app.py"}
    assert tracker.successful_edit_count == 1


def test_failed_workspace_write_does_not_count_as_edit():
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("edit_file", _args(), had_error=True)

    assert tracker.task_has_edits is False
    assert tracker.files_edited == set()
    assert tracker.successful_edit_count == 0


def test_other_effectful_file_tools_use_central_metadata():
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("move_file", _args(), had_error=False)

    assert tracker.task_has_edits is True
