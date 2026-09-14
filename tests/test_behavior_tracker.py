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


def test_creating_a_file_is_not_a_blind_edit():
    """A file that does not exist cannot have been read first.

    The baseline logged "You are editing PLAN.md without reading it first" for
    a `create_file` that produced the whole deliverable.
    """
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("create_file", _args("PLAN.md"), had_error=False)

    warnings = tracker.drain_feedback()
    assert "without reading it first" not in warnings
    assert tracker.task_has_edits is True


def test_editing_an_unread_existing_file_still_warns():
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("edit_file", _args("src/app.py"), had_error=False)

    assert "without reading it first" in tracker.drain_feedback()


def test_editing_a_read_file_is_praised_not_warned():
    tracker = BehaviorTracker(set())

    tracker.on_tool_call("read_file", _args("src/app.py"), had_error=False)
    tracker.on_tool_call("edit_file", _args("src/app.py"), had_error=False)

    feedback = tracker.drain_feedback()
    assert "without reading it first" not in feedback
    assert "before editing" in feedback
