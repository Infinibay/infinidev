"""Tool-independent workspace change detection."""

from __future__ import annotations

import subprocess

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.workspace_baseline import WorkspaceBaseline


def _git(root, *args: str) -> None:
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "tracked.py").write_text("value = 1\n")
    _git(tmp_path, "add", "tracked.py")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def test_detects_shell_style_edit_without_record_call(tmp_path) -> None:
    root = _repo(tmp_path)
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))

    (root / "tracked.py").write_text("value = 2\n")

    assert tracker.get_all_paths() == [str(root / "tracked.py")]
    assert "value = 2" in (tracker.get_diff(str(root / "tracked.py")) or "")


def test_preexisting_dirty_file_is_not_attributed_when_unchanged(tmp_path) -> None:
    root = _repo(tmp_path)
    (root / "tracked.py").write_text("user change\n")
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))

    assert tracker.get_all_paths() == []


def test_detects_change_relative_to_preexisting_dirty_content(tmp_path) -> None:
    root = _repo(tmp_path)
    (root / "tracked.py").write_text("user change\n")
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))

    (root / "tracked.py").write_text("agent change\n")

    diff = tracker.get_diff(str(root / "tracked.py")) or ""
    assert "-user change" in diff
    assert "+agent change" in diff


def test_detects_new_and_deleted_files(tmp_path) -> None:
    root = _repo(tmp_path)
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))

    (root / "new.py").write_text("new = True\n")
    (root / "tracked.py").unlink()

    assert set(tracker.get_all_paths()) == {
        str(root / "new.py"),
        str(root / "tracked.py"),
    }


def test_edit_then_restore_is_not_a_final_change(tmp_path) -> None:
    root = _repo(tmp_path)
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))
    path = root / "tracked.py"
    tracker.record(str(path), "value = 1\n", "value = 2\n")

    path.write_text("value = 1\n")

    assert tracker.get_all_paths() == []


def test_git_baseline_detects_ignored_generated_file(tmp_path) -> None:
    root = _repo(tmp_path)
    (root / ".gitignore").write_text("generated/\n")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "ignore generated")
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(root)))

    generated = root / "generated" / "artifact.txt"
    generated.parent.mkdir()
    generated.write_text("created by task\n")

    assert tracker.get_all_paths() == [str(generated)]
