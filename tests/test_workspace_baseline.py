"""Tool-independent workspace change detection."""

from __future__ import annotations

import subprocess
import tracemalloc

from infinidev.engine import workspace_baseline as workspace_baseline_module
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


def test_large_artifact_uses_metadata_without_heap_sized_buffer(tmp_path) -> None:
    artifact = tmp_path / "disk.raw"
    with artifact.open("wb") as handle:
        handle.truncate(128 * 1024 * 1024)

    was_tracing = tracemalloc.is_tracing()
    if not was_tracing:
        tracemalloc.start()
    before = tracemalloc.get_traced_memory()[0]
    tracemalloc.reset_peak()
    try:
        baseline = WorkspaceBaseline.capture(str(tmp_path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        if not was_tracing:
            tracemalloc.stop()

    assert baseline is not None
    state = baseline.files["disk.raw"]
    assert state.text is None
    assert state.digest.startswith("stat-v1:")
    assert peak - before < 8 * 1024 * 1024


def test_generated_build_directories_are_excluded_from_walk(tmp_path) -> None:
    artifact = tmp_path / "target" / "debug" / "artifact.bin"
    artifact.parent.mkdir(parents=True)
    with artifact.open("wb") as handle:
        handle.truncate(32 * 1024 * 1024)

    baseline = WorkspaceBaseline.capture(str(tmp_path))

    assert baseline is not None
    assert "target/debug/artifact.bin" not in baseline.files


def test_git_tracked_file_under_excluded_directory_is_kept(tmp_path) -> None:
    root = _repo(tmp_path)
    tracked = root / "target" / "tracked.rs"
    tracked.parent.mkdir()
    tracked.write_text("pub fn tracked() {}\n")
    _git(root, "add", "-f", "target/tracked.rs")
    _git(root, "commit", "-qm", "track target fixture")

    baseline = WorkspaceBaseline.capture(str(root))

    assert baseline is not None
    assert baseline.files["target/tracked.rs"].text == "pub fn tracked() {}\n"


def test_total_text_retention_is_bounded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(workspace_baseline_module, "_MAX_TOTAL_CAPTURE_BYTES", 5)
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text("ab")

    baseline = WorkspaceBaseline.capture(str(tmp_path))

    assert baseline is not None
    texts = [baseline.files[name].text for name in ("a.txt", "b.txt", "c.txt")]
    retained = sum(len(text.encode()) for text in texts if text is not None)
    assert retained <= 5
    assert texts.count(None) == 1


def test_oversized_metadata_still_detects_change(tmp_path) -> None:
    artifact = tmp_path / "large.bin"
    with artifact.open("wb") as handle:
        handle.truncate(workspace_baseline_module._MAX_HASH_BYTES_PER_FILE + 1)
    tracker = FileChangeTracker(WorkspaceBaseline.capture(str(tmp_path)))

    with artifact.open("r+b") as handle:
        handle.truncate(workspace_baseline_module._MAX_HASH_BYTES_PER_FILE + 2)

    assert tracker.get_all_paths() == [str(artifact)]


def test_record_does_not_rescan_workspace(tmp_path, monkeypatch) -> None:
    root = _repo(tmp_path)
    baseline = WorkspaceBaseline.capture(str(root))
    assert baseline is not None
    tracker = FileChangeTracker(baseline)

    def fail_if_called():
        raise AssertionError("record() must not reconcile the whole workspace")

    monkeypatch.setattr(baseline, "current_states", fail_if_called)

    diff = tracker.record(str(root / "tracked.py"), "value = 1\n", "value = 2\n")

    assert diff is not None
    assert "+value = 2" in diff


def test_deactivate_reconciles_once_and_getters_reuse_snapshot(
    tmp_path,
    monkeypatch,
) -> None:
    root = _repo(tmp_path)
    baseline = WorkspaceBaseline.capture(str(root))
    assert baseline is not None
    tracker = FileChangeTracker(baseline)
    (root / "tracked.py").write_text("value = 2\n")

    calls = 0
    current_states = baseline.current_states

    def counted_current_states():
        nonlocal calls
        calls += 1
        return current_states()

    monkeypatch.setattr(baseline, "current_states", counted_current_states)

    tracker.deactivate()
    assert tracker.get_all_paths() == [str(root / "tracked.py")]
    assert tracker.get_diff(str(root / "tracked.py")) is not None
    assert calls == 1
