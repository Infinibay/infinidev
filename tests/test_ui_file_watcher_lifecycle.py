"""Lifecycle integration tests for the TUI file explorer watcher."""

from __future__ import annotations

from infinidev.ui.handlers.files import FileManager


class _App:
    explorer_visible = False

    def focus_chat(self) -> None:
        pass

    def invalidate(self) -> None:
        pass


def _file_manager() -> FileManager:
    manager = FileManager.__new__(FileManager)
    manager._app = _App()
    manager._tree_control = None
    manager._file_picker = None
    manager._file_watcher = None
    return manager


def test_start_file_watcher_clears_clean_failure_for_retry(monkeypatch) -> None:
    created: list[_Watcher] = []

    class _Watcher:
        def __init__(self, **_kwargs) -> None:
            created.append(self)

        def start(self) -> bool:
            return False

    monkeypatch.setattr("infinidev.cli.file_watcher.WATCHFILES_AVAILABLE", True)
    monkeypatch.setattr("infinidev.cli.file_watcher.FileWatcher", _Watcher)
    manager = _file_manager()

    manager._start_file_watcher()
    manager._start_file_watcher()

    assert manager._file_watcher is None
    assert len(created) == 2


def test_start_file_watcher_preserves_partial_worker_when_cleanup_fails(
    monkeypatch,
) -> None:
    class _Watcher:
        def __init__(self, **_kwargs) -> None:
            self.stop_calls = 0

        def start(self) -> bool:
            raise RuntimeError("start failed after launching worker")

        def stop(self) -> None:
            self.stop_calls += 1
            raise TimeoutError("worker still alive")

    monkeypatch.setattr("infinidev.cli.file_watcher.WATCHFILES_AVAILABLE", True)
    monkeypatch.setattr("infinidev.cli.file_watcher.FileWatcher", _Watcher)
    manager = _file_manager()

    manager._start_file_watcher()

    watcher = manager._file_watcher
    assert isinstance(watcher, _Watcher)
    assert watcher.stop_calls == 1


def test_reopening_initialized_explorer_restarts_exited_watcher(monkeypatch) -> None:
    class _ExitedWatcher:
        def __init__(self) -> None:
            self.stop_calls = 0

        def is_running(self) -> bool:
            return False

        def stop(self) -> None:
            self.stop_calls += 1

    manager = _file_manager()
    manager._tree_control = object()
    manager._tree_window = object()
    exited = _ExitedWatcher()
    manager._file_watcher = exited
    replacements: list[_ReplacementWatcher] = []

    class _ReplacementWatcher:
        def __init__(self, **_kwargs) -> None:
            replacements.append(self)

        def start(self) -> bool:
            return True

    monkeypatch.setattr("infinidev.cli.file_watcher.WATCHFILES_AVAILABLE", True)
    monkeypatch.setattr("infinidev.cli.file_watcher.FileWatcher", _ReplacementWatcher)

    manager.toggle_explorer()

    assert exited.stop_calls == 1
    assert len(replacements) == 1
    assert manager._file_watcher is replacements[0]


def test_start_file_watcher_retains_unresponsive_exited_watcher() -> None:
    class _UnresponsiveWatcher:
        def is_running(self) -> bool:
            return False

        def stop(self) -> None:
            raise TimeoutError("worker still alive")

    manager = _file_manager()
    watcher = _UnresponsiveWatcher()
    manager._file_watcher = watcher

    manager._start_file_watcher()

    assert manager._file_watcher is watcher
