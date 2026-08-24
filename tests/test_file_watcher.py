"""Focused lifecycle coverage for the CLI file watcher."""

from __future__ import annotations

import threading

import pytest

from infinidev.cli.file_watcher import FileWatcher, _InfinidevWatchFilter


def test_watch_filter_ignores_state_dirs_with_windows_separators() -> None:
    watch_filter = _InfinidevWatchFilter()

    assert watch_filter(object(), r"C:\work\project\.infinidev\history\turn.json") is False
    assert watch_filter(object(), r"C:\work\project\.git\index") is False


def test_watch_filter_honors_case_insensitive_windows_paths() -> None:
    watch_filter = _InfinidevWatchFilter()

    assert watch_filter(object(), r"C:\work\project\.INFINIDEV\history\turn.json") is False
    assert watch_filter(object(), r"C:\work\project\.GIT\index") is False
    assert watch_filter(object(), r"C:\work\project\cache.DB-WAL") is False


class _ControllableThread:
    def __init__(self, *, stops_on_join: bool) -> None:
        self.alive = True
        self.stops_on_join = stops_on_join
        self.join_timeouts: list[float | None] = []

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        self.join_timeouts.append(timeout)
        if self.stops_on_join:
            self.alive = False


def test_no_visibility_callback_refreshes_workspace_changes(tmp_path) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)

    assert watcher._should_refresh(str(tmp_path / "src" / "changed.py")) is True
    assert watcher._should_refresh(str(tmp_path.parent / "outside.py")) is False


def test_visibility_callback_still_gates_workspace_changes(tmp_path) -> None:
    visible = tmp_path / "src"
    watcher = FileWatcher(
        str(tmp_path),
        lambda _path: None,
        visible_paths_callback=lambda: {str(visible)},
    )

    assert watcher._should_refresh(str(visible / "changed.py")) is True
    assert watcher._should_refresh(str(tmp_path / "hidden" / "changed.py")) is False


def test_stop_raises_when_watcher_thread_outlives_join(tmp_path) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    thread = _ControllableThread(stops_on_join=False)
    watcher._running = True
    watcher._watch_thread = thread  # type: ignore[assignment]

    with pytest.raises(TimeoutError, match="did not stop within 2 seconds"):
        watcher.stop()

    assert watcher._stop_event.is_set()
    assert not watcher.is_running()
    assert thread.join_timeouts == [2.0]


def test_stop_is_idempotent_after_watcher_thread_exits(tmp_path) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    thread = _ControllableThread(stops_on_join=True)
    watcher._running = True
    watcher._watch_thread = thread  # type: ignore[assignment]

    watcher.stop()
    watcher.stop()

    assert watcher._stop_event.is_set()
    assert not watcher.is_running()
    assert thread.join_timeouts == [2.0]


def test_start_refuses_restart_while_previous_thread_is_alive(
    tmp_path, monkeypatch
) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    previous_thread = _ControllableThread(stops_on_join=False)
    watcher._watch_thread = previous_thread  # type: ignore[assignment]
    watcher._stop_event.set()

    created_threads: list[object] = []
    monkeypatch.setattr(
        "infinidev.cli.file_watcher.Thread",
        lambda **_kwargs: created_threads.append(object()),
    )

    assert watcher.start() is False
    assert watcher._stop_event.is_set()
    assert watcher._watch_thread is previous_thread
    assert created_threads == []


def test_start_failure_rolls_back_state_and_allows_retry(tmp_path, monkeypatch) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)

    class _FailingThread:
        def __init__(self, **_kwargs) -> None:
            pass

        def start(self) -> None:
            raise RuntimeError("cannot start thread")

    monkeypatch.setattr("infinidev.cli.file_watcher.Thread", _FailingThread)

    with pytest.raises(RuntimeError, match="cannot start thread"):
        watcher.start()

    assert not watcher.is_running()
    assert watcher._watch_thread is None
    assert watcher._stop_event.is_set()

    class _StartedThread:
        def __init__(self, **_kwargs) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    replacement = _StartedThread()
    monkeypatch.setattr(
        "infinidev.cli.file_watcher.Thread", lambda **_kwargs: replacement
    )

    assert watcher.start() is True
    assert watcher.is_running()
    assert watcher._watch_thread is replacement
    assert replacement.started


def test_start_can_restart_after_previous_thread_exits(tmp_path, monkeypatch) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    previous_thread = _ControllableThread(stops_on_join=False)
    previous_thread.alive = False
    watcher._watch_thread = previous_thread  # type: ignore[assignment]
    watcher._stop_event.set()

    class _StartedThread:
        def __init__(self, **_kwargs) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    new_thread = _StartedThread()
    monkeypatch.setattr(
        "infinidev.cli.file_watcher.Thread", lambda **_kwargs: new_thread
    )

    assert watcher.start() is True
    assert not watcher._stop_event.is_set()
    assert watcher._watch_thread is new_thread
    assert new_thread.started


def test_concurrent_starts_create_exactly_one_watcher_thread(
    tmp_path, monkeypatch
) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    callers_ready = threading.Barrier(3)
    created_threads: list[_StartedThread] = []

    class _StartedThread:
        def __init__(self, **_kwargs) -> None:
            self.alive = False
            created_threads.append(self)

        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

    monkeypatch.setattr("infinidev.cli.file_watcher.Thread", _StartedThread)
    results: list[bool] = []

    def start_watcher() -> None:
        callers_ready.wait()
        results.append(watcher.start())

    callers = [threading.Thread(target=start_watcher) for _ in range(2)]
    for caller in callers:
        caller.start()
    callers_ready.wait()
    for caller in callers:
        caller.join(timeout=1.0)

    assert all(not caller.is_alive() for caller in callers)
    assert results == [True, True]
    assert len(created_threads) == 1
    assert watcher._watch_thread is created_threads[0]


def test_worker_reads_running_state_through_lifecycle_lock(tmp_path, monkeypatch) -> None:
    indexed: list[str] = []
    watch_entered = threading.Event()
    watcher = FileWatcher(
        str(tmp_path),
        lambda _path: None,
        index_callback=indexed.append,
    )
    changed_path = tmp_path / "changed.py"

    def changes(*_args, **_kwargs):
        watch_entered.set()
        yield {(object(), str(changed_path))}

    monkeypatch.setattr("infinidev.cli.file_watcher.watch", changes)
    watcher._running = True
    worker = threading.Thread(target=watcher._run_watcher)
    watcher._watch_thread = worker

    with watcher._state_lock:
        worker.start()
        assert watch_entered.wait(timeout=1.0)
        worker.join(timeout=0.05)
        assert worker.is_alive()
        assert indexed == []

    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert indexed == [str(changed_path.resolve())]
    assert not watcher.is_running()


def test_unexpected_worker_exit_allows_restart_without_clobbering_replacement(
    tmp_path, monkeypatch
) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)

    class _ExitedThread:
        def is_alive(self) -> bool:
            return False

    exited_thread = _ExitedThread()
    watcher._running = True
    watcher._watch_thread = exited_thread  # type: ignore[assignment]
    monkeypatch.setattr(
        "infinidev.cli.file_watcher.current_thread", lambda: exited_thread
    )
    monkeypatch.setattr("infinidev.cli.file_watcher.watch", lambda *_args, **_kwargs: ())

    watcher._run_watcher()

    assert not watcher.is_running()

    class _ReplacementThread:
        def __init__(self, **_kwargs) -> None:
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

    replacement = _ReplacementThread()
    monkeypatch.setattr(
        "infinidev.cli.file_watcher.Thread", lambda **_kwargs: replacement
    )

    assert watcher.start() is True
    assert watcher.is_running()
    assert watcher._watch_thread is replacement

    watcher._publish_worker_exit(exited_thread)  # type: ignore[arg-type]

    assert watcher.is_running()
    assert watcher._watch_thread is replacement


def test_callback_can_stop_and_restart_real_watcher(tmp_path, monkeypatch) -> None:
    callback_returns = [threading.Event(), threading.Event()]
    changed_paths = [tmp_path / "first.py", tmp_path / "restarted.py"]
    watch_calls = 0
    watcher: FileWatcher

    def changes(*_args, **_kwargs):
        nonlocal watch_calls
        changed_path = changed_paths[watch_calls]
        watch_calls += 1
        yield {(object(), str(changed_path))}

    callback_count = 0

    def stop_from_callback(_path: str) -> None:
        nonlocal callback_count
        callback_index = callback_count
        callback_count += 1
        watcher.stop()
        callback_returns[callback_index].set()

    monkeypatch.setattr("infinidev.cli.file_watcher.watch", changes)
    watcher = FileWatcher(str(tmp_path), stop_from_callback)

    for cycle in range(2):
        assert watcher.start() is True
        worker = watcher._watch_thread
        assert worker is not None
        assert callback_returns[cycle].wait(5.0), "self-stop did not return"
        worker.join(timeout=5.0)

        assert not worker.is_alive()
        assert not watcher.is_running()
        assert watcher._stop_event.is_set()

    assert callback_count == 2
    assert watch_calls == 2


def test_start_waits_for_stop_before_clearing_stop_event(
    tmp_path, monkeypatch
) -> None:
    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    join_started = threading.Event()
    allow_join = threading.Event()

    class _StoppingThread:
        def __init__(self) -> None:
            self.alive = True

        def is_alive(self) -> bool:
            return self.alive

        def join(self, timeout: float | None = None) -> None:
            join_started.set()
            assert timeout == 2.0
            assert allow_join.wait(timeout=1.0)
            self.alive = False

    class _ReplacementThread:
        def __init__(self, **_kwargs) -> None:
            self.alive = False

        def is_alive(self) -> bool:
            return self.alive

        def start(self) -> None:
            self.alive = True

    watcher._running = True
    watcher._watch_thread = _StoppingThread()  # type: ignore[assignment]
    replacement_threads: list[_ReplacementThread] = []

    def make_replacement(**kwargs) -> _ReplacementThread:
        thread = _ReplacementThread(**kwargs)
        replacement_threads.append(thread)
        return thread

    monkeypatch.setattr("infinidev.cli.file_watcher.Thread", make_replacement)
    stop_thread = threading.Thread(target=watcher.stop)
    stop_thread.start()
    assert join_started.wait(timeout=1.0)

    start_results: list[bool] = []
    start_thread = threading.Thread(target=lambda: start_results.append(watcher.start()))
    start_thread.start()

    assert watcher._stop_event.is_set()
    assert replacement_threads == []
    assert start_thread.is_alive()

    allow_join.set()
    stop_thread.join(timeout=1.0)
    start_thread.join(timeout=1.0)

    assert not stop_thread.is_alive()
    assert not start_thread.is_alive()
    assert start_results == [True]
    assert len(replacement_threads) == 1
    assert watcher._watch_thread is replacement_threads[0]
    assert not watcher._stop_event.is_set()
