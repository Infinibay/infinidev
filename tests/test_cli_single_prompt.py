"""Regression coverage for the non-interactive single-prompt CLI path."""

from __future__ import annotations

import builtins
import logging
import os
import sys

import pytest

from infinidev.engine import orchestration
from infinidev.flows.event_listeners import event_bus


@pytest.fixture
def cli_main():
    """Import the CLI without leaking its process-wide logging setup."""
    root = logging.getLogger()
    infinidev_logger = logging.getLogger("infinidev")
    root_level = root.level
    root_handlers = list(root.handlers)
    infinidev_level = infinidev_logger.level
    try:
        from infinidev.cli import main as module

        yield module
    finally:
        root.handlers[:] = root_handlers
        root.setLevel(root_level)
        infinidev_logger.setLevel(infinidev_level)


def test_classic_bootstrap_optional_failures_are_logged_and_non_fatal(
    cli_main, monkeypatch, caplog
):
    """Schema warming and watcher failures do not prevent classic-mode bootstrap."""
    from infinidev.cli import file_watcher, index_queue, initial_index
    from infinidev.code_intel import background_indexer
    from infinidev.engine.behavior import hook as behavior_hook
    from infinidev.engine.hooks import ui_hooks
    from infinidev.engine import tool_dispatch
    from infinidev import tools

    class FakeQueue:
        def __init__(self, project_id):
            self.project_id = project_id

        def start(self):
            return None

    watcher_calls = []

    class FailingWatcher:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            watcher_calls.append("start")
            raise RuntimeError("watcher unavailable")

        def stop(self):
            watcher_calls.append("stop")

    monkeypatch.setattr(cli_main, "init_db", lambda: None)
    monkeypatch.setattr(ui_hooks, "register_ui_hooks", lambda: None)
    monkeypatch.setattr(behavior_hook, "register_behavior_hooks", lambda: None)
    monkeypatch.setattr(initial_index, "run_initial_index", lambda **_kwargs: None)
    monkeypatch.setattr(index_queue, "IndexQueue", FakeQueue)
    monkeypatch.setattr(background_indexer, "acquire_global_queue", lambda _queue: True)
    monkeypatch.setattr(tools, "get_tools_for_role", lambda *_args, **_kwargs: [])

    def fail_to_warm_schemas(*_args, **_kwargs):
        raise RuntimeError("schema warming unavailable")

    monkeypatch.setattr(tool_dispatch, "build_tool_schemas", fail_to_warm_schemas)
    monkeypatch.setattr(file_watcher, "WATCHFILES_AVAILABLE", True)
    monkeypatch.setattr(file_watcher, "FileWatcher", FailingWatcher)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        queue = cli_main._bootstrap_single_prompt_runtime()

    assert queue._file_watcher is None
    records = {
        record.getMessage(): record
        for record in caplog.records
        if record.getMessage() in {
            "Failed to warm tool schemas",
            "Failed to start classic-mode file watcher",
        }
    }
    assert set(records) == {
        "Failed to warm tool schemas",
        "Failed to start classic-mode file watcher",
    }
    for record in records.values():
        assert record.levelno == logging.DEBUG
        assert record.exc_info is not None
        assert record.exc_info[0] is RuntimeError
    assert watcher_calls == ["start", "stop"]


@pytest.mark.parametrize("failure_stage", ["start", "register"])
def test_classic_bootstrap_stops_queue_before_failed_ownership_transfer(
    cli_main, monkeypatch, failure_stage
):
    """A queue not yet registered globally remains bootstrap's responsibility."""
    from infinidev import tools
    from infinidev.cli import index_queue, initial_index
    from infinidev.code_intel import background_indexer
    from infinidev.engine.behavior import hook as behavior_hook
    from infinidev.engine.hooks import ui_hooks
    from infinidev.engine import tool_dispatch

    calls = []

    class TrackingQueue:
        def __init__(self, project_id):
            self.project_id = project_id

        def start(self):
            calls.append("start")
            if failure_stage == "start":
                raise RuntimeError("queue start unavailable")

        def stop(self):
            calls.append("stop")

    def acquire_global_queue(_queue):
        calls.append("register")
        raise RuntimeError("queue registration unavailable")

    monkeypatch.setattr(cli_main, "init_db", lambda: None)
    monkeypatch.setattr(ui_hooks, "register_ui_hooks", lambda: None)
    monkeypatch.setattr(behavior_hook, "register_behavior_hooks", lambda: None)
    monkeypatch.setattr(initial_index, "run_initial_index", lambda **_kwargs: None)
    monkeypatch.setattr(index_queue, "IndexQueue", TrackingQueue)
    monkeypatch.setattr(background_indexer, "acquire_global_queue", acquire_global_queue)
    monkeypatch.setattr(tools, "get_tools_for_role", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(tool_dispatch, "build_tool_schemas", lambda *_args, **_kwargs: [])

    expected_error = (
        "queue start unavailable"
        if failure_stage == "start"
        else "queue registration unavailable"
    )
    with pytest.raises(RuntimeError, match=expected_error):
        cli_main._bootstrap_single_prompt_runtime()

    expected_calls = ["start"]
    if failure_stage == "register":
        expected_calls.append("register")
    assert calls == [*expected_calls, "stop"]


def test_classic_bootstrap_loser_stops_its_queue(cli_main, monkeypatch):
    """A concurrent bootstrap loser must stop rather than orphan its worker."""
    from infinidev import tools
    from infinidev.cli import index_queue, initial_index
    from infinidev.code_intel import background_indexer
    from infinidev.engine.behavior import hook as behavior_hook
    from infinidev.engine.hooks import ui_hooks

    calls = []

    class TrackingQueue:
        def __init__(self, project_id):
            self.project_id = project_id

        def start(self):
            calls.append("start")

        def stop(self):
            calls.append("stop")

    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: None)
    monkeypatch.setattr(background_indexer, "acquire_global_queue", lambda _queue: False)
    monkeypatch.setattr(cli_main, "init_db", lambda: calls.append("init"))
    monkeypatch.setattr(ui_hooks, "register_ui_hooks", lambda: None)
    monkeypatch.setattr(behavior_hook, "register_behavior_hooks", lambda: None)
    monkeypatch.setattr(initial_index, "run_initial_index", lambda **_kwargs: None)
    monkeypatch.setattr(index_queue, "IndexQueue", TrackingQueue)
    monkeypatch.setattr(
        tools,
        "get_tools_for_role",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("losing bootstrap must return before optional setup")
        ),
    )

    cli_main._bootstrap_single_prompt_runtime()

    assert calls == ["init", "start", "stop"]


def test_classic_bootstrap_preserves_existing_running_queue(cli_main, monkeypatch):
    """Repeated bootstrap must not orphan an already-owned worker and watcher."""
    from infinidev.cli import index_queue
    from infinidev.code_intel import background_indexer

    class ExistingQueue:
        def is_running(self):
            return True

    existing_queue = ExistingQueue()
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: existing_queue)

    def fail_if_created(*_args, **_kwargs):
        raise AssertionError("a replacement queue must not be created")

    monkeypatch.setattr(index_queue, "IndexQueue", fail_if_created)
    monkeypatch.setattr(
        cli_main,
        "init_db",
        lambda: (_ for _ in ()).throw(AssertionError("bootstrap must return early")),
    )

    cli_main._bootstrap_single_prompt_runtime()

    assert background_indexer.get_global_queue() is existing_queue


def test_classic_event_bus_import_failure_is_logged_and_non_fatal(
    cli_main, monkeypatch, caplog
):
    """A missing optional event bus remains observable but harmless."""
    monkeypatch.setattr(cli_main, "_log_file_path", "debug.log")
    original_import = builtins.__import__

    def fail_event_bus_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "infinidev.flows.event_listeners" and "event_bus" in fromlist:
            raise RuntimeError("event bus unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fail_event_bus_import)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._install_classic_event_bridge()

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to import event bus for classic-mode bridge"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_classic_event_bridge_failure_is_logged_and_non_fatal(
    cli_main, monkeypatch, caplog
):
    """An optional event bridge failure remains observable but harmless."""
    monkeypatch.setattr(cli_main, "_log_file_path", "debug.log")

    def fail_to_subscribe(_callback):
        raise RuntimeError("subscription unavailable")

    monkeypatch.setattr(event_bus, "subscribe", fail_to_subscribe)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._install_classic_event_bridge()

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to install classic-mode event bridge"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_classic_event_bridge_registration_is_idempotent(cli_main, monkeypatch):
    """Repeated installation must not duplicate process-wide event delivery."""
    callbacks = []
    monkeypatch.setattr(cli_main, "_log_file_path", "debug.log")
    monkeypatch.setattr(cli_main, "_classic_event_bridge_bus", None)
    monkeypatch.setattr(cli_main, "_classic_event_bridge_callback", None)
    monkeypatch.setattr(event_bus, "subscribe", callbacks.append)

    cli_main._install_classic_event_bridge()
    cli_main._install_classic_event_bridge()

    assert len(callbacks) == 1
    assert cli_main._classic_event_bridge_bus is event_bus
    assert cli_main._classic_event_bridge_callback is callbacks[0]


def test_classic_event_bridge_retries_after_subscription_failure(
    cli_main, monkeypatch, caplog
):
    """A failed subscription must not mark the bridge as installed."""
    callbacks = []
    attempts = 0
    monkeypatch.setattr(cli_main, "_log_file_path", "debug.log")
    monkeypatch.setattr(cli_main, "_classic_event_bridge_bus", None)
    monkeypatch.setattr(cli_main, "_classic_event_bridge_callback", None)

    def subscribe(callback):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("subscription unavailable")
        callbacks.append(callback)

    monkeypatch.setattr(event_bus, "subscribe", subscribe)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._install_classic_event_bridge()
        cli_main._install_classic_event_bridge()

    assert attempts == 2
    assert len(callbacks) == 1
    assert cli_main._classic_event_bridge_callback is callbacks[0]
    assert any(
        record.getMessage() == "Failed to install classic-mode event bridge"
        for record in caplog.records
    )


def test_classic_event_format_failure_is_logged_and_non_fatal(
    cli_main, monkeypatch, caplog
):
    """A malformed event must not break later event-bus delivery."""
    callbacks = []
    monkeypatch.setattr(cli_main, "_log_file_path", "debug.log")
    monkeypatch.setattr(event_bus, "subscribe", callbacks.append)
    monkeypatch.setattr(
        cli_main,
        "_format_event",
        lambda _event_type, _data: (_ for _ in ()).throw(
            RuntimeError("malformed event")
        ),
    )

    cli_main._install_classic_event_bridge()

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        callbacks[0]("loop_log", 1, "agent", {"message": "hello"})

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to format classic-mode event"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_fast_exit_logs_queue_stop_failure_and_finishes_cleanup(
    cli_main, monkeypatch, caplog
):
    """Queue cleanup failure must not prevent flushing or the forced exit."""
    from infinidev.code_intel import background_indexer

    class FailingQueue:
        def stop(self):
            raise RuntimeError("queue unavailable")

    class FakeStream:
        def __init__(self, name):
            self.name = name

        def flush(self):
            flushed.append(self.name)

    flushed = []
    exit_codes = []
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: FailingQueue())
    monkeypatch.setattr(sys, "stdout", FakeStream("stdout"))
    monkeypatch.setattr(sys, "stderr", FakeStream("stderr"))
    monkeypatch.setattr(os, "_exit", exit_codes.append)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._fast_exit_workaround()

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to stop background index queue during shutdown"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert flushed == ["stdout", "stderr"]
    assert exit_codes == [0]


def test_fast_exit_stream_flush_failures_do_not_skip_later_cleanup(
    cli_main, monkeypatch, caplog
):
    """A broken stdout must not prevent stderr flushing or the forced exit."""
    from infinidev.code_intel import background_indexer

    class FailingStream:
        def flush(self):
            raise RuntimeError("stream unavailable")

    class TrackingStream:
        def flush(self):
            flushed.append("stderr")

    exit_codes = []
    flushed = []
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: None)
    monkeypatch.setattr(sys, "stdout", FailingStream())
    monkeypatch.setattr(sys, "stderr", TrackingStream())
    monkeypatch.setattr(os, "_exit", exit_codes.append)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._fast_exit_workaround()

    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to flush stdout during shutdown"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert flushed == ["stderr"]
    assert exit_codes == [0]


def test_final_render_failure_is_logged_and_falls_back_to_plain_text(
    cli_main, monkeypatch, caplog, capsys
):
    """A Rich failure remains observable while the result is still shown."""
    from rich.console import Console

    def fail_to_render(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    monkeypatch.setattr(Console, "print", fail_to_render)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._render_final("final answer")

    assert capsys.readouterr().out == "final answer\n"
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to render final result with Rich"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_final_render_survives_terminal_detection_failure(
    cli_main, monkeypatch, caplog
):
    """A broken isatty implementation must not force the plain-text fallback."""
    from rich.console import Console
    from rich.markdown import Markdown

    class BrokenTerminalStream:
        def isatty(self):
            raise OSError("terminal detection unavailable")

    rendered = []
    monkeypatch.setattr(sys, "stdout", BrokenTerminalStream())
    monkeypatch.setattr(Console, "print", lambda _console, value: rendered.append(value))

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._render_final("**final answer**")

    assert len(rendered) == 1
    assert isinstance(rendered[0], Markdown)
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to detect whether stdout is a terminal"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is OSError
    assert not any(
        record.getMessage() == "Failed to render final result with Rich"
        for record in caplog.records
    )


def test_final_plain_text_failure_is_logged_and_non_fatal(
    cli_main, monkeypatch, caplog
):
    """Broken final-output streams remain observable without failing the run."""
    from rich.console import Console

    def fail_to_render(*_args, **_kwargs):
        raise RuntimeError("renderer unavailable")

    def fail_to_echo(*_args, **_kwargs):
        raise OSError("stdout unavailable")

    monkeypatch.setattr(Console, "print", fail_to_render)
    monkeypatch.setattr(cli_main.click, "echo", fail_to_echo)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._render_final("final answer")

    records = {
        record.getMessage(): record
        for record in caplog.records
        if record.getMessage() in {
            "Failed to render final result with Rich",
            "Failed to render final result as plain text",
        }
    }
    assert set(records) == {
        "Failed to render final result with Rich",
        "Failed to render final result as plain text",
    }
    assert records["Failed to render final result with Rich"].exc_info[0] is RuntimeError
    assert records["Failed to render final result as plain text"].exc_info[0] is OSError


def test_interactive_bootstrap_failure_cleans_started_queue(cli_main, monkeypatch):
    """A partial bootstrap must not leak its process-wide indexing queue."""
    from infinidev.code_intel import background_indexer
    from infinidev.tools import permission

    cleanup_calls = []
    queue_state = {"queue": None}

    class TrackingQueue:
        def stop(self):
            cleanup_calls.append("queue")

    def partial_bootstrap():
        queue_state["queue"] = TrackingQueue()
        raise RuntimeError("bootstrap unavailable")

    def release_global_queue(queue):
        if queue_state["queue"] is not queue:
            return False
        queue_state["queue"] = None
        cleanup_calls.append("clear")
        return True

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", partial_bootstrap)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(
        background_indexer, "get_global_queue", lambda: queue_state["queue"]
    )
    monkeypatch.setattr(
        background_indexer, "release_global_queue", release_global_queue
    )
    monkeypatch.setattr(
        permission,
        "set_permission_handler",
        lambda handler: cleanup_calls.append("permission") if handler is None else None,
    )

    with pytest.raises(RuntimeError, match="bootstrap unavailable"):
        cli_main._run_main(False, True, None, False, False)

    assert cleanup_calls == ["permission", "ken", "queue", "clear"]
    assert queue_state["queue"] is None


def test_interactive_setup_failure_runs_full_cleanup(cli_main, monkeypatch):
    """A setup error after subscription must release every classic resource."""
    from infinidev.cli import classic_renderer
    from infinidev.code_intel import background_indexer
    from infinidev.tools import permission

    cleanup_calls = []

    class TrackingRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            cleanup_calls.append("subscribe")

        def unsubscribe(self):
            cleanup_calls.append("renderer")

    class TrackingQueue:
        def stop(self):
            cleanup_calls.append("queue")

    def fail_prompt_session(**_kwargs):
        raise RuntimeError("prompt session unavailable")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "PromptSession", fail_prompt_session)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", TrackingRenderer)
    monkeypatch.setattr(
        background_indexer, "get_global_queue", lambda: TrackingQueue()
    )
    monkeypatch.setattr(
        permission,
        "set_permission_handler",
        lambda handler: cleanup_calls.append("permission") if handler is None else None,
    )

    with pytest.raises(RuntimeError, match="prompt session unavailable"):
        cli_main._run_main(False, True, None, False, False)

    assert cleanup_calls == ["subscribe", "permission", "renderer", "ken", "queue"]


def test_interactive_queue_lookup_failure_preserves_original_error(
    cli_main, monkeypatch, caplog
):
    """Queue discovery failure during teardown must not mask a setup error."""
    from infinidev.cli import classic_renderer
    from infinidev.code_intel import background_indexer
    from infinidev.tools import permission

    cleanup_calls = []

    class TrackingRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            cleanup_calls.append("subscribe")

        def unsubscribe(self):
            cleanup_calls.append("renderer")

    def fail_prompt_session(**_kwargs):
        raise RuntimeError("prompt session unavailable")

    def fail_queue_lookup():
        raise RuntimeError("queue lookup unavailable")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "PromptSession", fail_prompt_session)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", TrackingRenderer)
    monkeypatch.setattr(background_indexer, "get_global_queue", fail_queue_lookup)
    monkeypatch.setattr(
        permission,
        "set_permission_handler",
        lambda handler: cleanup_calls.append("permission") if handler is None else None,
    )

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        with pytest.raises(RuntimeError, match="prompt session unavailable"):
            cli_main._run_main(False, True, None, False, False)

    assert cleanup_calls == ["subscribe", "permission", "renderer", "ken"]
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to find background index queue"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_interactive_cleanup_resets_permissions_and_continues_after_failure(
    cli_main, monkeypatch, caplog
):
    """Permission reset failure must not skip later interactive cleanup."""
    from infinidev.code_intel import background_indexer

    cleanup_calls = []

    class TrackingRenderer:
        def unsubscribe(self):
            cleanup_calls.append("renderer")

    class TrackingWatcher:
        def stop(self):
            cleanup_calls.append("watcher")

    class TrackingQueue:
        _file_watcher = TrackingWatcher()

        def stop(self):
            cleanup_calls.append("queue")

    def fail_permission_reset(handler):
        assert handler is None
        cleanup_calls.append("permission")
        raise RuntimeError("permission cleanup unavailable")

    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(
        background_indexer, "get_global_queue", lambda: TrackingQueue()
    )
    monkeypatch.setattr(
        background_indexer,
        "release_global_queue",
        lambda _queue: cleanup_calls.append("clear") or True,
    )

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._cleanup_classic_runtime(TrackingRenderer(), fail_permission_reset)

    assert cleanup_calls == [
        "permission",
        "renderer",
        "ken",
        "watcher",
        "queue",
        "clear",
    ]
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to reset classic permission handler"
    )
    assert record.levelno == logging.DEBUG
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_interactive_cleanup_retains_queue_when_watcher_stop_fails(
    cli_main, monkeypatch, caplog
):
    """A watcher with uncertain shutdown must retain its queue for cleanup retry."""
    from infinidev.code_intel import background_indexer

    cleanup_calls = []

    class FailingWatcher:
        def stop(self):
            cleanup_calls.append("watcher")
            raise RuntimeError("watcher thread did not stop")

    class TrackingQueue:
        _file_watcher = FailingWatcher()

        def stop(self):
            cleanup_calls.append("queue")

    queue = TrackingQueue()
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: queue)
    monkeypatch.setattr(
        background_indexer,
        "release_global_queue",
        lambda value: cleanup_calls.append(("release", value)),
    )
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._cleanup_classic_runtime(None, lambda _handler: None)

    assert cleanup_calls == ["watcher"]
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to stop classic-mode file watcher"
    )
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_interactive_cleanup_retains_queue_on_real_watcher_join_timeout(
    cli_main, monkeypatch, caplog, tmp_path
):
    """A FileWatcher join timeout must keep its dependent queue registered."""
    from infinidev.cli.file_watcher import FileWatcher
    from infinidev.code_intel import background_indexer

    class StuckThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            assert timeout == 2.0

    watcher = FileWatcher(str(tmp_path), lambda _path: None)
    watcher._running = True
    watcher._watch_thread = StuckThread()

    class TrackingQueue:
        _file_watcher = watcher

        def stop(self):
            pytest.fail("queue must remain running while its watcher may be alive")

    queue = TrackingQueue()
    releases = []
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: queue)
    monkeypatch.setattr(background_indexer, "release_global_queue", releases.append)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._cleanup_classic_runtime(None, lambda _handler: None)

    assert releases == []
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to stop classic-mode file watcher"
    )
    assert record.exc_info is not None
    assert record.exc_info[0] is TimeoutError


def test_interactive_cleanup_retains_queue_when_stop_fails(
    cli_main, monkeypatch, caplog
):
    """A potentially live worker must remain globally discoverable after stop fails."""
    from infinidev.code_intel import background_indexer

    cleared = []

    class FailingQueue:
        def stop(self):
            raise RuntimeError("queue worker did not stop")

    queue = FailingQueue()
    monkeypatch.setattr(background_indexer, "get_global_queue", lambda: queue)
    monkeypatch.setattr(
        background_indexer,
        "release_global_queue",
        lambda value: cleared.append(value),
    )
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._cleanup_classic_runtime(None, lambda _handler: None)

    assert cleared == []
    record = next(
        record
        for record in caplog.records
        if record.getMessage() == "Failed to stop background index queue"
    )
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError


def test_interactive_cleanup_retains_timed_out_index_worker(
    cli_main, monkeypatch, caplog
):
    """The real queue timeout must preserve both worker and global ownership."""
    from infinidev.cli.index_queue import IndexQueue
    from infinidev.code_intel import background_indexer

    class TimedOutWorker:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            return None

    queue = IndexQueue(project_id=1)
    worker = TimedOutWorker()
    queue._worker = worker
    background_indexer.set_global_queue(queue)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    try:
        with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
            cli_main._cleanup_classic_runtime(None, lambda _handler: None)

        assert background_indexer.get_global_queue() is queue
        assert queue._worker is worker
        assert queue.is_running()
        record = next(
            record
            for record in caplog.records
            if record.getMessage() == "Failed to stop background index queue"
        )
        assert record.exc_info is not None
        assert record.exc_info[0] is TimeoutError
    finally:
        background_indexer.set_global_queue(None)


def test_background_queue_acquisition_is_atomic():
    """A live owner wins the global slot until it releases ownership."""
    from infinidev.code_intel import background_indexer

    class Queue:
        def __init__(self, running=True):
            self.running = running

        def is_running(self):
            return self.running

    winner = Queue()
    loser = Queue()
    stale = Queue(running=False)
    background_indexer.set_global_queue(None)
    try:
        assert background_indexer.acquire_global_queue(winner) is True
        assert background_indexer.acquire_global_queue(loser) is False
        assert background_indexer.get_global_queue() is winner

        background_indexer.set_global_queue(stale)
        assert background_indexer.acquire_global_queue(loser) is True
        assert background_indexer.get_global_queue() is loser
    finally:
        background_indexer.set_global_queue(None)


def test_background_queue_release_is_identity_aware():
    """A stale owner cannot clear a replacement process-wide queue."""
    from infinidev.code_intel import background_indexer

    original_queue = object()
    replacement_queue = object()
    background_indexer.set_global_queue(replacement_queue)
    try:
        assert background_indexer.release_global_queue(original_queue) is False
        assert background_indexer.get_global_queue() is replacement_queue
        assert background_indexer.release_global_queue(replacement_queue) is True
        assert background_indexer.get_global_queue() is None
    finally:
        background_indexer.set_global_queue(None)


def test_interactive_cleanup_preserves_concurrent_replacement(cli_main, monkeypatch):
    """Cleanup must release only the queue whose worker it stopped."""
    from infinidev.code_intel import background_indexer

    class ReplacementQueue:
        pass

    replacement_queue = ReplacementQueue()

    class StaleQueue:
        def stop(self):
            background_indexer.set_global_queue(replacement_queue)

    stale_queue = StaleQueue()
    background_indexer.set_global_queue(stale_queue)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    try:
        cli_main._cleanup_classic_runtime(None, lambda _handler: None)
        assert background_indexer.get_global_queue() is replacement_queue
    finally:
        background_indexer.set_global_queue(None)


def test_interactive_cleanup_does_not_stop_preexisting_replacement(cli_main, monkeypatch):
    """Teardown targets its own queue even when registration changed earlier."""
    from infinidev.code_intel import background_indexer

    stopped = []

    class Queue:
        def __init__(self, name):
            self.name = name

        def stop(self):
            stopped.append(self.name)

    owned_queue = Queue("owned")
    replacement_queue = Queue("replacement")
    background_indexer.set_global_queue(replacement_queue)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    try:
        cli_main._cleanup_classic_runtime(
            None,
            lambda _handler: None,
            owned_queue=owned_queue,
        )

        assert stopped == ["owned"]
        assert background_indexer.get_global_queue() is replacement_queue
    finally:
        background_indexer.set_global_queue(None)


def test_single_prompt_cleanup_failures_do_not_skip_later_cleanup(
    cli_main, monkeypatch, caplog
):
    """Teardown failures must not skip later single-prompt cleanup."""
    from infinidev.cli import classic_renderer, session_resume
    from infinidev.code_intel import background_indexer
    from infinidev.tools import permission

    cleanup_calls = []
    permission_handlers = []

    class TrackingWatcher:
        def stop(self):
            cleanup_calls.append("watcher")

    class TrackingQueue:
        _file_watcher = TrackingWatcher()

        def stop(self):
            cleanup_calls.append("queue")

    class FailingRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            return None

        def unsubscribe(self):
            cleanup_calls.append("renderer")
            raise RuntimeError("renderer cleanup unavailable")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(cli_main, "_render_final", lambda _result: None)
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(cli_main, "ReviewEngine", lambda: object())
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", FailingRenderer)
    monkeypatch.setattr(session_resume, "begin_fresh_session", lambda _session_id: None)
    monkeypatch.setattr(
        background_indexer, "get_global_queue", lambda: TrackingQueue()
    )
    monkeypatch.setattr(
        background_indexer,
        "release_global_queue",
        lambda _queue: cleanup_calls.append("clear") or True,
    )

    def set_permission_handler(handler):
        permission_handlers.append(handler)
        if handler is None:
            raise RuntimeError("permission cleanup unavailable")

    monkeypatch.setattr(permission, "set_permission_handler", set_permission_handler)
    monkeypatch.setattr(
        permission,
        "make_noninteractive_permission_handler",
        lambda _prompt: "noninteractive-handler",
    )
    monkeypatch.setattr(orchestration, "run_task", lambda **_kwargs: "finished")

    with caplog.at_level(logging.DEBUG, logger=cli_main.__name__):
        cli_main._run_single_prompt("Fix it")

    assert cleanup_calls == ["renderer", "ken", "watcher", "queue", "clear"]
    assert permission_handlers == ["noninteractive-handler", None]
    records = {
        record.getMessage(): record
        for record in caplog.records
        if record.getMessage()
        in {
            "Failed to unsubscribe classic renderer",
            "Failed to reset classic permission handler",
        }
    }
    assert set(records) == {
        "Failed to unsubscribe classic renderer",
        "Failed to reset classic permission handler",
    }
    for record in records.values():
        assert record.levelno == logging.DEBUG
        assert record.exc_info is not None
        assert record.exc_info[0] is RuntimeError


def test_single_prompt_initialization_failure_cleans_bootstrapped_services(
    cli_main, monkeypatch
):
    """A failure before renderer construction must not leak bootstrap resources."""
    from infinidev.code_intel import background_indexer
    from infinidev.tools import permission

    cleanup_calls = []

    class TrackingQueue:
        def stop(self):
            cleanup_calls.append("queue")

    def fail_agent(*_args, **_kwargs):
        raise RuntimeError("agent unavailable")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "InfinidevAgent", fail_agent)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(
        background_indexer, "get_global_queue", lambda: TrackingQueue()
    )
    monkeypatch.setattr(
        background_indexer,
        "release_global_queue",
        lambda _queue: cleanup_calls.append("clear") or True,
    )
    monkeypatch.setattr(
        permission,
        "set_permission_handler",
        lambda handler: cleanup_calls.append("permission") if handler is None else None,
    )

    with pytest.raises(RuntimeError, match="agent unavailable"):
        cli_main._run_single_prompt("Fix it")

    assert cleanup_calls == ["permission", "ken", "queue", "clear"]


def test_single_prompt_subscription_failure_still_runs_cleanup(
    cli_main, monkeypatch
):
    """Renderer subscription failure must not leak permissions or Ken resources."""
    from infinidev.cli import classic_renderer, session_resume
    from infinidev.tools import permission

    cleanup_calls = []
    permission_handlers = []

    class FailingRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            raise RuntimeError("renderer subscription unavailable")

        def unsubscribe(self):
            cleanup_calls.append("renderer")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", FailingRenderer)
    monkeypatch.setattr(session_resume, "begin_fresh_session", lambda _session_id: None)
    monkeypatch.setattr(
        permission,
        "set_permission_handler",
        lambda handler: permission_handlers.append(handler),
    )

    with pytest.raises(RuntimeError, match="renderer subscription unavailable"):
        cli_main._run_single_prompt("Fix it")

    assert cleanup_calls == ["renderer", "ken"]
    assert permission_handlers == [None]


def test_single_prompt_permission_setup_failure_still_runs_cleanup(
    cli_main, monkeypatch
):
    """Permission setup failure must not leak renderer or Ken resources."""
    from infinidev.cli import classic_renderer, session_resume
    from infinidev.tools import permission

    cleanup_calls = []
    permission_handlers = []

    class TrackingRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            return None

        def unsubscribe(self):
            cleanup_calls.append("renderer")

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(
        cli_main, "_end_ken_sessions", lambda: cleanup_calls.append("ken")
    )
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", TrackingRenderer)
    monkeypatch.setattr(session_resume, "begin_fresh_session", lambda _session_id: None)
    monkeypatch.setattr(
        permission,
        "make_noninteractive_permission_handler",
        lambda _prompt: "noninteractive-handler",
    )

    def fail_initial_permission_setup(handler):
        permission_handlers.append(handler)
        if handler is not None:
            raise RuntimeError("permission setup unavailable")

    monkeypatch.setattr(
        permission, "set_permission_handler", fail_initial_permission_setup
    )

    with pytest.raises(RuntimeError, match="permission setup unavailable"):
        cli_main._run_single_prompt("Fix it")

    assert cleanup_calls == ["renderer", "ken"]
    assert permission_handlers == ["noninteractive-handler", None]


def test_single_prompt_continue_preserves_cross_workspace_session(
    cli_main, monkeypatch, temp_db
):
    """One-shot ``-c`` reuses history without moving its workspace ownership."""
    from infinidev.cli import classic_renderer
    from infinidev.db.service import (
        get_last_session,
        register_session,
        store_conversation_turn,
    )
    from infinidev.tools import permission

    session_id = "single-prompt-cross-workspace"
    register_session(session_id, "/original-workspace")
    store_conversation_turn(session_id, "user", "continue this work")
    run_calls = []

    class SilentRenderer:
        def __init__(self, _status):
            pass

        def subscribe(self):
            return None

        def unsubscribe(self):
            return None

    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    monkeypatch.setattr(cli_main, "_render_final", lambda _result: None)
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(cli_main, "ReviewEngine", lambda: object())
    monkeypatch.setattr(classic_renderer, "ClassicRenderer", SilentRenderer)
    monkeypatch.setattr(permission, "set_permission_handler", lambda _handler: None)
    monkeypatch.setattr(
        permission,
        "make_noninteractive_permission_handler",
        lambda _prompt: lambda *_args, **_kwargs: True,
    )
    monkeypatch.setattr(
        orchestration,
        "run_task",
        lambda **kwargs: run_calls.append(kwargs) or "finished",
    )

    cli_main._run_single_prompt("Continue", continue_session=True)

    assert run_calls[0]["session_id"] == session_id
    assert get_last_session("/original-workspace")["session_id"] == session_id
    assert get_last_session(os.getcwd()) is None


def test_single_prompt_subscribes_live_classic_renderer(
    cli_main, monkeypatch, capsys
):
    """Tool starts must be visible while a one-shot prompt is running."""
    monkeypatch.setattr(cli_main, "_bootstrap_single_prompt_runtime", lambda: None)
    monkeypatch.setattr(cli_main, "_end_ken_sessions", lambda: None)
    monkeypatch.setattr(cli_main, "InfinidevAgent", lambda agent_id: object())
    monkeypatch.setattr(cli_main, "LoopEngine", lambda: object())
    monkeypatch.setattr(cli_main, "ReviewEngine", lambda: object())

    from infinidev.cli import session_resume
    from infinidev.tools import permission

    monkeypatch.setattr(session_resume, "begin_fresh_session", lambda _session_id: None)
    monkeypatch.setattr(permission, "set_permission_handler", lambda _handler: None)
    monkeypatch.setattr(
        permission,
        "make_noninteractive_permission_handler",
        lambda _prompt: lambda *_args, **_kwargs: True,
    )

    def fake_run_task(**_kwargs):
        event_bus.emit("loop_tool_start", 1, "cli_agent", {
            "tool_name": "execute_command",
            "tool_detail": "uv run pytest",
            "call_num": 1,
            "total_calls": 1,
        })
        return "finished"

    monkeypatch.setattr(orchestration, "run_task", fake_run_task)
    rendered_results = []
    original_render_final = cli_main._render_final

    def record_final(result):
        rendered_results.append(result)
        original_render_final(result)

    monkeypatch.setattr(cli_main, "_render_final", record_final)

    cli_main._run_single_prompt("Fix it")

    assert rendered_results == ["finished"]
    out = capsys.readouterr().out
    assert "running" in out
    assert "execute_command" in out
    assert "uv run pytest" in out
    assert "finished" in out

    event_bus.emit("loop_tool_start", 1, "cli_agent", {
        "tool_name": "read_file",
        "tool_detail": "should-not-render",
    })
    assert capsys.readouterr().out == ""
