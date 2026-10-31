"""Real-process regressions for interactive command output and cleanup."""

from __future__ import annotations

import os
import shlex
import signal
import sys
import threading
import time

import pytest

from infinidev.config.settings import settings
from infinidev.engine.command_output_store import CommandOutputHandle, CommandOutputStore
from infinidev.engine.tool_progress import tool_progress_context
from infinidev.tools.shell import execute_command_tool as command_module
from infinidev.tools.shell.execute_command_tool import ExecuteCommandTool


def _run_interactive(tool, source, tmp_path, *, timeout=3):
    command = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(source)}"
    return tool._run_with_stdin_detection(command, str(tmp_path), dict(os.environ), timeout)


@pytest.mark.parametrize("close_stdout", [False, True])
def test_eof_does_not_busy_poll_closed_streams(monkeypatch, tmp_path, close_stdout):
    original_select = command_module.select.select
    select_calls = 0

    def counted_select(*args):
        nonlocal select_calls
        select_calls += 1
        return original_select(*args)

    monkeypatch.setattr(command_module.select, "select", counted_select)
    source = "import os, time; os.close(2); "
    source += "os.close(1); " if close_stdout else "print('ready', flush=True); "
    source += "time.sleep(0.35)"
    if not close_stdout:
        source += "; print('finished', flush=True)"

    result = _run_interactive(ExecuteCommandTool(), source, tmp_path)

    assert result["success"] is True
    assert result["stdout"] == ("" if close_stdout else "ready\nfinished\n")
    assert select_calls < 40, f"EOF caused {select_calls} immediate polls"


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt])
def test_reader_exception_reaps_process(monkeypatch, tmp_path, error_type):
    original_popen = command_module.subprocess.Popen
    processes = []

    def tracked_popen(*args, **kwargs):
        proc = original_popen(*args, **kwargs)
        processes.append(proc)
        return proc

    def fail_on_output(chunk):
        raise error_type("reader interrupted")

    monkeypatch.setattr(command_module.subprocess, "Popen", tracked_popen)
    monkeypatch.setattr("infinidev.engine.tool_progress.emit_tool_output", fail_on_output)
    try:
        with pytest.raises(error_type, match="reader interrupted"):
            _run_interactive(
                ExecuteCommandTool(),
                "import time; print('started', flush=True); time.sleep(30)",
                tmp_path,
            )

        proc = processes[0]
        assert proc.poll() is not None, "reader failure left the command running"
        assert all(pipe.closed for pipe in (proc.stdin, proc.stdout, proc.stderr))
    finally:
        for proc in processes:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=3)


def test_timeout_still_applies_after_both_streams_close(tmp_path):
    started = time.monotonic()
    result = _run_interactive(
        ExecuteCommandTool(),
        "import os, time; os.close(1); os.close(2); time.sleep(30)",
        tmp_path,
        timeout=1,
    )

    assert result["success"] is False
    assert "timed out" in result["killed_reason"]
    assert time.monotonic() - started < 5


def test_cancellation_still_applies_after_both_streams_close(tmp_path):
    cancel_event = threading.Event()
    timer = threading.Timer(0.15, cancel_event.set)
    timer.start()
    try:
        with tool_progress_context("closed-streams", 1, "agent-a", cancel_event=cancel_event):
            result = _run_interactive(
                ExecuteCommandTool(),
                "import os, time; os.close(1); os.close(2); time.sleep(30)",
                tmp_path,
            )
    finally:
        timer.cancel()

    assert result["success"] is False
    assert result["killed_reason"] == "Command interrupted by user"


@pytest.mark.parametrize("reply", ["answer", None])
def test_stdin_prompt_can_be_answered_or_stopped(monkeypatch, tmp_path, reply):
    monkeypatch.setattr(command_module, "request_stdin_input", lambda *args: reply)
    result = _run_interactive(
        ExecuteCommandTool(),
        "import sys; sys.stderr.write('Password:'); sys.stderr.flush(); print(input())",
        tmp_path,
    )

    if reply is None:
        assert result["success"] is False
        assert "kill the process" in result["killed_reason"]
    else:
        assert result["success"] is True
        assert result["stdout"] == "answer\n"


def _read_capture(raw, tool):
    handle = CommandOutputHandle(
        artifact_id=raw["artifact_id"],
        artifact_type=raw["type"],
        stream=raw["stream"],
        char_count=raw["char_count"],
        byte_count=raw["byte_count"],
    )
    return CommandOutputStore().read_text(
        handle, project_id=tool.project_id, session_id=tool.session_id,
    )


@pytest.mark.parametrize("overflow_stream", ["stdout", "stderr"])
def test_incomplete_stream_is_not_published_as_full_capture(
    bound_tool, monkeypatch, tmp_path, overflow_stream,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
    monkeypatch.setattr(settings, "COMMAND_OUTPUT_MAX_ARTIFACT_BYTES", 65536)
    complete_stream = "stderr" if overflow_stream == "stdout" else "stdout"
    complete_text = "complete-head\n" + "λ" * 11000
    source = (
        "import sys, time; "
        f"sys.{overflow_stream}.write('discarded-head\\n' + 'x' * 131072); "
        f"sys.{overflow_stream}.flush(); "
        f"sys.{complete_stream}.write('complete-head\\n' + 'λ' * 11000); "
        f"sys.{complete_stream}.flush(); "
        "time.sleep(0.1)"
    )
    tool = bound_tool(ExecuteCommandTool)

    result = _run_interactive(tool, source, tmp_path)

    assert result["success"] is True
    assert result[overflow_stream] == "x" * (10000 if overflow_stream == "stdout" else 5000)
    handles = result["command_output_handles"]
    assert overflow_stream not in handles, "a truncated tail was advertised as complete"
    assert _read_capture(handles[complete_stream], tool) == complete_text


def test_complete_stream_at_buffer_limit_remains_recoverable(bound_tool, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
    monkeypatch.setattr(settings, "COMMAND_OUTPUT_MAX_ARTIFACT_BYTES", 65536)
    text = "x" * 65536
    tool = bound_tool(ExecuteCommandTool)

    result = _run_interactive(
        tool,
        "import sys, time; sys.stdout.write('x' * 65536); sys.stdout.flush(); time.sleep(0.1)",
        tmp_path,
    )

    assert result["success"] is True
    assert _read_capture(result["command_output_handles"]["stdout"], tool) == text
