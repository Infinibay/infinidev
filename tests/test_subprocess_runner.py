"""Regression tests for bounded subprocess execution."""

from __future__ import annotations

import os
import sys
import time

import pytest

from infinidev.engine.subprocess_runner import run_captured


def test_run_captured_closes_stdin_by_default(tmp_path) -> None:
    result = run_captured(
        [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
        cwd=str(tmp_path),
        timeout=5,
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "''"
    assert not result.timed_out


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX process groups")
def test_timeout_kills_child_that_ignores_sigterm_and_holds_output_pipe(tmp_path) -> None:
    script = (
        "import os, signal, time; "
        "child = os.fork(); "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN) if child == 0 else None; "
        "time.sleep(30) if child == 0 else None"
    )
    started = time.monotonic()

    result = run_captured(
        [sys.executable, "-c", script],
        cwd=str(tmp_path),
        timeout=0.1,
    )

    assert result.timed_out
    assert result.exit_code == -1
    assert time.monotonic() - started < 5
