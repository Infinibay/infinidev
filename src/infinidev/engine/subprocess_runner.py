"""Bounded subprocess execution shared by non-interactive command paths."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CapturedProcess:
    """Captured output and termination state for one subprocess."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False


def run_captured(
    command: str | Sequence[str],
    *,
    cwd: str | None,
    timeout: int,
    shell: bool = False,
    env: Mapping[str, str] | None = None,
    input_text: str | None = None,
) -> CapturedProcess:
    """Run with closed stdin and terminate the complete process group on timeout."""
    proc = subprocess.Popen(
        command,
        shell=shell,
        stdin=subprocess.PIPE if input_text is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
        env=env,
        start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = proc.communicate(input=input_text, timeout=timeout)
        return CapturedProcess(
            exit_code=proc.returncode,
            stdout=stdout or "",
            stderr=stderr or "",
        )
    except subprocess.TimeoutExpired:
        terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        return CapturedProcess(
            exit_code=-1,
            stdout=stdout or "",
            stderr=stderr or "",
            timed_out=True,
        )
    except BaseException:
        terminate_process_group(proc)
        raise


def terminate_process_group(proc: subprocess.Popen[str]) -> None:
    """Terminate a process session and escalate to SIGKILL after one second."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except OSError:
            proc.terminate()

        # Waiting only for the session leader is insufficient: it can exit
        # while a child ignores SIGTERM and keeps stdout/stderr pipes open.
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            proc.poll()
            if not _process_group_exists(proc.pid):
                return
            time.sleep(0.05)
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except OSError:
            proc.kill()
        proc.wait()
        return

    # Windows has no POSIX process groups. Keep the compatibility path
    # bounded even though it can only terminate the direct process.
    proc.terminate()  # pragma: no cover - Windows compatibility path
    try:  # pragma: no cover - Windows compatibility path
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()
        proc.wait()


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether a POSIX process group still has a live member."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
