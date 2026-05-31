"""Process-global registry for background shell commands.

The developer loop can launch long-running commands (dev servers, test
watchers, builds) without blocking its turn. Each launch returns a short
task id; the command keeps running in a child process while the loop moves
on. A daemon reader thread drains the child's stdout/stderr into in-memory
buffers so the OS pipe never fills and blocks the child, and so the loop can
ask for the accumulated output at any time.

The registry is a module-level singleton (like the file-change notification
queue) because both the tools that mutate it AND the prompt builder that
renders the ``<background-tasks>`` section need to reach the same state, and
neither has a natural place to thread an instance through. ``atexit`` kills
any survivors so we never leak children when the CLI exits.
"""

from __future__ import annotations

import atexit
import itertools
import logging
import os
import select
import subprocess
import threading
import time

logger = logging.getLogger(__name__)

# How much output we retain per stream. Background commands (servers, log
# tailers) can emit unbounded output over their lifetime; we keep only the
# most recent slice so a forgotten task can't exhaust memory. Mirrors the
# truncation execute_command applies to foreground output.
_MAX_BUFFER_BYTES = 256 * 1024


class BackgroundTask:
    """A single background child process plus its captured output.

    Output capture runs on a daemon thread (``_pump``) that selects over the
    child's stdout/stderr pipes until both hit EOF, then reaps the process.
    All buffer access is guarded by ``_lock`` because the pump thread writes
    while the tool threads read.
    """

    def __init__(
        self,
        task_id: str,
        command: str,
        description: str,
        proc: subprocess.Popen,
        cwd: str,
    ) -> None:
        self.id = task_id
        self.command = command
        self.description = description
        self.proc = proc
        self.cwd = cwd
        self.start_time = time.monotonic()
        self.start_wall = time.time()
        self.end_time: float | None = None
        self.killed_reason: str | None = None
        self._stdout = bytearray()
        self._stderr = bytearray()
        self._lock = threading.Lock()
        self._reader = threading.Thread(
            target=self._pump, name=f"bg-pump-{task_id}", daemon=True
        )
        self._reader.start()

    # ── Output capture ───────────────────────────────────────────────
    def _pump(self) -> None:
        """Drain stdout/stderr into buffers until both pipes hit EOF."""
        fds = [f for f in (self.proc.stdout, self.proc.stderr) if f is not None]
        while fds:
            try:
                ready, _, _ = select.select(fds, [], [], 0.2)
            except (ValueError, OSError):
                break
            for r in ready:
                try:
                    data = os.read(r.fileno(), 65536)
                except OSError:
                    data = b""
                if not data:
                    # EOF on this stream — stop watching it.
                    fds.remove(r)
                    try:
                        r.close()
                    except Exception:
                        pass
                    continue
                with self._lock:
                    buf = self._stdout if r is self.proc.stdout else self._stderr
                    buf.extend(data)
                    # Keep only the trailing window per stream.
                    if len(buf) > _MAX_BUFFER_BYTES:
                        del buf[:-_MAX_BUFFER_BYTES]
        try:
            self.proc.wait()
        except Exception:
            pass
        with self._lock:
            if self.end_time is None:
                self.end_time = time.monotonic()

    # ── State accessors ──────────────────────────────────────────────
    @property
    def is_running(self) -> bool:
        return self.proc.poll() is None

    @property
    def status(self) -> str:
        if self.is_running:
            return "running"
        if self.killed_reason is not None:
            return "killed"
        return "exited"

    @property
    def exit_code(self) -> int | None:
        return self.proc.poll()

    def runtime_seconds(self) -> float:
        end = self.end_time if self.end_time is not None else time.monotonic()
        return max(0.0, end - self.start_time)

    def output(self) -> tuple[str, str]:
        """Return decoded (stdout, stderr) snapshots."""
        with self._lock:
            out = bytes(self._stdout)
            err = bytes(self._stderr)
        return out.decode(errors="replace"), err.decode(errors="replace")

    # ── Control ──────────────────────────────────────────────────────
    def stop(self, force: bool = False, reason: str | None = None) -> bool:
        """Stop the process. Returns True if it was running and is now down.

        ``force`` skips the graceful SIGTERM and sends SIGKILL immediately.
        Otherwise we terminate, wait briefly, and escalate to kill only if
        the child ignores the term signal.
        """
        if not self.is_running:
            return False
        self.killed_reason = reason or (
            "Force-killed by agent" if force else "Stopped by agent"
        )
        try:
            if force:
                self.proc.kill()
            else:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
            # Reap so poll()/status reflect the kill immediately rather than
            # racing the pump thread. wait() is thread-safe — both this call
            # and the pump's reap return the same code.
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Error stopping background task %s: %s", self.id, exc)
        return True

    def summary(self) -> dict:
        """Compact dict describing the task (used by tools and the prompt)."""
        out, err = self.output()
        return {
            "id": self.id,
            "description": self.description,
            "command": self.command,
            "status": self.status,
            "exit_code": self.exit_code,
            "runtime_seconds": round(self.runtime_seconds(), 1),
            "stdout": out,
            "stderr": err,
        }


class BackgroundTaskManager:
    """Singleton registry of background tasks for the current process."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}
        self._counter = itertools.count(1)
        self._lock = threading.Lock()
        atexit.register(self.shutdown)

    def start(
        self,
        command: str,
        description: str,
        cwd: str,
        env: dict[str, str] | None = None,
    ) -> BackgroundTask:
        """Spawn ``command`` detached and register it. Raises on spawn failure."""
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            bufsize=0,
            # New session so a later kill can't take down the parent CLI and
            # so the child detaches from our controlling terminal.
            start_new_session=True,
        )
        with self._lock:
            task_id = f"bg-{next(self._counter)}"
            task = BackgroundTask(task_id, command, description, proc, cwd)
            self._tasks[task_id] = task
        logger.debug("Started background task %s: %s", task_id, command)
        return task

    def get(self, task_id: str) -> BackgroundTask | None:
        with self._lock:
            return self._tasks.get(task_id)

    def list(self) -> list[BackgroundTask]:
        with self._lock:
            return list(self._tasks.values())

    def active(self) -> list[BackgroundTask]:
        """Tasks that are still running."""
        return [t for t in self.list() if t.is_running]

    def stop(self, task_id: str, force: bool = False) -> BackgroundTask | None:
        task = self.get(task_id)
        if task is None:
            return None
        task.stop(force=force)
        return task

    def shutdown(self) -> None:
        """Force-kill every still-running task. Registered with atexit."""
        for task in self.list():
            if task.is_running:
                try:
                    task.stop(force=True, reason="Process exit cleanup")
                except Exception:
                    pass


_manager: BackgroundTaskManager | None = None
_manager_lock = threading.Lock()


def get_background_manager() -> BackgroundTaskManager:
    """Return the process-global background task manager (lazy singleton)."""
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = BackgroundTaskManager()
    return _manager
