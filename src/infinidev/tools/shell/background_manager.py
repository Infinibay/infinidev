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
import signal
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
        # Close any pipes still open — the EOF branch closes per-stream, but
        # the select-error `break` above exits with handles still in `fds`,
        # which would otherwise leak the pipe fds for the process's lifetime.
        for f in (self.proc.stdout, self.proc.stderr):
            if f is not None:
                try:
                    f.close()
                except Exception:
                    pass
        try:
            self.proc.wait()
        except Exception:
            pass
        newly_finished = False
        with self._lock:
            if self.end_time is None:
                self.end_time = time.monotonic()
                newly_finished = True
        # Signal a natural completion so the developer loop notices a task
        # that finished WHILE it was working on something else. Skip tasks the
        # agent stopped itself (killed_reason is set) — it already knows — and
        # skip ones already surfaced via wait/status (acknowledged). The queue
        # is drained into a <background-task-finished> prompt block each turn.
        if newly_finished and self.killed_reason is None:
            _queue_completion(self.id)

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

    def status_line(self) -> str:
        """One-line human verdict — the single source of truth for the
        'running for 12s' / 'exited ok' / 'FAILED' phrasing shared by the
        <background-tasks> prompt block, the TUI explorer, and /tasks.
        """
        runtime = self.runtime_seconds()
        if self.status == "running":
            return f"running for {runtime:.0f}s"
        if self.status == "killed":
            return f"stopped after {runtime:.0f}s"
        code = self.exit_code if self.exit_code is not None else "?"
        verdict = "ok" if self.exit_code == 0 else f"FAILED (exit {code})"
        return f"exited {verdict} after {runtime:.0f}s"

    def output(self) -> tuple[str, str]:
        """Return decoded (stdout, stderr) snapshots."""
        with self._lock:
            out = bytes(self._stdout)
            err = bytes(self._stderr)
        return out.decode(errors="replace"), err.decode(errors="replace")

    # ── Control ──────────────────────────────────────────────────────
    def wait(
        self,
        timeout: float | None = None,
        until_text: str | None = None,
        poll_interval: float = 0.1,
    ) -> str:
        """Block until a completion condition is met or ``timeout`` elapses.

        The wait ends on whichever comes first:
          - the process exits, or
          - ``until_text`` (if given) appears in captured stdout/stderr —
            the readiness signal for commands that never exit (dev servers,
            watchers), where waiting for exit would always time out.

        Returns the reason the wait ended: ``"matched"`` (until_text seen),
        ``"exited"`` (process is no longer running), or ``"timeout"``.
        ``timeout=None`` waits indefinitely; callers should always pass a
        bound so a wait can never block the CLI forever.

        We poll ``poll()`` rather than call ``proc.wait()`` so we never
        contend with the pump thread's reaping wait() on the same handle.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            if until_text and self._buffers_contain(until_text):
                return "matched"
            if not self.is_running:
                # Process gone, but the pump thread may still be draining the
                # final slice of output. Give it a moment so an until_text
                # printed just before exit isn't missed.
                self._reader.join(timeout=0.5)
                if until_text and self._buffers_contain(until_text):
                    return "matched"
                return "exited"
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return "timeout"
                time.sleep(min(poll_interval, remaining))
            else:
                time.sleep(poll_interval)

    def _buffers_contain(self, needle: str) -> bool:
        """True if ``needle`` is present in either captured stream."""
        out, err = self.output()
        return needle in out or needle in err

    def _signal_group(self, sig: int) -> None:
        """Send ``sig`` to the child's whole process group.

        ``start_new_session=True`` (see ``BackgroundTaskManager.start``) makes
        ``proc.pid`` the leader of a new process group, so the real workload —
        which runs as a grandchild of ``/bin/sh -c`` (node, docker, pytest
        workers, …) — shares that group. Signalling only ``self.proc`` would
        kill the shell and leak the grandchildren; signalling the group reaches
        them all. Falls back to the direct child if the group is already gone.
        POSIX-only, consistent with the select/os.read-based pump.
        """
        try:
            os.killpg(os.getpgid(self.proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                if sig == signal.SIGKILL:
                    self.proc.kill()
                else:
                    self.proc.terminate()
            except Exception:
                pass

    def stop(self, force: bool = False, reason: str | None = None) -> bool:
        """Stop the process group. Returns True if it was running.

        ``force`` skips the graceful SIGTERM and sends SIGKILL immediately.
        Otherwise we terminate, wait briefly, and escalate to kill only if
        the child ignores the term signal.
        """
        if not self.is_running:
            return False
        intended_reason = reason or (
            "Force-killed by agent" if force else "Stopped by agent"
        )
        # Set eagerly so the pump thread's completion de-dup sees this as an
        # agent-initiated stop and doesn't queue a spurious "finished" notice.
        self.killed_reason = intended_reason
        try:
            if force:
                self._signal_group(signal.SIGKILL)
            else:
                self._signal_group(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._signal_group(signal.SIGKILL)
            # Reap so poll()/status reflect the outcome immediately rather than
            # racing the pump thread. wait() is thread-safe — both this call
            # and the pump's reap return the same code.
            try:
                self.proc.wait(timeout=3)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("Error stopping background task %s: %s", self.id, exc)
        # If the child actually exited on its own (won the TOCTOU race against
        # our signal), subprocess reports a non-negative return code; a
        # signalled death is -signum. Don't mislabel a natural completion
        # (including a graceful exit-0 on SIGTERM) as an agent kill.
        rc = self.proc.poll()
        if rc is not None and rc >= 0:
            self.killed_reason = None
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


# ── Completion notifications ─────────────────────────────────────────────
# When a background task finishes naturally while the developer loop is busy
# with something else, the loop has no way to know unless it polls. This
# module-level queue (mirrors code_intel.file_change_notifications) records
# the ids of tasks that just completed; build_iteration_prompt drains it once
# per iteration and renders a <background-task-finished> block so the agent is
# told about it on its next step. Guarded by its own lock — the pump threads
# write to it; the prompt builder reads and clears it.
_pending_completions: list[str] = []
_completions_lock = threading.Lock()


def _queue_completion(task_id: str) -> None:
    """Record that ``task_id`` finished (called by the pump thread)."""
    with _completions_lock:
        if task_id not in _pending_completions:
            _pending_completions.append(task_id)


def acknowledge_completion(task_id: str) -> None:
    """Drop a task from the pending queue — its completion was already shown.

    Called by ``wait_for_background_task`` / ``background_status`` when they
    surface a finished task to the model directly, so the next iteration's
    <background-task-finished> block doesn't redundantly re-announce it.
    """
    with _completions_lock:
        if task_id in _pending_completions:
            _pending_completions.remove(task_id)


def drain_completed_notifications() -> list["BackgroundTask"]:
    """Pop and return the tasks that finished since the last drain."""
    with _completions_lock:
        ids = list(_pending_completions)
        _pending_completions.clear()
    if not ids:
        return []
    mgr = get_background_manager()
    return [t for t in (mgr.get(i) for i in ids) if t is not None]


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
