"""pexpect-based driver for the Infinidev TUI.

These tests spawn a real ``infinidev`` process inside a pseudo-TTY,
send keystrokes, and assert on what comes back. They exercise the
true user-facing path (TUI rendering, prompt_toolkit input, the
orchestration pipeline, the engine loop) without any mocks.

The driver is intentionally tolerant about ANSI colour codes and
unicode borders — TUIs emit a lot of decorative noise that's
irrelevant to the assertion. ``Session.read_text()`` strips it
before matching.

Cost of one session: ~3-15 seconds (TUI boot ~2s, plus whatever
the test asks the model to do). Use small/local models in tests
or mock the LLM with ``--model none`` (a future flag).

Usage:

    from tests.interactive._session import Session

    with Session(workspace="/tmp/empty") as s:
        s.wait_for_ready()
        t0 = time.time()
        s.send("Hola")
        s.wait_for("Infinidev:", timeout=10)
        elapsed = time.time() - t0
        assert elapsed < 5, f"hola took {elapsed:.1f}s"
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pexpect

# ANSI escape stripping — covers CSI sequences (colours, cursor moves)
# plus the OSC sequences some TUIs emit for window titles. We do NOT
# strip box-drawing chars because some assertions want to know which
# pane the text appeared in.
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]|\x1b\][^\x07]*\x07")


def strip_ansi(text: str) -> str:
    """Remove ANSI colour and CSI sequences from a TUI output buffer."""
    return _ANSI_RE.sub("", text)


class Session:
    """A live infinidev TUI process driven over a pseudo-TTY.

    Parameters
    ----------
    workspace
        Directory to launch the TUI in. Defaults to a fresh tempdir.
    model
        LiteLLM-format model id. Defaults to ``ollama/qwen3.5:4b``
        (smallest local model — fast TUI tests).
    extra_args
        Additional CLI flags to forward to ``infinidev``.
    cols / rows
        Pseudo-TTY size. Defaults to 120x40 which is wide enough
        that prompt_toolkit doesn't truncate output mid-line.
    """

    def __init__(
        self,
        workspace: Optional[str] = None,
        model: str = "ollama/qwen3.5:4b",
        extra_args: Optional[list[str]] = None,
        cols: int = 120,
        rows: int = 40,
    ) -> None:
        self._workspace = workspace
        self._model = model
        self._extra_args = list(extra_args or [])
        self._cols = cols
        self._rows = rows
        self._tmpdir: Optional[str] = None
        self._child: Optional[pexpect.spawn] = None

    # ── Lifecycle ──────────────────────────────────────────────────────

    def __enter__(self) -> "Session":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    def start(self) -> None:
        cwd = self._workspace
        if cwd is None:
            self._tmpdir = tempfile.mkdtemp(prefix="infinidev_test_")
            cwd = self._tmpdir

        env = os.environ.copy()
        env["INFINIBAY_LLM_MODEL"] = self._model
        # Force a deterministic terminal so prompt_toolkit doesn't
        # try to negotiate truecolor / sixel / kitty graphics.
        env["TERM"] = "xterm-256color"
        # Disable any background indexer threads that might log over
        # the captured output (the test doesn't need them).
        env.setdefault("INFINIBAY_CODE_INTEL_AUTO_INDEX", "false")

        # Always go through `python -m infinidev.cli.main` so we hit
        # the canonical entry point regardless of where the
        # `infinidev` script lives in the PATH.
        cmd = sys.executable
        args = ["-m", "infinidev.cli.main"]
        if self._model:
            args += ["--model", self._model]
        args += self._extra_args

        self._child = pexpect.spawn(
            cmd, args,
            cwd=cwd,
            env=env,
            dimensions=(self._rows, self._cols),
            encoding="utf-8",
            codec_errors="replace",
            timeout=60,
        )

    def stop(self) -> None:
        if self._child and self._child.isalive():
            try:
                self._child.sendcontrol("c")
                self._child.expect(pexpect.EOF, timeout=5)
            except (pexpect.TIMEOUT, pexpect.EOF):
                pass
            try:
                self._child.terminate(force=True)
            except Exception:
                pass
        if self._tmpdir and os.path.isdir(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ── Interaction ────────────────────────────────────────────────────

    def wait_for_ready(self, timeout: float = 30.0) -> None:
        """Block until the TUI is past boot and ready for input.

        Looks for the canonical welcome banner. If we never see it,
        the TUI either failed to launch or printed an error before
        the prompt_toolkit screen took over.
        """
        self._child.expect(
            r"(Welcome to Infinidev|Type your instruction)",
            timeout=timeout,
        )

    def send(self, text: str) -> None:
        """Send a line of text to the TUI as if typed by the user."""
        # prompt_toolkit's input handler usually accepts ``\r`` for
        # commit; ``sendline`` appends ``\r\n`` which is fine.
        self._child.sendline(text)

    def wait_for(self, pattern: str, timeout: float = 30.0) -> str:
        """Wait until *pattern* (regex) appears in the output buffer.

        Returns the matched text. Raises ``pexpect.TIMEOUT`` if it
        never shows up — let the test catch that and fail with a
        readable message.
        """
        self._child.expect(pattern, timeout=timeout)
        return self._child.after if isinstance(self._child.after, str) else ""

    def read_text(self) -> str:
        """Return everything seen so far on the TTY, ANSI-stripped."""
        # ``before`` holds everything BEFORE the last successful match.
        # For free-form inspection we want both before + after.
        raw = (self._child.before or "") + (self._child.after or "")
        return strip_ansi(raw if isinstance(raw, str) else "")

    @property
    def child(self) -> pexpect.spawn:
        """Escape hatch — direct access to the spawned child."""
        return self._child


@contextmanager
def temporary_workspace():
    """Convenience context manager for tests that don't care about cwd."""
    d = tempfile.mkdtemp(prefix="infinidev_ws_")
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
