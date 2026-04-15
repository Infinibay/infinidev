"""System clipboard integration for the TUI.

Copies text to the system clipboard using native tools (pbcopy, wl-copy,
xclip, xsel) with an OSC 52 terminal escape fallback — no external Python
dependencies required.
"""

from __future__ import annotations

import base64
import logging
import os
import shutil
import subprocess
import sys

log = logging.getLogger("infinidev.clipboard")


def _is_ssh() -> bool:
    """Heuristic: are we in an SSH session (no local display)?"""
    if os.environ.get("SSH_CONNECTION"):
        return True
    if os.environ.get("SSH_TTY"):
        return True
    return False


def _in_screen() -> bool:
    """Check if we are inside GNU screen."""
    term = os.environ.get("TERM", "")
    return term.startswith("screen")


def _in_tmux() -> bool:
    """Check if we are inside tmux."""
    return bool(os.environ.get("TMUX"))


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard.

    Tries native tools (pbcopy, wl-copy, xclip, xsel), then falls back
    to the OSC 52 terminal escape sequence (supported by most modern
    terminal emulators, including over SSH with tmux/screen).

    In SSH sessions without a local display, native X/Wayland tools are
    skipped to avoid multi-second timeouts.

    Returns True on success, False if nothing worked.
    """
    # In SSH without a local display, skip tools that need X/Wayland
    # to avoid 3-second timeouts on each attempt.
    ssh = _is_ssh()

    for cmd, args, needs_display in _CLIPBOARD_COMMANDS:
        if ssh and needs_display:
            continue
        if shutil.which(cmd) is not None:
            try:
                subprocess.run(
                    [cmd, *args],
                    input=text.encode(),
                    check=True,
                    timeout=3,
                )
                log.debug("copied via %s (%d chars)", cmd, len(text))
                return True
            except (subprocess.SubprocessError, OSError) as e:
                log.debug("failed %s: %s", cmd, e)
                continue

    log.debug("no native clipboard tool found, trying OSC 52")
    ok = _osc52_copy(text)
    log.debug("OSC 52 result: %s", ok)
    return ok


def _osc52_copy(text: str) -> bool:
    """Write an OSC 52 escape sequence to the terminal.

    Handles tmux and GNU screen wrapping automatically.
    """
    try:
        encoded = base64.b64encode(text.encode()).decode()
        seq = f"\033]52;c;{encoded}\a"

        # Wrap for multiplexers
        if _in_tmux():
            # tmux needs DCS passthrough: \ePtmux;\e<seq>\e\\
            seq = f"\033Ptmux;\033{seq}\033\\"
        elif _in_screen():
            # GNU screen DCS passthrough: \eP<seq>\e\\
            seq = f"\033P{seq}\033\\"

        payload = seq.encode()

        # Try /dev/tty first (works even when stdout is redirected)
        try:
            tty_fd = os.open("/dev/tty", os.O_WRONLY | os.O_NOCTTY)
            try:
                os.write(tty_fd, payload)
                # Drain the fd to ensure the kernel buffer is flushed
                try:
                    import fcntl
                    fcntl.tcdrain(tty_fd)
                except (OSError, AttributeError):
                    pass
                return True
            finally:
                os.close(tty_fd)
        except OSError:
            pass

        # Fallback: write to the original stdout fd (before any Python redirect)
        try:
            fd = sys.__stdout__.fileno()
            os.write(fd, payload)
            try:
                import fcntl
                fcntl.tcdrain(fd)
            except (OSError, AttributeError):
                pass
            return True
        except (OSError, AttributeError, ValueError):
            pass

        return False
    except Exception:
        return False


# (command, extra_args, needs_display_server) — tried in order
_CLIPBOARD_COMMANDS: list[tuple[str, list[str], bool]] = [
    ("pbcopy", [], False),                                    # macOS
    ("wl-copy", [], True),                                    # Wayland
    ("xclip", ["-selection", "clipboard"], True),             # X11
    ("xsel", ["--clipboard", "--input"], True),               # X11
]
