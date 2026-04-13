"""System clipboard integration for the TUI.

Copies text to the system clipboard using native tools (xclip, xsel,
wl-copy) with an OSC 52 terminal escape fallback — no external Python
dependencies required.
"""

from __future__ import annotations

import base64
import os
import shutil
import subprocess
import sys


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard.

    Tries Wayland (wl-copy), then X11 (xclip, xsel), then falls back
    to the OSC 52 terminal escape sequence (supported by most modern
    terminal emulators).
    Returns True on success, False if nothing worked.
    """
    import logging
    log = logging.getLogger("infinidev.copy_debug")

    for cmd, args in _CLIPBOARD_COMMANDS:
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
    # Fallback: OSC 52 escape sequence — tells the terminal emulator
    # to set its clipboard.  Works in kitty, alacritty, iTerm2, foot,
    # WezTerm, Windows Terminal, and many others.
    ok = _osc52_copy(text)
    log.debug("OSC 52 result: %s", ok)
    return ok


def _osc52_copy(text: str) -> bool:
    """Write an OSC 52 escape sequence to the terminal."""
    try:
        encoded = base64.b64encode(text.encode()).decode()
        # \033]52;c;<base64>\a  — 'c' = system clipboard
        seq = f"\033]52;c;{encoded}\a"
        payload = seq.encode()
        # Try /dev/tty first (works even when stdout is redirected)
        try:
            tty_fd = os.open("/dev/tty", os.O_WRONLY)
            try:
                os.write(tty_fd, payload)
                return True
            finally:
                os.close(tty_fd)
        except OSError:
            pass
        # Fallback: write to the original stdout fd (before any Python redirect)
        try:
            os.write(sys.__stdout__.fileno(), payload)
            return True
        except (OSError, AttributeError, ValueError):
            pass
        return False
    except Exception:
        return False


# (command, extra_args) — tried in order
_CLIPBOARD_COMMANDS: list[tuple[str, list[str]]] = [
    ("wl-copy", []),
    ("xclip", ["-selection", "clipboard"]),
    ("xsel", ["--clipboard", "--input"]),
]
