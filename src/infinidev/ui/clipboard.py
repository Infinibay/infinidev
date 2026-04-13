"""System clipboard integration for the TUI.

Copies text to the system clipboard using native tools (xclip, xsel,
wl-copy) — no external Python dependencies required.
"""

from __future__ import annotations

import shutil
import subprocess


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the system clipboard.

    Tries Wayland (wl-copy), then X11 (xclip, xsel) in order.
    Returns True on success, False if no clipboard tool is available.
    """
    for cmd, args in _CLIPBOARD_COMMANDS:
        if shutil.which(cmd) is not None:
            try:
                subprocess.run(
                    [cmd, *args],
                    input=text.encode(),
                    check=True,
                    timeout=3,
                )
                return True
            except (subprocess.SubprocessError, OSError):
                continue
    return False


# (command, extra_args) — tried in order
_CLIPBOARD_COMMANDS: list[tuple[str, list[str]]] = [
    ("wl-copy", []),
    ("xclip", ["-selection", "clipboard"]),
    ("xsel", ["--clipboard", "--input"]),
]
