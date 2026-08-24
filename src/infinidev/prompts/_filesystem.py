"""Filesystem durability helpers shared by prompt-profile persistence."""

from __future__ import annotations

import os
from pathlib import Path


def fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on platforms that support directory handles."""
    if os.name == "nt":
        return

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
