"""Build truthful shell subprocess invocations."""

from __future__ import annotations

import os
import shutil


def shell_invocation(command: str) -> tuple[str | list[str], bool]:
    """Return subprocess args that propagate failure from any pipeline stage.

    ``shell=True`` uses ``/bin/sh`` on POSIX, whose pipeline status is only
    the last command. That turns ``pytest | tail`` green when pytest fails.
    Bash's ``pipefail`` preserves ordinary shell syntax while making the
    subprocess exit code represent the complete pipeline. On hosts without
    bash, retain Python's platform shell behavior instead of refusing every
    command.
    """
    bash = shutil.which("bash") if os.name == "posix" else None
    if bash is None:
        return command, True
    return [bash, "-o", "pipefail", "-c", command], False


__all__ = ["shell_invocation"]
