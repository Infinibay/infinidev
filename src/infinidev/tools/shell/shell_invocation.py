"""Build truthful shell subprocess invocations."""

from __future__ import annotations

import os
import re
import shutil


_LEADING_CD_RE = re.compile(
    r"^\s*cd[ \t]+(?P<path>'[^']*'|\"[^\"]*\"|[^\s;&|`$<>]+)"
    r"[ \t]*&&\s*(?P<command>\S.*)$",
    re.DOTALL,
)


def split_leading_cd(command: str) -> tuple[str, str] | None:
    """Parse a strict literal ``cd PATH && COMMAND`` shell prelude.

    Leave directory words requiring shell expansion or builtin semantics in
    the original command. The caller still checks the remaining command and
    directory under its usual permissions.
    """
    match = _LEADING_CD_RE.match(command)
    if match is None:
        return None
    word = match.group("path")
    if word.startswith("'") and word.endswith("'"):
        path = word[1:-1]
        if "'" in path:
            return None
    elif word.startswith('"') and word.endswith('"'):
        path = word[1:-1]
        if any(char in path for char in '"$`\\'):
            return None
    else:
        path = word
        if word.startswith(("~", "#")) or any(char in word for char in "'\"\\*?[]{}()"):
            return None

    if not path or path.startswith("-"):
        return None
    return match.group("command").strip(), path


def resolve_command_cwd(
    cwd: str | None, *, repository_path: str | None, workspace_path: str | None,
) -> str:
    """Resolve a command directory against the active repository or workspace."""
    if cwd and os.path.isabs(cwd):
        return cwd
    base = repository_path or workspace_path or os.getcwd()
    return os.path.join(base, cwd) if cwd else base


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


__all__ = ["resolve_command_cwd", "shell_invocation", "split_leading_cd"]
