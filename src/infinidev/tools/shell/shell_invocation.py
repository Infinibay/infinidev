"""Build truthful shell subprocess invocations."""

from __future__ import annotations

import os
import re
import shlex
import shutil


_LEADING_CD_RE = re.compile(
    r"^\s*cd\s+(?P<path>'[^']*'|\"[^\"]*\"|[^\s;&|`$<>]+)"
    r"\s*&&\s*(?P<command>.+)$",
    re.DOTALL,
)


def split_leading_cd(command: str) -> tuple[str, str] | None:
    """Parse a strict literal ``cd PATH && COMMAND`` shell prelude.

    Substitutions, redirects, and nested control operators are deliberately
    excluded. Callers remain responsible for validating whether the parsed
    directory is an allowed target.
    """
    match = _LEADING_CD_RE.match(command)
    if match is None:
        return None
    try:
        path_tokens = shlex.split(match.group("path"))
    except ValueError:
        return None
    if len(path_tokens) != 1:
        return None
    return match.group("command").strip(), path_tokens[0]


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
