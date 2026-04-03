"""Shared helpers for git tools."""

from __future__ import annotations

import subprocess


class GitToolError(Exception):
    """Raised by run_git for timeout, missing binary, or failed command."""


def run_git(
    args: list[str],
    *,
    cwd: str | None = None,
    timeout: int = 15,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a git command with standard error handling.

    Raises GitToolError on timeout or missing git binary.
    If *check* is True, also raises GitToolError on non-zero exit code.
    """
    try:
        result = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired:
        cmd_str = " ".join(args[:3])
        raise GitToolError(f"Git operation timed out: {cmd_str}")
    except FileNotFoundError:
        raise GitToolError("Git is not installed or not in PATH")

    if check and result.returncode != 0:
        raise GitToolError(result.stderr.strip() or f"git exited with code {result.returncode}")

    return result
