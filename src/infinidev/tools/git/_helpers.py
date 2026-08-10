"""Shared helpers for git tools."""

from __future__ import annotations

import os
import subprocess
from typing import Any


class GitToolError(Exception):
    """Raised by run_git for timeout, missing binary, or failed command."""


def _inside(root: str, candidate: str) -> bool:
    try:
        return os.path.commonpath((root, candidate)) == root
    except ValueError:
        return False


def _has_git_ancestor(path: str) -> bool:
    current = os.path.realpath(path)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return True
        parent = os.path.dirname(current)
        if parent == current:
            return False
        current = parent


def resolve_git_cwd(tool: Any, path: str | None = None) -> str:
    """Resolve an explicit, bound, or unambiguous nested Git directory."""
    workspace = tool.workspace_path
    workspace_real = (
        os.path.realpath(os.path.expanduser(workspace)) if workspace else None
    )

    if path:
        candidate = os.path.realpath(os.path.expanduser(tool._resolve_path(path)))
        if workspace_real and not _inside(workspace_real, candidate):
            raise GitToolError(
                f"Git path escapes the workspace: {path}. Choose a repository under "
                f"{workspace_real}."
            )
        if not os.path.isdir(candidate):
            raise GitToolError(f"Git path is not a directory: {path}")
        if not _has_git_ancestor(candidate):
            raise GitToolError(f"Path is not inside a Git repository: {path}")
        return candidate

    repository = tool.repository_path
    if repository and os.path.isdir(repository) and _has_git_ancestor(repository):
        return os.path.realpath(repository)

    if workspace_real and os.path.isdir(workspace_real):
        if _has_git_ancestor(workspace_real):
            return workspace_real
        nested = sorted(
            os.path.realpath(os.path.join(workspace_real, name))
            for name in os.listdir(workspace_real)
            if os.path.isdir(os.path.join(workspace_real, name))
            and os.path.exists(os.path.join(workspace_real, name, ".git"))
        )
        if len(nested) == 1:
            return nested[0]
        if len(nested) > 1:
            choices = ", ".join(os.path.basename(item) for item in nested[:6])
            raise GitToolError(
                "Multiple Git repositories are available "
                f"({choices}). Re-call the tool with path='<repository>'."
            )

    raise GitToolError(
        "No Git repository target is available. Re-call the tool with "
        "path='<repository>' relative to the workspace."
    )


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
