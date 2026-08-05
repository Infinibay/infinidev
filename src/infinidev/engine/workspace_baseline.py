"""Task-start workspace snapshots for tool-independent change detection."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass


_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_EXCLUDED_DIRS = {".git", ".infinidev", ".venv", "node_modules", "__pycache__"}


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class WorkspaceFileState:
    """Content identity and optional text captured at task start."""

    digest: str
    text: str | None


class WorkspaceBaseline:
    """A bounded snapshot used to discover changes regardless of tool path."""

    def __init__(self, root: str, files: dict[str, WorkspaceFileState], *, git: bool) -> None:
        self.root = os.path.realpath(root)
        self.files = files
        self.git = git

    @classmethod
    def capture(cls, root: str | None) -> "WorkspaceBaseline | None":
        """Capture tracked/untracked workspace files, or return None without a root."""

        if not root or not os.path.isdir(root):
            return None
        real_root = os.path.realpath(root)
        paths, git = _workspace_paths(real_root)
        return cls(real_root, _read_states(real_root, paths), git=git)

    def current_states(self) -> dict[str, WorkspaceFileState]:
        """Read the same workspace domain at task completion."""

        paths, _ = _workspace_paths(self.root, prefer_git=self.git)
        return _read_states(self.root, paths)

    def changed_paths(self) -> list[str]:
        """Relative paths whose final content differs from task start."""

        current = self.current_states()
        return sorted(
            path
            for path in set(self.files) | set(current)
            if self.files.get(path) != current.get(path)
        )


def _workspace_paths(root: str, *, prefer_git: bool = True) -> tuple[list[str], bool]:
    if prefer_git:
        try:
            probe = subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if probe.returncode == 0 and probe.stdout.strip() == "true":
                listed = subprocess.run(
                    ["git", "ls-files", "-co", "--exclude-standard", "-z"],
                    cwd=root,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if listed.returncode == 0:
                    git_paths = {
                        item.decode(errors="surrogateescape")
                        for item in listed.stdout.split(b"\0")
                        if item
                    }
                    # Git's exclude rules intentionally hide generated and
                    # ignored files. They still belong to the task's observed
                    # workspace state: a build script can create or mutate one,
                    # and review/rollback must see it. Preserve Git's accurate
                    # tracked set while unioning a bounded filesystem walk.
                    return sorted(git_paths | set(_walk_workspace(root))), True
        except (OSError, subprocess.SubprocessError):
            pass

    return _walk_workspace(root), False


def _walk_workspace(root: str) -> list[str]:
    """List the bounded workspace domain, including ignored generated files."""

    paths: list[str] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = [directory for directory in dirs if directory not in _EXCLUDED_DIRS]
        for filename in files:
            absolute = os.path.join(current_root, filename)
            paths.append(os.path.relpath(absolute, root))
    return paths


def _read_states(root: str, paths: list[str]) -> dict[str, WorkspaceFileState]:
    states: dict[str, WorkspaceFileState] = {}
    for relative in paths:
        absolute = os.path.realpath(os.path.join(root, relative))
        if not (absolute == root or absolute.startswith(root + os.sep)):
            continue
        try:
            if not os.path.isfile(absolute):
                continue
            with open(absolute, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        text = (
            data.decode("utf-8", errors="replace")
            if len(data) <= _MAX_CAPTURE_BYTES
            else None
        )
        states[relative] = WorkspaceFileState(digest=_digest(data), text=text)
    return states
