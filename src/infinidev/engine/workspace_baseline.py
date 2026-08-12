"""Task-start workspace snapshots for tool-independent change detection."""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass


_MAX_CAPTURE_BYTES = 2 * 1024 * 1024
_MAX_TOTAL_CAPTURE_BYTES = 64 * 1024 * 1024
_MAX_HASH_BYTES_PER_FILE = 8 * 1024 * 1024
_MAX_TOTAL_HASH_BYTES = 512 * 1024 * 1024
_HASH_CHUNK_BYTES = 1024 * 1024
_MAX_FILES = 100_000
_EXCLUDED_DIRS = {
    ".cache",
    ".git",
    ".infinidev",
    ".ken",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "node_modules",
    "target",
}
_EXCLUDED_DIR_PREFIXES = ("pytest-cache-files-",)


def _metadata_digest(stat_result: os.stat_result) -> str:
    """Bounded identity for files that are unsafe or too expensive to read."""

    return (
        f"stat-v1:{stat_result.st_mode}:{stat_result.st_size}:"
        f"{stat_result.st_mtime_ns}:{stat_result.st_ctime_ns}"
    )


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
                tracked = subprocess.run(
                    ["git", "ls-files", "-c", "-z"],
                    cwd=root,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                untracked = subprocess.run(
                    ["git", "ls-files", "-o", "--exclude-standard", "-z"],
                    cwd=root,
                    capture_output=True,
                    timeout=10,
                    check=False,
                )
                if tracked.returncode == 0 and untracked.returncode == 0:
                    tracked_paths = sorted({
                        item.decode(errors="surrogateescape")
                        for item in tracked.stdout.split(b"\0")
                        if item
                    })
                    untracked_paths = sorted({
                        item.decode(errors="surrogateescape")
                        for item in untracked.stdout.split(b"\0")
                        if item and not _excluded_generated_path(
                            item.decode(errors="surrogateescape")
                        )
                    })
                    # Git's exclude rules intentionally hide generated and
                    # ignored files. They still belong to the task's observed
                    # workspace state when they are not runtime-private. Keep
                    # tracked files even below an excluded directory, omit
                    # untracked caches/private state, then add the bounded walk.
                    return _merge_paths(
                        tracked_paths,
                        untracked_paths,
                        _walk_workspace(root),
                    ), True
        except (OSError, subprocess.SubprocessError):
            pass

    return _walk_workspace(root), False


def _walk_workspace(root: str) -> list[str]:
    """List the bounded workspace domain, including ignored generated files."""

    paths: list[str] = []
    for current_root, dirs, files in os.walk(root):
        dirs[:] = sorted(
            directory
            for directory in dirs
            if not _excluded_generated_dir(directory)
        )
        for filename in sorted(files):
            absolute = os.path.join(current_root, filename)
            paths.append(os.path.relpath(absolute, root))
            if len(paths) >= _MAX_FILES:
                return paths
    return paths


def _excluded_generated_dir(name: str) -> bool:
    return name in _EXCLUDED_DIRS or name.startswith(_EXCLUDED_DIR_PREFIXES)


def _excluded_generated_path(path: str) -> bool:
    return any(_excluded_generated_dir(part) for part in path.split("/"))


def _merge_paths(*path_groups: list[str]) -> list[str]:
    """Return a stable, bounded union while prioritising Git-visible files."""

    merged: list[str] = []
    seen: set[str] = set()
    for paths in path_groups:
        for path in paths:
            if path in seen:
                continue
            seen.add(path)
            merged.append(path)
            if len(merged) >= _MAX_FILES:
                return merged
    return merged


def _read_content_state(
    path: str,
    *,
    max_hash_bytes: int,
    max_text_bytes: int,
) -> tuple[WorkspaceFileState | None, int, int]:
    """Hash one bounded file in chunks and optionally retain its text."""

    hasher = hashlib.sha256()
    text_buffer = bytearray() if max_text_bytes >= 0 else None
    bytes_read = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(min(_HASH_CHUNK_BYTES, max_hash_bytes - bytes_read + 1))
            if not chunk:
                break
            bytes_read += len(chunk)
            if bytes_read > max_hash_bytes:
                return None, bytes_read, 0
            hasher.update(chunk)
            if text_buffer is not None:
                if len(text_buffer) + len(chunk) <= max_text_bytes:
                    text_buffer.extend(chunk)
                else:
                    text_buffer = None

    text = (
        bytes(text_buffer).decode("utf-8", errors="replace")
        if text_buffer is not None
        else None
    )
    text_bytes = len(text_buffer) if text_buffer is not None else 0
    return WorkspaceFileState(digest=hasher.hexdigest(), text=text), bytes_read, text_bytes


def _read_states(root: str, paths: list[str]) -> dict[str, WorkspaceFileState]:
    states: dict[str, WorkspaceFileState] = {}
    remaining_hash_bytes = _MAX_TOTAL_HASH_BYTES
    remaining_text_bytes = _MAX_TOTAL_CAPTURE_BYTES
    for relative in paths[:_MAX_FILES]:
        absolute = os.path.realpath(os.path.join(root, relative))
        if not (absolute == root or absolute.startswith(root + os.sep)):
            continue
        try:
            if not os.path.isfile(absolute):
                continue
            stat_result = os.stat(absolute)
        except OSError:
            continue

        size = max(0, stat_result.st_size)
        if size > _MAX_HASH_BYTES_PER_FILE or size > remaining_hash_bytes:
            states[relative] = WorkspaceFileState(
                digest=_metadata_digest(stat_result),
                text=None,
            )
            continue

        max_text_bytes = (
            min(_MAX_CAPTURE_BYTES, remaining_text_bytes)
            if size <= _MAX_CAPTURE_BYTES and size <= remaining_text_bytes
            else -1
        )
        try:
            state, bytes_read, text_bytes = _read_content_state(
                absolute,
                max_hash_bytes=min(_MAX_HASH_BYTES_PER_FILE, remaining_hash_bytes),
                max_text_bytes=max_text_bytes,
            )
        except OSError:
            continue
        if state is None:
            try:
                stat_result = os.stat(absolute)
            except OSError:
                continue
            states[relative] = WorkspaceFileState(
                digest=_metadata_digest(stat_result),
                text=None,
            )
            continue

        states[relative] = state
        remaining_hash_bytes = max(0, remaining_hash_bytes - bytes_read)
        remaining_text_bytes = max(0, remaining_text_bytes - text_bytes)
    return states
