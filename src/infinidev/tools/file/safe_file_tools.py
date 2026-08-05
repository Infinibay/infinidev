"""Recoverable file lifecycle and multi-edit tools."""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import time
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file._helpers import atomic_write, guard_file_access


class FileReplacement(BaseModel):
    """One exact replacement applied sequentially in a patch."""

    old_string: str = Field(min_length=1, description="Exact text to replace.")
    new_string: str = Field(description="Replacement text; empty deletes the match.")
    replace_all: bool = Field(default=False)


class DeleteFileInput(BaseModel):
    file_path: str
    rationale: str = ""


class MoveFileInput(BaseModel):
    source_path: str
    destination_path: str
    overwrite: bool = False
    rationale: str = ""


class FilePatchInput(BaseModel):
    file_path: str
    replacements: list[FileReplacement] = Field(min_length=1)
    rationale: str = ""


class RollbackTaskChangesInput(BaseModel):
    file_paths: list[str] | None = Field(
        default=None,
        description="Changed paths to restore; omit to restore every task-created change.",
    )
    rationale: str = ""


def _apply_replacements(content: str, replacements: list[FileReplacement]) -> str:
    result = content
    for index, replacement in enumerate(replacements, start=1):
        occurrences = result.count(replacement.old_string)
        if occurrences == 0:
            raise ValueError(f"replacement {index}: old_string not found")
        if occurrences > 1 and not replacement.replace_all:
            raise ValueError(
                f"replacement {index}: old_string appears {occurrences} times; "
                "add context or set replace_all=true"
            )
        result = (
            result.replace(replacement.old_string, replacement.new_string)
            if replacement.replace_all
            else result.replace(replacement.old_string, replacement.new_string, 1)
        )
    return result


class DeleteFileTool(InfinibayBaseTool):
    name: str = "delete_file"
    description: str = (
        "Remove one workspace file recoverably. The file is moved into "
        ".infinidev/trash and the result returns its recovery path. Refuses directories."
    )
    args_schema: Type[BaseModel] = DeleteFileInput

    def _run(self, file_path: str, rationale: str = "") -> str:
        del rationale
        path = self._resolve_path(os.path.expanduser(file_path))
        if error := guard_file_access(self, path, "edit_file"):
            return error
        if not os.path.isfile(path):
            return self._error(f"Not a file: {path}")

        workspace = self.workspace_path or os.path.dirname(path)
        trash = os.path.join(workspace, ".infinidev", "trash")
        os.makedirs(trash, exist_ok=True)
        stamp = f"{time.time_ns()}-{os.path.basename(path)}"
        recovery_path = os.path.join(trash, stamp)
        try:
            shutil.move(path, recovery_path)
        except OSError as exc:
            return self._error(f"Could not move file to recovery trash: {exc}")
        return self._success({
            "file_path": path,
            "action": "deleted",
            "recovery_path": recovery_path,
        })


class MoveFileTool(InfinibayBaseTool):
    name: str = "move_file"
    description: str = (
        "Move or rename one workspace file. Both paths are permission-checked; "
        "the destination is never overwritten unless overwrite=true."
    )
    args_schema: Type[BaseModel] = MoveFileInput

    def _run(
        self,
        source_path: str,
        destination_path: str,
        overwrite: bool = False,
        rationale: str = "",
    ) -> str:
        del rationale
        source = self._resolve_path(os.path.expanduser(source_path))
        destination = self._resolve_path(os.path.expanduser(destination_path))
        for path in (source, destination):
            if error := guard_file_access(self, path, "edit_file"):
                return error
        if not os.path.isfile(source):
            return self._error(f"Source is not a file: {source}")
        if os.path.exists(destination) and not overwrite:
            return self._error(
                f"Destination exists: {destination}. Set overwrite=true only if intended."
            )
        os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
        try:
            os.replace(source, destination) if overwrite else shutil.move(source, destination)
        except OSError as exc:
            return self._error(f"Move failed: {exc}")
        return self._success({
            "source_path": source,
            "destination_path": destination,
            "action": "moved",
            "overwrote": overwrite,
        })


class ApplyFilePatchTool(InfinibayBaseTool):
    name: str = "apply_file_patch"
    description: str = (
        "Apply multiple exact replacements to one file atomically. Every replacement "
        "is validated against the evolving in-memory result before a single write occurs."
    )
    args_schema: Type[BaseModel] = FilePatchInput

    def _run(
        self,
        file_path: str,
        replacements: list[FileReplacement] | list[dict],
        rationale: str = "",
    ) -> str:
        del rationale
        path = self._resolve_path(os.path.expanduser(file_path))
        if error := guard_file_access(self, path, "edit_file"):
            return error
        if not os.path.isfile(path):
            return self._error(f"Not a file: {path}")
        try:
            with open(path, "r", encoding="utf-8") as handle:
                before = handle.read()
            parsed = [
                item if isinstance(item, FileReplacement) else FileReplacement.model_validate(item)
                for item in replacements
            ]
            after = _apply_replacements(before, parsed)
            if after == before:
                return self._error("Patch produced no change")
            atomic_write(path, after)
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error(f"Patch rejected; nothing changed: {exc}")
        return self._success({
            "file_path": path,
            "action": "modified",
            "replacements": len(parsed),
            "before_hash": hashlib.sha256(before.encode()).hexdigest()[:16],
            "after_hash": hashlib.sha256(after.encode()).hexdigest()[:16],
        })


class PreviewChangesTool(InfinibayBaseTool):
    name: str = "preview_changes"
    description: str = (
        "Preview the unified diff for a proposed multi-replacement patch without "
        "writing the file. Uses the same validation as apply_file_patch."
    )
    args_schema: Type[BaseModel] = FilePatchInput
    is_read_only: bool = True

    def _run(
        self,
        file_path: str,
        replacements: list[FileReplacement] | list[dict],
        rationale: str = "",
    ) -> str:
        del rationale
        path = self._resolve_path(os.path.expanduser(file_path))
        if error := self._validate_sandbox_path(path):
            return self._error(error)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                before = handle.read()
            parsed = [
                item if isinstance(item, FileReplacement) else FileReplacement.model_validate(item)
                for item in replacements
            ]
            after = _apply_replacements(before, parsed)
        except (OSError, UnicodeError, ValueError) as exc:
            return self._error(f"Preview rejected: {exc}")
        diff = "\n".join(difflib.unified_diff(
            before.splitlines(), after.splitlines(),
            fromfile=f"a/{file_path}", tofile=f"b/{file_path}", lineterm="",
        ))
        return diff or "(no change)"


class RollbackTaskChangesTool(InfinibayBaseTool):
    name: str = "rollback_task_changes"
    description: str = (
        "Restore files to their exact task-start contents using the engine baseline. "
        "Never uses HEAD, so pre-existing user edits are preserved."
    )
    args_schema: Type[BaseModel] = RollbackTaskChangesInput

    def _run(
        self,
        file_paths: list[str] | None = None,
        rationale: str = "",
    ) -> str:
        del rationale
        from infinidev.tools.base.context import get_context_for_agent

        agent_id = getattr(self, "_bound_agent_id", None) or self.agent_id
        context = get_context_for_agent(agent_id) if agent_id else None
        tracker = context.file_tracker if context else None
        baseline = getattr(tracker, "baseline", None)
        if tracker is None or baseline is None:
            return self._error("No task-start workspace baseline is available")

        changed = tracker.get_all_paths()
        requested = {
            self._resolve_path(os.path.expanduser(path)) for path in (file_paths or changed)
        }
        targets = [path for path in changed if path in requested]
        if not targets:
            return self._error("No task-created changes matched the requested paths")

        staged: list[tuple[str, str | None]] = []
        for path in targets:
            if error := guard_file_access(self, path, "edit_file"):
                return error
            relative = os.path.relpath(path, baseline.root)
            before_state = baseline.files.get(relative)
            if before_state is not None and before_state.text is None:
                return self._error(
                    f"Cannot restore oversized baseline content safely: {path}"
                )
            staged.append((path, before_state.text if before_state else None))

        restored: list[str] = []
        removed: list[str] = []
        try:
            for path, original in staged:
                if original is None:
                    if os.path.exists(path):
                        os.unlink(path)
                    removed.append(path)
                else:
                    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
                    atomic_write(path, original)
                    restored.append(path)
        except OSError as exc:
            return self._error(f"Rollback failed: {exc}")
        return self._success({"restored": restored, "removed_new_files": removed})


__all__ = [
    "ApplyFilePatchTool",
    "DeleteFileTool",
    "MoveFileTool",
    "PreviewChangesTool",
    "RollbackTaskChangesTool",
]
