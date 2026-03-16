"""Tool for viewing Git status."""

import subprocess
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool


class GitStatusInput(BaseModel):
    pass


class GitStatusTool(InfinibayBaseTool):
    name: str = "git_status"
    description: str = (
        "Show the current Git status including modified, staged, and untracked files."
    )
    args_schema: Type[BaseModel] = GitStatusInput

    def _run(self) -> str:
        if self._is_pod_mode():
            return self._run_in_pod()

        cwd = self._git_cwd
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=15, cwd=cwd,
            )
            if result.returncode != 0:
                return self._error(f"Git status failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            return self._error("Git status timed out")
        except FileNotFoundError:
            return self._error("Git is not installed or not in PATH")

        raw = result.stdout.rstrip("\n")
        lines = raw.split("\n") if raw else []

        staged = []
        modified = []
        untracked = []

        for line in lines:
            if len(line) < 3:
                continue
            index_status = line[0]
            worktree_status = line[1]
            file_path = line[3:]

            if index_status in ("A", "M", "D", "R", "C"):
                staged.append({"status": index_status, "file": file_path})
            if worktree_status in ("M", "D"):
                modified.append({"status": worktree_status, "file": file_path})
            if index_status == "?" and worktree_status == "?":
                untracked.append(file_path)

        # Get current branch
        try:
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=10, cwd=cwd,
            )
            branch = branch_result.stdout.strip() if branch_result.returncode == 0 else "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            branch = "unknown"

        return self._success({
            "branch": branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "clean": len(lines) == 0,
        })

    def _run_in_pod(self) -> str:
        """Execute git status inside the agent's pod."""
        try:
            r = self._exec_in_pod(["git", "status", "--porcelain"], timeout=15)
            if r.exit_code != 0:
                return self._error(f"Git status failed: {r.stderr.strip()}")
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        raw = r.stdout.rstrip("\n")
        lines = raw.split("\n") if raw else []

        staged = []
        modified = []
        untracked = []

        for line in lines:
            if len(line) < 3:
                continue
            index_status = line[0]
            worktree_status = line[1]
            file_path = line[3:]

            if index_status in ("A", "M", "D", "R", "C"):
                staged.append({"status": index_status, "file": file_path})
            if worktree_status in ("M", "D"):
                modified.append({"status": worktree_status, "file": file_path})
            if index_status == "?" and worktree_status == "?":
                untracked.append(file_path)

        # Get branch
        try:
            br = self._exec_in_pod(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10,
            )
            branch = br.stdout.strip() if br.exit_code == 0 else "unknown"
        except RuntimeError:
            branch = "unknown"

        return self._success({
            "branch": branch,
            "staged": staged,
            "modified": modified,
            "untracked": untracked,
            "clean": len(lines) == 0,
        })
