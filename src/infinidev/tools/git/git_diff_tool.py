"""Tool for viewing Git diffs."""

import os
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import (
    GitToolError,
    resolve_git_cwd,
    run_git,
)
from infinidev.tools.git.git_diff_input import GitDiffInput


class GitDiffTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "git_diff"
    description: str = (
        "Show Git diff of changes. Can diff against a branch, "
        "show staged changes, or diff a specific file."
    )
    args_schema: Type[BaseModel] = GitDiffInput

    def _run(
        self,
        path: str | None = None,
        branch: str | None = None,
        file: str | None = None,
        staged: bool = False,
    ) -> str:
        try:
            cwd = resolve_git_cwd(self, path)
        except GitToolError as e:
            return self._error(str(e))

        cmd = ["git", "diff"]

        if staged:
            cmd.append("--cached")
        elif branch:
            cmd.append(branch)

        if file:
            cmd.extend(["--", file])

        if self._is_pod_mode():
            return self._run_in_pod(cmd, cwd)

        try:
            result = run_git(cmd, cwd=cwd, timeout=30, check=True)
        except GitToolError as e:
            return self._error(str(e))

        output = result.stdout
        if not output.strip():
            return "No differences found."

        return output

    def _run_in_pod(self, cmd: list[str], host_cwd: str) -> str:
        """Execute git diff inside the agent's pod."""
        workspace = self.workspace_path
        cwd = None
        if workspace:
            workspace_real = os.path.realpath(workspace)
            if os.path.commonpath((workspace_real, host_cwd)) == workspace_real:
                cwd = os.path.relpath(host_cwd, workspace_real)
        try:
            r = self._exec_in_pod(cmd, cwd=cwd, timeout=30)
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        if r.exit_code != 0:
            return self._error(f"Git diff failed: {r.stderr.strip()}")

        if not r.stdout.strip():
            return "No differences found."
        return r.stdout

