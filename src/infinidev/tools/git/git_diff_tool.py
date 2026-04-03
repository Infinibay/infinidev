"""Tool for viewing Git diffs."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError
from infinidev.tools.git.git_diff_input import GitDiffInput


class GitDiffTool(InfinibayBaseTool):
    name: str = "git_diff"
    description: str = (
        "Show Git diff of changes. Can diff against a branch, "
        "show staged changes, or diff a specific file."
    )
    args_schema: Type[BaseModel] = GitDiffInput

    def _run(
        self,
        branch: str | None = None,
        file: str | None = None,
        staged: bool = False,
    ) -> str:
        cmd = ["git", "diff"]

        if staged:
            cmd.append("--cached")
        elif branch:
            cmd.append(branch)

        if file:
            cmd.extend(["--", file])

        if self._is_pod_mode():
            return self._run_in_pod(cmd)

        try:
            result = run_git(cmd, cwd=self._git_cwd, timeout=30, check=True)
        except GitToolError as e:
            return self._error(str(e))

        output = result.stdout
        if not output.strip():
            return "No differences found."

        return output

    def _run_in_pod(self, cmd: list[str]) -> str:
        """Execute git diff inside the agent's pod."""
        try:
            r = self._exec_in_pod(cmd, timeout=30)
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        if r.exit_code != 0:
            return self._error(f"Git diff failed: {r.stderr.strip()}")

        if not r.stdout.strip():
            return "No differences found."
        return r.stdout

