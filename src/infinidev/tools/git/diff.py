"""Tool for viewing Git diffs."""

import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class GitDiffInput(BaseModel):
    branch: str | None = Field(
        default=None, description="Branch to diff against (e.g. 'main')"
    )
    file: str | None = Field(
        default=None, description="Specific file to diff"
    )
    staged: bool = Field(
        default=False, description="Show only staged changes"
    )


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
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30,
                cwd=self._git_cwd,
            )
            if result.returncode != 0:
                return self._error(f"Git diff failed: {result.stderr.strip()}")
        except subprocess.TimeoutExpired:
            return self._error("Git diff timed out")
        except FileNotFoundError:
            return self._error("Git is not installed or not in PATH")

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
