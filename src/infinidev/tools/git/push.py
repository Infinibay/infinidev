"""Tool for Git push operations (async-capable)."""

import asyncio
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.git._helpers import run_git, GitToolError


class GitPushInput(BaseModel):
    branch: str | None = Field(
        default=None, description="Branch to push. If None, pushes current branch."
    )
    force: bool = Field(default=False, description="Force push (use with caution)")


class GitPushTool(InfinibayBaseTool):
    name: str = "git_push"
    description: str = (
        "Push commits to the remote repository. "
        "Pushes the current branch by default."
    )
    args_schema: Type[BaseModel] = GitPushInput

    def _run(self, branch: str | None = None, force: bool = False) -> str:
        if self._is_pod_mode():
            return self._run_in_pod(branch, force)

        cwd = self._git_cwd
        try:
            # Verify origin remote exists before pushing
            check = run_git(["git", "remote", "get-url", "origin"], cwd=cwd, timeout=10)
            if check.returncode != 0:
                return self._error(
                    "No remote 'origin' configured. Cannot push without a "
                    "remote. Ask the user to configure one."
                )

            # Get current branch if not specified
            if branch is None:
                result = run_git(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, timeout=10, check=True,
                )
                branch = result.stdout.strip()

            cmd = ["git", "push", "-u", "origin", branch]
            if force:
                cmd.insert(2, "--force")

            result = run_git(cmd, cwd=cwd, timeout=settings.GIT_PUSH_TIMEOUT)
            if result.returncode != 0:
                stderr = result.stderr.strip()
                if "rejected" in stderr:
                    return self._error(
                        f"Push rejected (remote has new commits). "
                        f"Pull first or use force=True. Details: {stderr}"
                    )
                return self._error(f"Push failed: {stderr}")

        except GitToolError as e:
            return self._error(str(e))

        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus
            event_bus.emit(
                FlowEvent(
                    event_type="git_pushed",
                    project_id=self.project_id,
                    entity_type="branch",
                    entity_id=None,
                    data={
                        "branch": branch,
                        "remote": "origin",
                        "forced": force,
                        "agent_id": self.agent_id,
                    },
                )
            )
        except Exception:
            pass  # Don't fail the push if event emission fails

        self._log_tool_usage(f"Pushed {branch} to origin")
        return self._success({
            "branch": branch,
            "remote": "origin",
            "forced": force,
        })

    def _run_in_pod(self, branch: str | None, force: bool) -> str:
        """Execute git push inside the agent's pod."""
        try:
            if branch is None:
                r = self._exec_in_pod(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10,
                )
                if r.exit_code != 0:
                    return self._error("Failed to determine current branch")
                branch = r.stdout.strip()

            cmd = ["git", "push", "-u", "origin", branch]
            if force:
                cmd.insert(2, "--force")

            r = self._exec_in_pod(cmd, timeout=settings.GIT_PUSH_TIMEOUT)
            if r.exit_code != 0:
                stderr = r.stderr.strip()
                if "rejected" in stderr:
                    return self._error(
                        f"Push rejected (remote has new commits). "
                        f"Pull first or use force=True. Details: {stderr}"
                    )
                return self._error(f"Push failed: {stderr}")

        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus
            event_bus.emit(FlowEvent(
                event_type="git_pushed",
                project_id=self.project_id,
                entity_type="branch",
                entity_id=None,
                data={
                    "branch": branch,
                    "remote": "origin",
                    "forced": force,
                    "agent_id": self.agent_id,
                },
            ))
        except Exception:
            pass

        self._log_tool_usage(f"Pushed {branch} to origin (pod)")
        return self._success({"branch": branch, "remote": "origin", "forced": force})

    async def _arun(self, branch: str | None = None, force: bool = False) -> str:
        """Async push using asyncio subprocess."""
        async_cwd = self._git_cwd
        try:
            if branch is None:
                proc = await asyncio.create_subprocess_exec(
                    "git", "rev-parse", "--abbrev-ref", "HEAD",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=async_cwd,
                )
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=10)
                if proc.returncode != 0:
                    return self._error("Failed to determine current branch")
                branch = stdout.decode().strip()

            cmd = ["git", "push", "-u", "origin", branch]
            if force:
                cmd.insert(2, "--force")

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=async_cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=settings.GIT_PUSH_TIMEOUT
            )

            if proc.returncode != 0:
                err = stderr.decode().strip()
                return self._error(f"Push failed: {err}")

        except asyncio.TimeoutError:
            return self._error(f"Push timed out after {settings.GIT_PUSH_TIMEOUT}s")

        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus
            event_bus.emit(
                FlowEvent(
                    event_type="git_pushed",
                    project_id=self.project_id,
                    entity_type="branch",
                    entity_id=None,
                    data={
                        "branch": branch,
                        "remote": "origin",
                        "forced": force,
                        "agent_id": self.agent_id,
                    },
                )
            )
        except Exception:
            pass  # Don't fail the push if event emission fails

        self._log_tool_usage(f"Pushed {branch} to origin (async)")
        return self._success({"branch": branch, "remote": "origin", "forced": force})
