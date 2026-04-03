"""Tool for Git commit operations."""

import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.git._helpers import run_git, GitToolError


class GitCommitInput(BaseModel):
    message: str = Field(..., description="Commit message")
    files: list[str] | None = Field(
        default=None,
        description="Specific files to stage. If None, stages all changes (git add -A).",
    )


class GitCommitTool(InfinibayBaseTool):
    name: str = "git_commit"
    description: str = (
        "Stage and commit changes to Git. "
        "Optionally specify files to stage, otherwise stages all changes."
    )
    args_schema: Type[BaseModel] = GitCommitInput

    def _run(self, message: str, files: list[str] | None = None) -> str:
        if self._is_pod_mode():
            return self._run_in_pod(message, files)

        cwd = self._git_cwd
        try:
            # Check if there are changes to commit
            status = run_git(["git", "status", "--porcelain"], cwd=cwd)
            if not status.stdout.strip():
                return self._error("No changes to commit")

            # Stage files
            if files:
                for f in files:
                    run_git(["git", "add", f], cwd=cwd, check=True)
            else:
                run_git(["git", "add", "-A"], cwd=cwd, check=True)

            # Commit
            result = run_git(["git", "commit", "-m", message], cwd=cwd, timeout=30, check=True)

            # Get commit hash
            hash_result = run_git(["git", "rev-parse", "HEAD"], cwd=cwd, timeout=10)
            commit_hash = hash_result.stdout.strip() if hash_result.returncode == 0 else None

            # Get current branch
            branch_result = run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd, timeout=10)
            branch_name = branch_result.stdout.strip() if branch_result.returncode == 0 else None

            # Update branch record in DB
            if branch_name and commit_hash:
                def _update_branch(conn: sqlite3.Connection):
                    conn.execute(
                        """UPDATE branches
                           SET last_commit_hash = ?, last_commit_at = CURRENT_TIMESTAMP
                           WHERE branch_name = ? AND status = 'active'""",
                        (commit_hash, branch_name),
                    )
                    conn.commit()

                try:
                    execute_with_retry(_update_branch)
                except Exception:
                    pass

                try:
                    from infinidev.flows.event_listeners import FlowEvent, event_bus
                    event_bus.emit(
                        FlowEvent(
                            event_type="git_committed",
                            project_id=self.project_id,
                            entity_type="commit",
                            entity_id=None,
                            data={
                                "commit_hash": commit_hash,
                                "branch": branch_name,
                                "message": message,
                                "agent_id": self.agent_id,
                            },
                        )
                    )
                except Exception:
                    pass  # Don't fail the commit if event emission fails

        except GitToolError as e:
            return self._error(str(e))

        self._log_tool_usage(f"Committed: {message[:80]}")
        return self._success({
            "commit_hash": commit_hash,
            "branch": branch_name,
            "message": message,
        })

    def _run_in_pod(self, message: str, files: list[str] | None = None) -> str:
        """Execute git commit operations inside the agent's pod."""
        try:
            # Check status
            status = self._exec_in_pod(["git", "status", "--porcelain"], timeout=15)
            if not status.stdout.strip():
                return self._error("No changes to commit")

            # Stage files
            if files:
                for f in files:
                    r = self._exec_in_pod(["git", "add", f], timeout=15)
                    if r.exit_code != 0:
                        return self._error(f"Failed to stage {f}: {r.stderr.strip()}")
            else:
                r = self._exec_in_pod(["git", "add", "-A"], timeout=15)
                if r.exit_code != 0:
                    return self._error(f"Failed to stage changes: {r.stderr.strip()}")

            # Commit
            r = self._exec_in_pod(["git", "commit", "-m", message], timeout=30)
            if r.exit_code != 0:
                return self._error(f"Commit failed: {r.stderr.strip()}")

            # Get commit hash
            hr = self._exec_in_pod(["git", "rev-parse", "HEAD"], timeout=10)
            commit_hash = hr.stdout.strip() if hr.exit_code == 0 else None

            # Get branch
            br = self._exec_in_pod(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10,
            )
            branch_name = br.stdout.strip() if br.exit_code == 0 else None

        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        # Update DB and emit events (same as host mode)
        if branch_name and commit_hash:
            def _update_branch(conn: sqlite3.Connection):
                conn.execute(
                    """UPDATE branches
                       SET last_commit_hash = ?, last_commit_at = CURRENT_TIMESTAMP
                       WHERE branch_name = ? AND status = 'active'""",
                    (commit_hash, branch_name),
                )
                conn.commit()

            try:
                execute_with_retry(_update_branch)
            except Exception:
                pass

            try:
                from infinidev.flows.event_listeners import FlowEvent, event_bus
                event_bus.emit(FlowEvent(
                    event_type="git_committed",
                    project_id=self.project_id,
                    entity_type="commit",
                    entity_id=None,
                    data={
                        "commit_hash": commit_hash,
                        "branch": branch_name,
                        "message": message,
                        "agent_id": self.agent_id,
                    },
                ))
            except Exception:
                pass

        self._log_tool_usage(f"Committed (pod): {message[:80]}")
        return self._success({
            "commit_hash": commit_hash,
            "branch": branch_name,
            "message": message,
        })
