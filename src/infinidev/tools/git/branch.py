"""Tool for Git branch operations."""

import logging
import re
import sqlite3
import subprocess
from typing import Type

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class GitBranchInput(BaseModel):
    branch_name: str = Field(..., description="Name of the branch")
    create: bool = Field(default=True, description="Create the branch if True, checkout if False")
    base_branch: str = Field(default="main", description="Base branch to create from")


class GitBranchTool(InfinibayBaseTool):
    name: str = "git_branch"
    description: str = (
        "Create or checkout a Git branch. "
        "When creating, it branches from the specified base branch."
    )
    args_schema: Type[BaseModel] = GitBranchInput

    def _run(
        self, branch_name: str, create: bool = True, base_branch: str = "main"
    ) -> str:
        # Validate branch name
        if not re.match(r"^[a-zA-Z0-9._/-]+$", branch_name):
            return self._error(
                f"Invalid branch name: '{branch_name}'. "
                "Use only alphanumerics, dots, dashes, underscores, and slashes."
            )

        if self._is_pod_mode():
            return self._run_in_pod(branch_name, create, base_branch)

        cwd = self._git_cwd
        try:
            if create:
                # Verify origin remote exists before fetching
                check = subprocess.run(
                    ["git", "remote", "get-url", "origin"],
                    capture_output=True, text=True, timeout=10, cwd=cwd,
                )
                if check.returncode != 0:
                    return self._error(
                        "No remote 'origin' configured. Cannot fetch from remote. "
                        "Will use local branch as base instead."
                    )
                # Fetch latest; fall back to local ref if fetch fails
                fetch = subprocess.run(
                    ["git", "fetch", "origin", base_branch],
                    capture_output=True, text=True, timeout=30, cwd=cwd,
                )
                if fetch.returncode == 0:
                    checkout_ref = f"origin/{base_branch}"
                else:
                    logger.warning(
                        "git fetch origin %s failed (exit %d): %s — "
                        "falling back to local '%s'",
                        base_branch, fetch.returncode,
                        fetch.stderr.strip(), base_branch,
                    )
                    checkout_ref = base_branch
                result = subprocess.run(
                    ["git", "checkout", "-b", branch_name, checkout_ref],
                    capture_output=True, text=True, timeout=15, cwd=cwd,
                )
                if result.returncode != 0:
                    # Maybe branch already exists, try checkout
                    result = subprocess.run(
                        ["git", "checkout", branch_name],
                        capture_output=True, text=True, timeout=15, cwd=cwd,
                    )
                    if result.returncode != 0:
                        return self._error(f"Git error: {result.stderr.strip()}")
                    action = "checked_out"
                else:
                    action = "created"
            else:
                result = subprocess.run(
                    ["git", "checkout", branch_name],
                    capture_output=True, text=True, timeout=15, cwd=cwd,
                )
                if result.returncode != 0:
                    return self._error(f"Git error: {result.stderr.strip()}")
                action = "checked_out"
        except subprocess.TimeoutExpired:
            return self._error("Git operation timed out")
        except FileNotFoundError:
            return self._error("Git is not installed or not in PATH")

        # Register branch in DB
        project_id = self.project_id
        agent_id = self.agent_id or "unknown"

        if action == "created":
            def _register(conn: sqlite3.Connection):
                conn.execute(
                    """INSERT OR IGNORE INTO branches
                       (project_id, task_id, repo_name, branch_name, base_branch, status, created_by)
                       VALUES (?, ?, ?, ?, ?, 'active', ?)""",
                    (project_id, self.session_id, ".", branch_name, base_branch, agent_id),
                )
                conn.commit()

            try:
                execute_with_retry(_register)
            except Exception:
                pass  # Non-critical

        self._log_tool_usage(f"Branch {action}: {branch_name}")

        # Auto-populate branch_name on the task
        if self.session_id and action == "created":
            def _update_task_branch(conn: sqlite3.Connection):
                conn.execute(
                    "UPDATE tasks SET branch_name = ? WHERE id = ?",
                    (branch_name, self.session_id),
                )
                conn.commit()
            try:
                execute_with_retry(_update_task_branch)
            except Exception:
                pass  # Non-critical

        if action == "created":
            try:
                from infinidev.flows.event_listeners import FlowEvent, event_bus

                event_bus.emit(FlowEvent(
                    event_type="branch_created",
                    project_id=project_id,
                    entity_type="task",
                    entity_id=self.session_id,
                    data={
                        "branch_name": branch_name,
                        "base_branch": base_branch,
                        "agent_id": agent_id,
                        "session_id": self.session_id,
                    },
                ))
            except Exception:
                pass  # Non-critical

        return self._success({
            "branch": branch_name,
            "action": action,
            "base": base_branch if action == "created" else None,
        })

    def _run_in_pod(
        self, branch_name: str, create: bool, base_branch: str,
    ) -> str:
        """Execute git branch operations inside the agent's pod."""
        try:
            if create:
                # Try fetching latest from remote; fall back to local ref
                fetch = self._exec_in_pod(
                    ["git", "fetch", "origin", base_branch], timeout=30,
                )
                if fetch.exit_code == 0:
                    checkout_ref = f"origin/{base_branch}"
                else:
                    logger.warning(
                        "git fetch origin %s failed (exit %d): %s — "
                        "falling back to local '%s'",
                        base_branch, fetch.exit_code,
                        fetch.stderr.strip(), base_branch,
                    )
                    checkout_ref = base_branch

                r = self._exec_in_pod(
                    ["git", "checkout", "-b", branch_name, checkout_ref],
                    timeout=15,
                )
                if r.exit_code != 0:
                    r = self._exec_in_pod(
                        ["git", "checkout", branch_name], timeout=15,
                    )
                    if r.exit_code != 0:
                        return self._error(f"Git error: {r.stderr.strip()}")
                    action = "checked_out"
                else:
                    action = "created"
            else:
                r = self._exec_in_pod(
                    ["git", "checkout", branch_name], timeout=15,
                )
                if r.exit_code != 0:
                    return self._error(f"Git error: {r.stderr.strip()}")
                action = "checked_out"
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        # Register branch in DB
        project_id = self.project_id
        agent_id = self.agent_id or "unknown"

        if action == "created":
            def _register(conn):
                conn.execute(
                    """INSERT OR IGNORE INTO branches
                       (project_id, task_id, repo_name, branch_name, base_branch, status, created_by)
                       VALUES (?, ?, ?, ?, ?, 'active', ?)""",
                    (project_id, self.session_id, ".", branch_name, base_branch, agent_id),
                )
                conn.commit()

            try:
                from infinidev.tools.base.db import execute_with_retry
                execute_with_retry(_register)
            except Exception:
                pass

        self._log_tool_usage(f"Branch {action}: {branch_name} (pod)")

        # Auto-populate branch_name on the task
        if self.session_id and action == "created":
            def _update_task_branch(conn):
                conn.execute(
                    "UPDATE tasks SET branch_name = ? WHERE id = ?",
                    (branch_name, self.session_id),
                )
                conn.commit()
            try:
                from infinidev.tools.base.db import execute_with_retry as _retry
                _retry(_update_task_branch)
            except Exception:
                pass  # Non-critical

        if action == "created":
            try:
                from infinidev.flows.event_listeners import FlowEvent, event_bus
                event_bus.emit(FlowEvent(
                    event_type="branch_created",
                    project_id=project_id,
                    entity_type="task",
                    entity_id=self.session_id,
                    data={
                        "branch_name": branch_name,
                        "base_branch": base_branch,
                        "agent_id": agent_id,
                        "session_id": self.session_id,
                    },
                ))
            except Exception:
                pass

        return self._success({
            "branch": branch_name,
            "action": action,
            "base": base_branch if action == "created" else None,
        })
