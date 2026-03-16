"""Tool for creating pull requests via Forgejo/Gitea API."""

import asyncio
import json
import os
import sqlite3
import subprocess
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry


class CreatePRInput(BaseModel):
    title: str = Field(..., description="Pull request title")
    body: str = Field(..., description="Pull request description/body")
    base: str = Field(default="main", description="Base branch to merge into")
    draft: bool = Field(default=False, description="Create as draft PR")


class CreatePRTool(InfinibayBaseTool):
    name: str = "create_pr"
    description: str = (
        "Create a pull request on the remote repository. "
        "Requires the current branch to be pushed to the remote. "
        "Auto-populates pr_number and pr_url on the associated task."
    )
    args_schema: Type[BaseModel] = CreatePRInput

    def _run(
        self, title: str, body: str, base: str = "main", draft: bool = False
    ) -> str:
        # Get current branch
        if self._is_pod_mode():
            try:
                r = self._exec_in_pod(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"], timeout=10,
                )
                if r.exit_code != 0:
                    return self._error("Failed to determine current branch")
                head_branch = r.stdout.strip()
            except RuntimeError as e:
                return self._error(f"Pod execution failed: {e}")
        else:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                    capture_output=True, text=True, timeout=10,
                    cwd=self._git_cwd,
                )
                if result.returncode != 0:
                    return self._error("Failed to determine current branch")
                head_branch = result.stdout.strip()
            except Exception as e:
                return self._error(f"Git error: {e}")

        if head_branch == base:
            return self._error(f"Cannot create PR: head branch '{head_branch}' is the same as base '{base}'")

        # Get remote URL to detect Forgejo/GitHub
        remote_result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=10,
            cwd=self._git_cwd,
        )
        remote_url = remote_result.stdout.strip() if remote_result.returncode == 0 else ""

        # Extract repo info from remote URL
        # Supports: https://host/owner/repo.git or git@host:owner/repo.git
        repo_name = "."
        api_base = os.environ.get("FORGEJO_API_URL", "")

        if api_base:
            # Use Forgejo API
            return self._create_forgejo_pr(
                api_base, title, body, head_branch, base, draft, repo_name
            )
        else:
            # Fallback: use git command to create PR info
            return self._create_local_pr_record(
                title, body, head_branch, base, draft, repo_name
            )

    def _create_forgejo_pr(
        self, api_base: str, title: str, body: str,
        head: str, base: str, draft: bool, repo_name: str,
    ) -> str:
        """Create PR via Forgejo/Gitea REST API using curl."""
        token = os.environ.get("FORGEJO_TOKEN", "")
        repo_path = os.environ.get("FORGEJO_REPO", "")  # e.g. "owner/repo"

        if not token or not repo_path:
            return self._error(
                "FORGEJO_TOKEN and FORGEJO_REPO environment variables required"
            )

        api_url = f"{api_base}/repos/{repo_path}/pulls"
        payload = {
            "title": title,
            "body": body,
            "head": head,
            "base": base,
        }

        try:
            result = subprocess.run(
                [
                    "curl", "-s", "-X", "POST", api_url,
                    "-H", f"Authorization: token {token}",
                    "-H", "Content-Type: application/json",
                    "-d", json.dumps(payload),
                ],
                capture_output=True, text=True,
                timeout=settings.WEB_TIMEOUT,
            )
            if result.returncode != 0:
                return self._error(f"API request failed: {result.stderr}")

            response = json.loads(result.stdout)
            pr_url = response.get("html_url", response.get("url", ""))
            pr_number = response.get("number", 0)
        except json.JSONDecodeError:
            return self._error(f"Invalid API response: {result.stdout[:200]}")
        except Exception as e:
            return self._error(f"PR creation failed: {e}")

        # Record in DB
        self._record_pr(head, repo_name, pr_url, forgejo_pr_index=pr_number)

        # Auto-populate pr_number/pr_url on the task
        if self.session_id and pr_number:
            def _update_task_pr(conn: sqlite3.Connection):
                conn.execute(
                    "UPDATE tasks SET pr_number = ?, pr_url = ? WHERE id = ?",
                    (pr_number, pr_url, self.session_id),
                )
                conn.commit()
            try:
                execute_with_retry(_update_task_pr)
            except Exception:
                pass  # Non-critical

        self._log_tool_usage(f"Created PR #{pr_number}: {title}")

        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus

            event_bus.emit(FlowEvent(
                event_type="pr_created",
                project_id=self.project_id,
                entity_type="task",
                entity_id=self.session_id,
                data={
                    "pr_number": pr_number,
                    "url": pr_url,
                    "title": title,
                    "head": head,
                    "base": base,
                    "agent_id": self.agent_id,
                    "session_id": self.session_id,
                },
            ))
        except Exception:
            pass  # Non-critical

        return self._success({
            "pr_number": pr_number,
            "url": pr_url,
            "title": title,
            "head": head,
            "base": base,
        })

    def _create_local_pr_record(
        self, title: str, body: str, head: str, base: str,
        draft: bool, repo_name: str,
    ) -> str:
        """When no API is configured, just record the PR intent in DB."""
        self._record_pr(head, repo_name, "")

        # No Forgejo PR was created, so clear any stale pr_number/pr_url
        # (leave them as NULL)

        self._log_tool_usage(f"PR record created: {title}")

        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus

            event_bus.emit(FlowEvent(
                event_type="pr_created",
                project_id=self.project_id,
                entity_type="task",
                entity_id=self.session_id,
                data={
                    "pr_number": None,
                    "url": None,
                    "title": title,
                    "head": head,
                    "base": base,
                    "agent_id": self.agent_id,
                    "session_id": self.session_id,
                },
            ))
        except Exception:
            pass  # Non-critical

        return self._success({
            "pr_number": None,
            "url": None,
            "title": title,
            "head": head,
            "base": base,
            "note": "No API configured. PR recorded locally in code_reviews table.",
        })

    def _record_pr(
        self, branch: str, repo_name: str, url: str, forgejo_pr_index: int | None = None
    ):
        """Record PR in code_reviews table."""
        project_id = self.project_id
        session_id = self.session_id
        agent_run_id = self.agent_run_id or "unknown"

        def _insert(conn: sqlite3.Connection):
            conn.execute(
                """INSERT INTO code_reviews
                   (project_id, task_id, branch, agent_run_id, repo_name, summary,
                    status, forgejo_pr_index)
                   VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)""",
                (project_id, task_id, branch, agent_run_id, repo_name, url,
                 forgejo_pr_index),
            )
            conn.commit()

        try:
            execute_with_retry(_insert)
        except Exception:
            pass

    async def _arun(
        self, title: str, body: str, base: str = "main", draft: bool = False
    ) -> str:
        """Async version - delegates to sync for now."""
        return self._run(title=title, body=body, base=base, draft=draft)
