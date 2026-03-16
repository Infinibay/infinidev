"""Tool for merging pull requests via Forgejo API or local git merge."""

import json
import logging
import sqlite3
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.base.db import execute_with_retry

logger = logging.getLogger(__name__)


class MergePRInput(BaseModel):
    pr_number: int = Field(
        ..., description="The PR number (Forgejo PR index) to merge"
    )
    merge_message: str = Field(
        default="", description="Optional merge commit message"
    )


class MergePRTool(InfinibayBaseTool):
    name: str = "merge_pr"
    description: str = (
        "Merge an approved pull request. Looks up the PR by its Forgejo PR "
        "number, merges it via the Forgejo API (or local git merge if no API "
        "is configured), and marks the branch as merged."
    )
    args_schema: Type[BaseModel] = MergePRInput

    def _run(self, pr_number: int, merge_message: str = "") -> str:
        project_id = self.project_id

        # Look up the PR in code_reviews by forgejo_pr_index
        def _find_pr(conn: sqlite3.Connection) -> dict | None:
            row = conn.execute(
                """SELECT id, branch, repo_name, status, task_id
                   FROM code_reviews
                   WHERE project_id = ? AND forgejo_pr_index = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (project_id, pr_number),
            ).fetchone()
            return dict(row) if row else None

        pr = execute_with_retry(_find_pr)
        if not pr:
            return self._error(
                f"PR #{pr_number} not found in project {project_id}"
            )

        if pr["status"] == "merged":
            return self._error(f"PR #{pr_number} is already merged")

        # Perform the merge via PRService
        try:
            from infinidev.git.pr_service import PRService

            pr_svc = PRService()
            repo_path = self._get_repo_path(pr.get("session_id"))
            pr_svc.merge_pr(
                pr["id"], repo_path or ".", merge_message or None
            )
        except Exception as e:
            logger.warning("MergePRTool: merge failed for PR #%d: %s", pr_number, e)
            return self._error(f"Merge failed: {e}")

        self._log_tool_usage(f"Merged PR #{pr_number} (branch: {pr['branch']})")

        # Emit event
        try:
            from infinidev.flows.event_listeners import FlowEvent, event_bus

            event_bus.emit(FlowEvent(
                event_type="pr_merged",
                project_id=project_id,
                entity_type="task",
                entity_id=pr.get("session_id"),
                data={
                    "pr_number": pr_number,
                    "branch": pr["branch"],
                    "agent_id": self.agent_id,
                },
            ))
        except Exception:
            pass

        return self._success({
            "pr_number": pr_number,
            "branch": pr["branch"],
            "status": "merged",
        })

    def _get_repo_path(self, task_id: int | None) -> str | None:
        """Resolve the repository local path for the task's project."""
        if not task_id:
            return None
        try:
            def _query(conn: sqlite3.Connection) -> str | None:
                row = conn.execute(
                    """SELECT r.local_path FROM repositories r
                       JOIN tasks t ON t.project_id = r.project_id
                       WHERE t.id = ? AND r.status = 'active'
                       LIMIT 1""",
                    (task_id,),
                ).fetchone()
                return row["local_path"] if row else None

            return execute_with_retry(_query)
        except Exception:
            return None
