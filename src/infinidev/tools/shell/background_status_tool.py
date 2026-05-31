"""Tool for inspecting background shell commands (status + output)."""

import logging
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.shell.background_manager import (
    BackgroundTask,
    get_background_manager,
)
from infinidev.tools.shell.background_status_input import BackgroundStatusInput

logger = logging.getLogger(__name__)


def _tail(text: str, lines: int) -> str:
    """Return the last ``lines`` lines of ``text`` (all of it if lines <= 0)."""
    if lines <= 0 or not text:
        return text
    parts = text.splitlines()
    if len(parts) <= lines:
        return text
    return "\n".join(parts[-lines:])


class BackgroundStatusTool(InfinibayBaseTool):
    name: str = "background_status"
    description: str = (
        "Inspect background commands started with run_in_background. Pass a "
        "task_id to get that task's status, exit code, runtime, and captured "
        "stdout/stderr. Omit task_id to list ALL background tasks. Read-only."
    )
    args_schema: Type[BaseModel] = BackgroundStatusInput
    is_read_only: bool = True

    def _run(self, task_id: str | None = None, tail_lines: int = 100) -> str:
        manager = get_background_manager()

        if not task_id:
            tasks = manager.list()
            if not tasks:
                return self._success(
                    {"tasks": [], "message": "No background tasks have been started."}
                )
            return self._success(
                {
                    "tasks": [
                        {
                            "id": t.id,
                            "description": t.description,
                            "status": t.status,
                            "exit_code": t.exit_code,
                            "runtime_seconds": round(t.runtime_seconds(), 1),
                        }
                        for t in tasks
                    ]
                }
            )

        task: BackgroundTask | None = manager.get(task_id)
        if task is None:
            known = [t.id for t in manager.list()]
            return self._error(
                f"No background task with id '{task_id}'. Known ids: {known or 'none'}"
            )

        summary = task.summary()
        summary["stdout"] = _tail(summary["stdout"], tail_lines)
        summary["stderr"] = _tail(summary["stderr"], tail_lines)
        if task.killed_reason:
            summary["killed_reason"] = task.killed_reason
        return self._success(summary)
