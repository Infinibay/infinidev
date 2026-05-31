"""Tool for stopping a background shell command."""

import logging
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.shell.background_manager import get_background_manager
from infinidev.tools.shell.stop_background_task_input import StopBackgroundTaskInput

logger = logging.getLogger(__name__)


class StopBackgroundTaskTool(InfinibayBaseTool):
    name: str = "stop_background_task"
    description: str = (
        "Stop a background command started with run_in_background. By default "
        "sends a graceful SIGTERM (escalating to SIGKILL if ignored); pass "
        "force=True to SIGKILL immediately. Returns the final status and the "
        "last captured output."
    )
    args_schema: Type[BaseModel] = StopBackgroundTaskInput

    def _run(self, task_id: str, force: bool = False) -> str:
        manager = get_background_manager()
        task = manager.get(task_id)
        if task is None:
            known = [t.id for t in manager.list()]
            return self._error(
                f"No background task with id '{task_id}'. Known ids: {known or 'none'}"
            )

        if not task.is_running:
            return self._success(
                {
                    "id": task.id,
                    "description": task.description,
                    "status": task.status,
                    "exit_code": task.exit_code,
                    "message": f"Task '{task.id}' was already not running.",
                }
            )

        task.stop(force=force)
        out, err = task.output()
        return self._success(
            {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "exit_code": task.exit_code,
                "runtime_seconds": round(task.runtime_seconds(), 1),
                "stdout": out[-4000:],
                "stderr": err[-4000:],
                "message": f"Stopped task '{task.id}' (force={force}).",
            }
        )
