"""Tool for launching long-running shell commands in the background."""

import logging
import os
from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.shell.background_manager import get_background_manager
from infinidev.tools.shell.execute_command_tool import check_command_permission
from infinidev.tools.shell.run_in_background_input import RunInBackgroundInput

logger = logging.getLogger(__name__)


class RunInBackgroundTool(InfinibayBaseTool):
    name: str = "run_in_background"
    description: str = (
        "Start a long-running shell command in the BACKGROUND and return "
        "immediately with a task id. Use for dev servers, file watchers, "
        "test/build watchers — anything you want running WHILE you keep "
        "working, not a command whose output you need right now (use "
        "execute_command for those). The task and its short description are "
        "tracked in the <background-tasks> section every turn. Check its "
        "output with background_status and stop it with stop_background_task."
    )
    args_schema: Type[BaseModel] = RunInBackgroundInput

    def _run(
        self,
        command: str,
        description: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        if not isinstance(command, str) or not command.strip():
            return self._error("Empty command")
        if not isinstance(description, str) or not description.strip():
            return self._error("A short description is required")

        perm_error = check_command_permission(
            command, description="Run shell command in the background"
        )
        if perm_error:
            return self._error(perm_error)

        run_env = os.environ.copy()
        if isinstance(env, dict):
            # LLMs sometimes send non-string env values; subprocess requires strings.
            run_env.update({str(k): str(v) for k, v in env.items()})

        if not cwd or not isinstance(cwd, str):
            cwd = self.workspace_path or os.getcwd()

        try:
            task = get_background_manager().start(
                command=command,
                description=description.strip(),
                cwd=cwd,
                env=run_env,
            )
        except Exception as e:
            return self._error(f"Failed to start background command: {e}")

        return self._success(
            {
                "id": task.id,
                "description": task.description,
                "status": task.status,
                "message": (
                    f"Started in background as '{task.id}'. It is now tracked "
                    f"in <background-tasks>. Use background_status(task_id="
                    f"'{task.id}') to read its output, or stop_background_task("
                    f"task_id='{task.id}') to stop it."
                ),
            }
        )
