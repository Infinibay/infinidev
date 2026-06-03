"""Tool for blocking until a background shell command finishes (or is ready).

The other background tools are all non-blocking: ``run_in_background`` returns
a task id immediately, and ``background_status`` only snapshots the current
state. Without a blocking primitive the agent's only way to "wait" for a build
or migration is to poll ``background_status`` across separate loop iterations —
each one rebuilds the whole prompt and burns a step just to read "still
running". This tool lets the agent block the turn until the task actually
finishes (or a server prints its readiness line), then continue with the
result in hand.
"""

import logging
from typing import Type

from pydantic import BaseModel

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.shell.background_manager import (
    BackgroundTask,
    acknowledge_completion,
    get_background_manager,
)
from infinidev.tools.shell.background_status_tool import _tail
from infinidev.tools.shell.wait_for_background_task_input import (
    WaitForBackgroundTaskInput,
)

logger = logging.getLogger(__name__)


class WaitForBackgroundTaskTool(InfinibayBaseTool):
    name: str = "wait_for_background_task"
    description: str = (
        "BLOCK until a background task started with run_in_background finishes, "
        "then return its final status and output. Use this instead of polling "
        "background_status in a loop when the next thing you do depends on the "
        "task being done (a build, a test run, a migration). For commands that "
        "never exit on their own (dev servers, watchers), pass until_text with "
        "a readiness marker (e.g. 'Listening on') and the wait ends as soon as "
        "that text appears. The wait is bounded by a timeout; if it elapses the "
        "task keeps running and you get timed_out=True so you can wait again or "
        "move on. Read-only — it does not stop the task."
    )
    args_schema: Type[BaseModel] = WaitForBackgroundTaskInput
    is_read_only: bool = True

    def _run(
        self,
        task_id: str,
        until_text: str | None = None,
        timeout: int | None = None,
        tail_lines: int = 50,
    ) -> str:
        manager = get_background_manager()
        task: BackgroundTask | None = manager.get(task_id)
        if task is None:
            known = [t.id for t in manager.list()]
            return self._error(
                f"No background task with id '{task_id}'. Known ids: {known or 'none'}"
            )

        if isinstance(until_text, str) and not until_text.strip():
            until_text = None

        # Resolve and clamp the timeout. The caller's value (or the default)
        # is capped at a hard ceiling so a single wait can never block the CLI
        # indefinitely; a non-positive value would mean "wait forever", which
        # we never allow here.
        max_timeout = max(1, settings.BACKGROUND_WAIT_MAX_TIMEOUT)
        if timeout is None or timeout <= 0:
            timeout = settings.BACKGROUND_WAIT_TIMEOUT
        effective_timeout = min(max(1, timeout), max_timeout)

        # Already finished, and we're not waiting for a readiness marker:
        # return immediately rather than sleeping a full poll interval.
        if not task.is_running and not until_text:
            return self._result(task, "exited", tail_lines)

        reason = task.wait(timeout=effective_timeout, until_text=until_text)
        return self._result(task, reason, tail_lines, requested_timeout=effective_timeout)

    def _result(
        self,
        task: BackgroundTask,
        reason: str,
        tail_lines: int,
        requested_timeout: int | None = None,
    ) -> str:
        summary = task.summary()
        summary["stdout"] = _tail(summary["stdout"], tail_lines)
        summary["stderr"] = _tail(summary["stderr"], tail_lines)
        summary["wait_result"] = reason
        summary["timed_out"] = reason == "timeout"
        if task.killed_reason:
            summary["killed_reason"] = task.killed_reason
        # We've just shown the model this task's finished state, so suppress
        # the redundant <background-task-finished> nudge next iteration.
        if reason == "exited":
            acknowledge_completion(task.id)

        if reason == "timeout":
            summary["message"] = (
                f"Task '{task.id}' is still running after {requested_timeout}s. "
                "It was NOT stopped. Wait again, check it with background_status, "
                "or move on."
            )
        elif reason == "matched":
            summary["message"] = (
                f"Task '{task.id}' produced the expected output and is still "
                "running in the background."
            )
        else:  # exited
            verdict = "ok" if task.exit_code == 0 else f"FAILED (exit {task.exit_code})"
            summary["message"] = f"Task '{task.id}' finished — {verdict}."
        return self._success(summary)
