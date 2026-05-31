"""Input schema for StopBackgroundTaskTool."""

from pydantic import BaseModel, Field


class StopBackgroundTaskInput(BaseModel):
    task_id: str = Field(
        ..., description="The background task id to stop (e.g. 'bg-1')."
    )
    force: bool = Field(
        default=False,
        description=(
            "If False (default), send a graceful SIGTERM and escalate to "
            "SIGKILL only if the process ignores it. If True, force-kill "
            "immediately with SIGKILL (no chance for cleanup)."
        ),
    )
