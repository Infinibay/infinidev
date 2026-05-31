"""Input schema for BackgroundStatusTool."""

from pydantic import BaseModel, Field


class BackgroundStatusInput(BaseModel):
    task_id: str | None = Field(
        default=None,
        description=(
            "The background task id to inspect (e.g. 'bg-1'). Omit to get a "
            "compact list of ALL background tasks and their statuses."
        ),
    )
    tail_lines: int = Field(
        default=100,
        description=(
            "How many trailing lines of stdout/stderr to return for a single "
            "task. 0 or negative returns the full retained buffer."
        ),
    )
