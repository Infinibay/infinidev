"""Input schema for WaitForBackgroundTaskTool."""

from pydantic import BaseModel, Field


class WaitForBackgroundTaskInput(BaseModel):
    task_id: str = Field(
        ..., description="The background task id to wait on (e.g. 'bg-1')."
    )
    until_text: str | None = Field(
        default=None,
        description=(
            "Optional readiness marker. If given, the wait ends as soon as "
            "this substring appears in the task's stdout/stderr — use it for "
            "commands that never exit on their own (dev servers, watchers), "
            "e.g. until_text='Listening on'. If omitted, the wait ends when "
            "the process exits."
        ),
    )
    timeout: int | None = Field(
        default=None,
        description=(
            "Maximum seconds to block. Omit to use the configured default. "
            "Capped at a hard ceiling so a wait can never freeze the CLI. "
            "If the condition isn't met in time, the tool returns with "
            "timed_out=True and the task keeps running — you can wait again "
            "or move on."
        ),
    )
    tail_lines: int = Field(
        default=50,
        description=(
            "How many trailing lines of stdout/stderr to return once the "
            "wait ends. 0 or negative returns the full retained buffer."
        ),
    )
