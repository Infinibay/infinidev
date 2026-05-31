"""Input schema for RunInBackgroundTool."""

from pydantic import BaseModel, Field


class RunInBackgroundInput(BaseModel):
    command: str = Field(..., description="Shell command to run in the background")
    description: str = Field(
        ...,
        min_length=8,
        description=(
            "REQUIRED. A short (a few words) human-readable label for what "
            "this background command is, e.g. 'vite dev server', 'pytest "
            "--watch', 'docker compose up'. It is shown back to you in the "
            "<background-tasks> section every turn so you remember what is "
            "still running."
        ),
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the command"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables"
    )
