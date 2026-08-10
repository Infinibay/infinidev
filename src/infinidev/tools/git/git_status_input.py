"""Input schema for viewing Git status."""

from pydantic import BaseModel, Field


class GitStatusInput(BaseModel):
    path: str | None = Field(
        default=None,
        description=(
            "Repository directory relative to the workspace. Omit when the runtime "
            "already selected a target or the workspace contains exactly one repo."
        ),
    )


