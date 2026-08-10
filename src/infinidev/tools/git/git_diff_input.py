"""Input schema for viewing Git diffs."""

from pydantic import BaseModel, Field


class GitDiffInput(BaseModel):
    path: str | None = Field(
        default=None,
        description=(
            "Repository directory relative to the workspace. Omit when the runtime "
            "already selected a target or the workspace contains exactly one repo."
        ),
    )
    branch: str | None = Field(
        default=None, description="Branch to diff against (e.g. 'main')"
    )
    file: str | None = Field(
        default=None, description="Specific file to diff"
    )
    staged: bool = Field(
        default=False, description="Show only staged changes"
    )


