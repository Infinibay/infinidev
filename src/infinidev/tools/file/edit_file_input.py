"""Input schema for the edit_file tool."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EditFileInput(BaseModel):
    file_path: str = Field(..., description="Path to the file to edit.")
    old_string: str = Field(
        ...,
        description=(
            "Exact text to replace, copied from the file — whitespace and "
            "indentation included. Must appear exactly once unless "
            "replace_all is set; if it does not, include more surrounding "
            "lines until it is unique."
        ),
    )
    new_string: str = Field(
        ..., description="Text to put in its place. Empty string deletes it."
    )
    replace_all: bool = Field(
        default=False,
        description="Replace every occurrence instead of requiring exactly one.",
    )
    rationale: str = Field(
        default="",
        description="One line on why this edit is needed.",
    )
