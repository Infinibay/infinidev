"""Validated arguments for bounded private command-output reads."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Keep tool-schema construction independent of ``infinidev.engine``. Importing
# the engine while ``infinidev.tools`` is registering classes creates a cycle
# through analysis.planner -> tools. The store enforces this ceiling again at
# runtime, so this duplicate is only the schema's static upper bound.
COMMAND_OUTPUT_MAX_READ_BYTES = 64 * 1024


class ReadCommandOutputInput(BaseModel):
    """Opaque handle fields plus a bounded UTF-8 byte range."""

    artifact_id: int = Field(gt=0, description="Artifact ID from execute_command")
    type: Literal["command_output"] = Field(
        description="Type copied exactly from the handle: command_output",
    )
    stream: Literal["stdout", "stderr"] = Field(
        description="Stream named by the handle"
    )
    char_count: int = Field(
        ge=0, description="Character count copied exactly from the handle"
    )
    byte_count: int = Field(
        ge=0, description="UTF-8 byte count copied exactly from the handle"
    )
    offset: int = Field(
        default=0,
        ge=0,
        description="UTF-8 byte offset, on a character boundary",
    )
    limit: int = Field(
        default=16_384,
        ge=1,
        le=COMMAND_OUTPUT_MAX_READ_BYTES,
        description=(
            "Maximum UTF-8 bytes to return; use returned_end as the next offset"
        ),
    )

    model_config = {"populate_by_name": True}
