"""Input schema for the update_finding tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from infinidev.tools.knowledge.finding_types import FINDING_TYPES, FINDING_TYPE_HELP


class UpdateFindingInput(BaseModel):
    finding_id: int = Field(..., description="ID of the finding to update")
    title: str | None = Field(default=None, description="New title (topic)")
    content: str | None = Field(default=None, description="New content")
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0, description="New confidence level"
    )
    finding_type: str | None = Field(
        default=None,
        description=(
            f"New finding type: {', '.join(FINDING_TYPES)}. {FINDING_TYPE_HELP}"
        ),
    )
    tags: list[str] | None = Field(default=None, description="Replace tags")
    sources: list[str] | None = Field(default=None, description="Replace sources")

    # ── Anchor updates (None = unchanged, empty string = clear) ────────
    #
    # Updating an anchor is useful for two cases: (a) promoting a
    # previously-unanchored observation into a lesson by attaching it
    # to a file, (b) moving an anchor when a symbol has been renamed.
    anchor_file: str | None = Field(
        default=None,
        description=(
            "New anchor_file (workspace-relative or absolute). Set to "
            "empty string '' to clear. None = leave unchanged."
        ),
    )
    anchor_symbol: str | None = Field(
        default=None,
        description=(
            "New anchor_symbol. Empty string '' clears, None leaves unchanged."
        ),
    )
    anchor_tool: str | None = Field(
        default=None,
        description=(
            "New anchor_tool. Empty string '' clears, None leaves unchanged."
        ),
    )
    anchor_error: str | None = Field(
        default=None,
        description=(
            "New anchor_error. Empty string '' clears, None leaves unchanged."
        ),
    )
