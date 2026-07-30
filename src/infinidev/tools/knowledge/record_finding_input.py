"""Input schema for the record_finding tool."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.knowledge.finding_types import FINDING_TYPE_HELP, FindingType


class RecordFindingInput(BaseModel):
    title: str = Field(..., description="Finding title/topic")
    content: str = Field(..., description="Detailed finding content")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence level (0.0 to 1.0)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    finding_type: FindingType = Field(
        default="observation", description=FINDING_TYPE_HELP
    )
    sources: list[str] = Field(
        default_factory=list, description="Source URLs or references"
    )
    artifact_id: int | None = Field(
        default=None, description="Optional ID of a related artifact"
    )

    # ── Anchored memory parameters (all optional) ──────────────────────
    #
    # If any of these are set, the finding becomes an "anchored memory"
    # that will be automatically appended to the tool result the next
    # time the agent touches the matching anchor — no retrieval step,
    # no separate query, the lesson just appears next to the data that
    # provoked it.
    #
    # Typical use:
    #   record_finding(
    #       title="...", content="...",
    #       finding_type="lesson",
    #       anchor_file="src/infinidev/engine/loop/engine.py",
    #   )
    #
    # Multiple anchors can be set on a single finding; the memory
    # fires if ANY of them matches during a tool call (OR semantics).
    #
    # Each description says what the anchor fires on and nothing else. The
    # mechanism these used to spell out (which tools match, that paths are
    # checked both relative and absolute) is the matcher's business, not a
    # decision the caller makes.
    anchor_file: str | None = Field(
        default=None,
        description=(
            "Workspace-relative or absolute path. Fires when a later tool "
            "call touches this file."
        ),
    )
    anchor_symbol: str | None = Field(
        default=None,
        description=(
            "Qualified symbol name ('ClassName.method'). Fires when a later "
            "tool call references this symbol."
        ),
    )
    anchor_tool: str | None = Field(
        default=None,
        description=(
            "Tool name or command prefix ('pytest', 'git_commit'). Fires "
            "when that tool runs, or an execute_command starts with it."
        ),
    )
    anchor_error: str | None = Field(
        default=None,
        description=(
            "Substring of an error message ('database is locked'). Fires "
            "when a tool result contains it."
        ),
    )
