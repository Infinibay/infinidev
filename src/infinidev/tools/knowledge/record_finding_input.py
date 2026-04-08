"""Input schema for the record_finding tool."""

from __future__ import annotations

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.knowledge.finding_types import FINDING_TYPES, FINDING_TYPE_HELP


class RecordFindingInput(BaseModel):
    title: str = Field(..., description="Finding title/topic")
    content: str = Field(..., description="Detailed finding content")
    confidence: float = Field(
        default=0.5, ge=0.0, le=1.0, description="Confidence level (0.0 to 1.0)"
    )
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    finding_type: str = Field(
        default="observation",
        description=(
            f"Finding type: {', '.join(FINDING_TYPES)}. {FINDING_TYPE_HELP}"
        ),
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
    anchor_file: str | None = Field(
        default=None,
        description=(
            "File path this memory is anchored to. Matches on "
            "read_file / edit_symbol / replace_lines / etc. when the "
            "agent touches this file. Use a workspace-relative or "
            "absolute path — the matcher checks both. Required for "
            "lessons/rules/landmines about a specific file."
        ),
    )
    anchor_symbol: str | None = Field(
        default=None,
        description=(
            "Qualified symbol name this memory is anchored to (e.g. "
            "'ClassName.method' or 'function_name'). Matches on "
            "get_symbol_code / edit_symbol / search_symbols when the "
            "agent references this symbol."
        ),
    )
    anchor_tool: str | None = Field(
        default=None,
        description=(
            "Tool name or command prefix this memory is anchored to "
            "(e.g. 'pytest', 'git_commit', 'execute_command'). Matches "
            "when the agent calls that tool, OR when an execute_command "
            "starts with that token. Use for 'when doing X, remember Y' "
            "rules that are not tied to a specific file."
        ),
    )
    anchor_error: str | None = Field(
        default=None,
        description=(
            "Substring of an error message this memory is anchored to "
            "(e.g. 'database is locked', 'SIGSEGV during shutdown'). "
            "Matches when a tool result contains this substring. Use "
            "for 'if you see this error, try X' rules."
        ),
    )
