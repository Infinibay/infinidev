"""Tool for unified cross-source knowledge search using FTS5."""

from typing import Literal

from pydantic import BaseModel, Field

from infinidev.tools.knowledge.finding_types import FindingType


class SearchKnowledgeInput(BaseModel):
    query: str = Field(
        default="",
        description=(
            "Full-text query. Operators: '|' OR, '&' AND, 'arch*' prefix, "
            "'\"exact phrase\"'. Leave empty to browse findings by filter "
            "instead of searching."
        ),
    )
    sources: list[Literal["findings", "reports"]] = Field(
        default=["findings", "reports"],
        description="Where to look. Reports need a query; browsing covers findings only.",
    )
    limit: int = Field(default=20, ge=1, le=100, description="Max results per source")
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Minimum confidence filter (findings only)",
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Findings from this session. Defaults to the current one; "
            "pass '0' for every session in the project."
        ),
    )
    finding_type: FindingType | None = Field(
        default=None, description="Keep only findings of this type."
    )


