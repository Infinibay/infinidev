"""Pydantic models for the exploration tree engine.

Decomposes complex problems into sub-problems, explores them recursively
with tools, propagates results upward, and synthesizes findings.

Three phases: INIT (decompose) -> EXPLORE (loop) -> SYNTHESIZE (final).

Brainstorming Mode: for very short problem statements (< 100 chars),
the engine enters a speculative exploration mode where hypotheses can be
generated, factorized into subproblems, and explored through an
intermediate factorization phase (Phase 2.5).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Blocker(BaseModel):
    """An external blocker preventing node resolution."""

    description: str
    blocker_type: Literal["api", "library", "permission", "infra", "unknown"] = "unknown"
    workaround: str | None = None


