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


class Question(BaseModel):
    """A question to be answered during node exploration."""

    content: str
    question_type: Literal["pivot", "informational"] = "informational"
    answer: str | None = None
    answered_by_tool: str | None = None
    impact: str | None = None  # What changed when answered (for pivots)


