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


class Fact(BaseModel):
    """A verified or asserted piece of knowledge within a tree node."""

    content: str
    source: Literal["initial", "discovered", "inherited"] = "discovered"
    evidence: str = ""  # Raw tool output backing this fact
    source_tool: str = ""  # Which tool produced the evidence
    confidence: Literal["high", "medium", "low"] = "medium"


