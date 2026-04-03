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


class BrainstormingDetector:
    """Detects when to enter brainstorming mode based on problem characteristics.

    Triggers brainstorming mode when:
    - Problem statement is very short (< 100 characters)
    - Problem is empty or whitespace-only
    """

    BRAINSTORMING_THRESHOLD = 100  # Maximum problem length to trigger brainstorming

    def should_brainstorm(self, problem: str) -> bool:
        """Determine if the system should enter brainstorming mode.

        Args:
            problem: The problem statement from the user.

        Returns:
            True if brainstorming mode should be activated.
        """
        if not problem or not problem.strip():
            return True

        return len(problem) < self.BRAINSTORMING_THRESHOLD


