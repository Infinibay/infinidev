"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class OpenedFile(BaseModel):
    """A file cached in the prompt so the LLM doesn't need to re-read it.

    Files that were **written or edited** by the agent are marked as
    ``pinned=True``.  Pinned files never expire and are not evicted by
    the LRU policy — they stay in the prompt for the entire task so the
    model can always refer back to what it wrote.
    """

    path: str
    content: str
    ttl: int = 8  # Remaining tool calls before expiry
    pinned: bool = False  # True for files the agent wrote/edited

    def tick(self, n: int = 1) -> None:
        """Decrement TTL by *n* tool calls (no-op for pinned files)."""
        if not self.pinned:
            self.ttl = max(0, self.ttl - n)

    @property
    def expired(self) -> bool:
        return not self.pinned and self.ttl <= 0


