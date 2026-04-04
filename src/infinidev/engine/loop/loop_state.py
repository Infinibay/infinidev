"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.loop_plan import LoopPlan
from infinidev.engine.loop.opened_file import OpenedFile

# Default TTL for opened files (in tool calls)
OPENED_FILE_TTL = 20
# Max number of files to keep in the cache (to avoid prompt bloat)
MAX_OPENED_FILES = 10
# Max file content size to cache (larger files are not cached)
MAX_CACHE_CONTENT_SIZE = 32000  # ~8K tokens — enough for most source files


class LoopState(BaseModel):
    """Full state of the loop engine across iterations."""

    plan: LoopPlan = Field(default_factory=LoopPlan)
    history: list[ActionRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)  # Scratchpad notes that persist across iterations
    opened_files: dict[str, OpenedFile] = Field(default_factory=dict)  # File content cache
    current_step_index: int = 0
    iteration_count: int = 0
    total_tool_calls: int = 0
    total_tokens: int = 0
    last_prompt_tokens: int = 0       # prompt_tokens from most recent LLM call
    last_completion_tokens: int = 0   # completion_tokens from most recent LLM call
    tool_calls_since_last_note: int = 0  # For gentle note-taking nudge

    def cache_file(self, path: str, content: str, pinned: bool = False) -> None:
        """Add or update a file in the opened files cache."""
        if len(content) > MAX_CACHE_CONTENT_SIZE:
            # Too large to cache — skip
            return
        self.opened_files[path] = OpenedFile(
            path=path, content=content, ttl=OPENED_FILE_TTL, pinned=pinned,
        )
        # Evict oldest *unpinned* file if over limit
        while len(self.opened_files) > MAX_OPENED_FILES:
            unpinned = {k: v for k, v in self.opened_files.items() if not v.pinned}
            if not unpinned:
                break  # all files are pinned — don't evict
            oldest = min(unpinned, key=lambda k: unpinned[k].ttl)
            del self.opened_files[oldest]

    def refresh_file(self, path: str, content: str) -> None:
        """Update content for a file the agent wrote/edited.

        Marks the file as **pinned** so it stays in the prompt for the
        entire task — the model should always be able to see what it wrote.
        """
        if len(content) > MAX_CACHE_CONTENT_SIZE:
            self.opened_files.pop(path, None)
            return
        if path in self.opened_files:
            self.opened_files[path].content = content
            self.opened_files[path].ttl = OPENED_FILE_TTL
            self.opened_files[path].pinned = True
        else:
            self.cache_file(path, content, pinned=True)

    def tick_opened_files(self, tool_calls: int = 1) -> None:
        """Age all cached files and remove expired ones."""
        for f in self.opened_files.values():
            f.tick(tool_calls)
        self.opened_files = {
            k: v for k, v in self.opened_files.items() if not v.expired
        }

