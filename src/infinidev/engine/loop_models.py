"""Pydantic models for the plan-execute-summarize loop engine."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    """A single step in the agent's execution plan."""

    index: int
    description: str
    status: Literal["pending", "active", "done", "skipped"] = "pending"


class StepOperation(BaseModel):
    """A structured operation to apply to the plan."""

    op: Literal["add", "modify", "remove"]
    index: int
    description: str = ""  # required for add/modify, ignored for remove


class LoopPlan(BaseModel):
    """The agent's mutable execution plan."""

    steps: list[PlanStep] = Field(default_factory=list)

    @property
    def active_step(self) -> PlanStep | None:
        """Return the first step with status='active', or None."""
        for step in self.steps:
            if step.status == "active":
                return step
        return None

    @property
    def has_pending(self) -> bool:
        """True if any step is pending or active."""
        return any(s.status in ("pending", "active") for s in self.steps)

    def mark_active_done(self) -> None:
        """Mark the current active step as done (without activating next)."""
        for step in self.steps:
            if step.status == "active":
                step.status = "done"
                break

    def activate_next(self) -> None:
        """Activate the next pending step."""
        for step in self.steps:
            if step.status == "pending":
                step.status = "active"
                break

    def advance(self) -> None:
        """Mark the active step as done and activate the next pending step."""
        self.mark_active_done()
        self.activate_next()

    def apply_operations(self, ops: list[StepOperation]) -> None:
        """Apply structured add/modify/remove operations to the plan."""
        for op in ops:
            if op.op == "add":
                # Replace existing step at same index if present, but preserve done steps
                self.steps = [s for s in self.steps if s.index != op.index or s.status == "done"]
                # Only add if we actually removed the old step (i.e., it wasn't done)
                if not any(s.index == op.index and s.status == "done" for s in self.steps):
                    self.steps.append(PlanStep(index=op.index, description=op.description))

            elif op.op == "modify":
                for step in self.steps:
                    if step.index == op.index:
                        step.description = op.description
                        break

            elif op.op == "remove":
                for step in self.steps:
                    if step.index == op.index and step.status in ("pending", "active"):
                        step.status = "skipped"
                        break

        self.steps.sort(key=lambda s: s.index)

    def render(self) -> str:
        """Render the plan as a numbered list with status markers."""
        lines: list[str] = []
        for step in self.steps:
            tag = f"[{step.status}] " if step.status != "pending" else ""
            lines.append(f"{step.index}. {tag}{step.description}")
        return "\n".join(lines)


class ActionRecord(BaseModel):
    """Structured summary of a completed step."""

    step_index: int
    summary: str
    tool_calls_count: int = 0
    files_to_preload: list[str] = Field(default_factory=list)
    changes_made: str = ""
    discovered_context: str = ""
    pending_items: str = ""
    anti_patterns: str = ""


class StepResult(BaseModel):
    """Parsed result from the LLM's step_complete tool call."""

    summary: str
    next_steps: list[StepOperation] = Field(default_factory=list)
    status: Literal["continue", "done", "blocked", "explore"] = "continue"
    final_answer: str | None = None


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
