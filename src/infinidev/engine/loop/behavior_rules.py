"""Modular behavior rules for the loop engine.

Each rule is a subclass of BehaviorRule. The BehaviorTracker auto-discovers
all subclasses and runs them on every tool call. To add a new rule, create
a new class here — no other files need to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal

# ── Tool name sets ──────────────────────────────────────────────────────

_EDIT_TOOLS = frozenset({
    "replace_lines", "create_file",
    "add_content_after_line", "add_content_before_line",
    "edit_symbol", "add_symbol", "remove_symbol",
})

_READ_TOOLS = frozenset({
    "read_file", "partial_read",
})

_TEST_PATTERNS = frozenset({
    "pytest", "unittest", "test", "npm test", "cargo test",
    "go test", "make test", "jest", "mocha", "vitest",
})


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class Feedback:
    """A single piece of behavioral feedback."""
    kind: Literal["good", "bad"]
    message: str
    score_delta: int  # +1 for good, -1 for bad


@dataclass
class RuleContext:
    """Snapshot passed to each rule on every tool call."""
    tool_name: str
    tool_args: dict
    had_error: bool
    files_read: set[str]
    files_edited: set[str]
    opened_files: set[str]
    tool_history: list[str]
    task_has_edits: bool
    notes_count: int
    step_end: bool = False  # True when called from on_step_end


# ── Base class ──────────────────────────────────────────────────────────

class BehaviorRule(ABC):
    """Base class for all behavior rules."""

    name: str

    @abstractmethod
    def on_tool_call(self, ctx: RuleContext) -> list[Feedback]:
        """Called after every tool call. Return feedback items."""
        ...

    def on_step_end(self, ctx: RuleContext) -> list[Feedback]:
        """Called when step completes. Override for end-of-step checks."""
        return []


# ── Rules ───────────────────────────────────────────────────────────────

class ReadBeforeEditRule(BehaviorRule):
    """Reward reading a file before editing it; punish blind edits."""

    name = "read_before_edit"

    def __init__(self) -> None:
        self._praised_paths: set[str] = set()

    def on_tool_call(self, ctx: RuleContext) -> list[Feedback]:
        if ctx.tool_name not in _EDIT_TOOLS:
            return []

        path = ctx.tool_args.get("file_path", ctx.tool_args.get("path", ""))
        if not path:
            return []

        if path in ctx.files_read or path in ctx.opened_files:
            if path not in self._praised_paths:
                self._praised_paths.add(path)
                return [Feedback("good", f"Good: you read {_short(path)} before editing.", +1)]
        else:
            return [Feedback("bad",
                f"WARNING: You are editing {_short(path)} without reading it first. "
                "Always read a file before modifying it.", -1)]
        return []


class TestAfterEditRule(BehaviorRule):
    """Reward running tests after edits; warn at step end if none ran."""

    name = "test_after_edit"

    def __init__(self) -> None:
        self._test_praised = False

    def on_tool_call(self, ctx: RuleContext) -> list[Feedback]:
        if not ctx.task_has_edits:
            return []
        if ctx.tool_name != "execute_command":
            return []
        cmd = ctx.tool_args.get("command", "")
        if any(pat in cmd.lower() for pat in _TEST_PATTERNS):
            if not self._test_praised:
                self._test_praised = True
                return [Feedback("good", "Good: running tests after making changes.", +1)]
        return []

    def on_step_end(self, ctx: RuleContext) -> list[Feedback]:
        if ctx.task_has_edits and not self._test_praised:
            has_edits_this_step = bool(ctx.files_edited)
            if has_edits_this_step:
                return [Feedback("bad",
                    "NOTE: You edited files but did not run tests this step. "
                    "Consider running tests to verify your changes.", -1)]
        return []


class NoteTakingRule(BehaviorRule):
    """Reward saving notes after reads; warn if too many calls without noting."""

    name = "note_taking"

    def __init__(self) -> None:
        self._calls_since_note = 0
        self._note_praised = False

    def on_tool_call(self, ctx: RuleContext) -> list[Feedback]:
        if ctx.tool_name == "add_note":
            if len(ctx.files_read) >= 2 and not self._note_praised:
                self._note_praised = True
                self._calls_since_note = 0
                return [Feedback("good", "Good: saving notes after reading files.", +1)]
            self._calls_since_note = 0
            return []

        self._calls_since_note += 1
        if self._calls_since_note >= 5 and len(ctx.files_read) >= 2 and ctx.notes_count == 0:
            self._calls_since_note = 0  # Don't spam
            return [Feedback("bad",
                "WARNING: You have read multiple files but saved no notes. "
                "Use add_note to record key findings before they scroll out of context.", -1)]
        return []


# ── Helpers ─────────────────────────────────────────────────────────────

def _short(path: str) -> str:
    """Shorten a file path for display."""
    parts = path.rsplit("/", 2)
    return "/".join(parts[-2:]) if len(parts) > 2 else path


def discover_rules() -> list[BehaviorRule]:
    """Instantiate all BehaviorRule subclasses."""
    return [cls() for cls in BehaviorRule.__subclasses__()]
