"""Behavior tracker — scores model actions and injects inline feedback."""

from __future__ import annotations

import json
from typing import Any

from infinidev.engine.loop.behavior_rules import (
    BehaviorRule, Feedback, RuleContext, discover_rules,
    _EDIT_TOOLS, _READ_TOOLS,
)


class BehaviorTracker:
    """Tracks good/bad model behaviors per step, queues inline feedback."""

    def __init__(self, opened_files: set[str]) -> None:
        self.files_read: set[str] = set()
        self.files_edited: set[str] = set()
        self.tool_history: list[str] = []
        self.task_has_edits: bool = False
        self.notes_count: int = 0
        self.score: int = 0
        self._opened_files = opened_files
        self._feedback_log: list[Feedback] = []
        self._pending: list[str] = []
        self._rules: list[BehaviorRule] = discover_rules()

    def on_tool_call(self, name: str, args_str: str, had_error: bool) -> None:
        """Record a tool call, run all rules, queue feedback."""
        try:
            args = json.loads(args_str) if isinstance(args_str, str) else (args_str or {})
        except (json.JSONDecodeError, TypeError):
            args = {}

        path = args.get("file_path", args.get("path", ""))

        # Update state
        self.tool_history.append(name)
        if name in _READ_TOOLS and path:
            self.files_read.add(path)
        if name in _EDIT_TOOLS and path:
            self.files_edited.add(path)
            self.task_has_edits = True
        if name == "add_note":
            self.notes_count += 1

        ctx = self._build_ctx(name, args, had_error)
        for rule in self._rules:
            for fb in rule.on_tool_call(ctx):
                self._record(fb)

    def on_step_end(self) -> None:
        """Run end-of-step checks across all rules."""
        ctx = self._build_ctx("", {}, False)
        ctx.step_end = True
        for rule in self._rules:
            for fb in rule.on_step_end(ctx):
                self._record(fb)

    def drain_feedback(self) -> str:
        """Return and clear queued feedback as a single string block."""
        if not self._pending:
            return ""
        block = "\n".join(self._pending)
        self._pending.clear()
        return block

    def summary(self) -> dict[str, Any]:
        """Return behavior summary for the ActionRecord."""
        good = [fb.message for fb in self._feedback_log if fb.kind == "good"]
        bad = [fb.message for fb in self._feedback_log if fb.kind == "bad"]
        return {
            "behavior_score": self.score,
            "good_patterns": good,
            "bad_patterns": bad,
        }

    def _build_ctx(self, name: str, args: dict, had_error: bool) -> RuleContext:
        # No defensive copies — rules are internal and must not mutate ctx.
        return RuleContext(
            tool_name=name,
            tool_args=args,
            had_error=had_error,
            files_read=self.files_read,
            files_edited=self.files_edited,
            opened_files=self._opened_files,
            tool_history=self.tool_history,
            task_has_edits=self.task_has_edits,
            notes_count=self.notes_count,
        )

    def _record(self, fb: Feedback) -> None:
        self.score += fb.score_delta
        self._feedback_log.append(fb)
        self._pending.append(f"[BEHAVIOR] {fb.message}")
