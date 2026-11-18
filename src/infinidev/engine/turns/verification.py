"""Observe the workspace revision associated with a principal-turn test."""

from __future__ import annotations

import json
import time
from typing import Literal

from infinidev.engine._best_effort import best_effort
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.guidance.test_runners import annotate_test_result_payload
from infinidev.tools.base.tool_effects import ToolEffects, local_effects_for_name


def may_change_workspace(name: str, effects: ToolEffects | None) -> bool:
    """Use the dispatch tool's declared effects or the built-in catalog."""
    if not isinstance(effects, ToolEffects) or effects.is_empty:
        effects = local_effects_for_name(name)
    return effects.writes_workspace or effects.runs_process or effects.mutates_git


class TurnTestVerification:
    """Retain the latest observed test status within one invocation."""

    def __init__(self, tracker: FileChangeTracker) -> None:
        self._tracker = tracker
        self._tested_revision: str | None = None
        self.status: Literal["current", "stale", "unconfirmed"] | None = None
        self.snapshot_ns = 0

    def snapshot(self) -> str | None:
        started = time.monotonic_ns()
        try:
            with best_effort("principal test workspace snapshot unavailable"):
                return self._tracker.workspace_revision()
            return None
        finally:
            self.snapshot_ns += time.monotonic_ns() - started

    def observe_test(self, revision_before: str | None, result: str) -> None:
        self._tested_revision = None
        self.status = "unconfirmed"
        if revision_before is None:
            return
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            return
        if (
            not isinstance(payload, dict)
            or type(payload.get("exit_code")) is not int
            or payload["exit_code"] != 0
            or payload.get("success") is False
            or payload.get("error")
            or annotate_test_result_payload(payload).get("test_output_conflict") is True
        ):
            return
        if self.snapshot() == revision_before:
            self._tested_revision = revision_before
            self.status = "current"

    def refresh(self) -> bool:
        """Check a current result and report whether it just became stale."""
        if self._tested_revision is None:
            return False
        revision = self.snapshot()
        if revision == self._tested_revision:
            return False
        self._tested_revision = None
        self.status = "stale" if revision is not None else "unconfirmed"
        return self.status == "stale"
