"""Observe the workspace revision associated with a principal-turn test."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from infinidev.engine._best_effort import best_effort
from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.guidance.test_runners import annotate_test_result_payload, is_test_command
from infinidev.engine.turns.models import TestObservation
from infinidev.tools.base.tool_effects import ToolEffects, local_effects_for_name

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState


def may_change_workspace(name: str, effects: ToolEffects | None) -> bool:
    """Use the dispatch tool's declared effects or the built-in catalog."""
    if not isinstance(effects, ToolEffects) or effects.is_empty:
        effects = local_effects_for_name(name)
    return effects.writes_workspace or effects.runs_process or effects.mutates_git


class TurnTestVerification:
    """Track the latest check and revalidate a saved observation when resuming."""

    def __init__(self, tracker: FileChangeTracker, state: LoopState) -> None:
        self._tracker = tracker
        self._state = state
        self.observation: TestObservation | None = None
        self.result_matches_output = False
        self.from_prior_turn = False
        self.status: Literal["current", "stale", "unconfirmed"] | None = None
        self.snapshot_ns = 0

    def matches_test(self, command: object) -> bool:
        """Recognize the effective shell command using the active test declarations."""
        return isinstance(command, str) and is_test_command(command, self._state)

    def snapshot(self) -> str | None:
        started = time.monotonic_ns()
        try:
            with best_effort("principal test workspace snapshot unavailable"):
                return self._tracker.workspace_revision()
            return None
        finally:
            self.snapshot_ns += time.monotonic_ns() - started

    def _workspace_launch(self, cwd: object) -> bool:
        baseline = self._tracker.baseline
        if baseline is None or not isinstance(cwd, str) or "\0" in cwd or not os.path.isabs(cwd):
            return False
        try:
            launch = Path(os.path.realpath(cwd, strict=True))
            return launch.is_relative_to(baseline.root) and launch.is_dir()
        except OSError:
            return False

    def observe_test(
        self, revision_before: str | None, result: str, *, result_matches_output: bool,
    ) -> None:
        """Record the original result and whether the displayed output can locate it."""
        self.observation = None
        self.result_matches_output = False
        self.from_prior_turn = False
        self.status = "unconfirmed"
        if revision_before is None:
            return
        try:
            payload = json.loads(result)
        except (ValueError, TypeError):
            return
        if (
            not isinstance(payload, dict)
            or not self._workspace_launch(payload.get("launch_cwd"))
            or type(payload.get("exit_code")) is not int
            or payload["exit_code"] != 0
            or payload.get("success") is False
            or payload.get("error")
            or annotate_test_result_payload(payload).get("test_output_conflict") is True
        ):
            return
        if self.snapshot() == revision_before:
            self.observation = TestObservation(
                revision=revision_before, launch_cwd=payload["launch_cwd"],
            )
            self.result_matches_output = result_matches_output
            self.status = "current"

    def restore(self, observation: TestObservation) -> None:
        """Revalidate the saved observation before presenting it to the model."""
        self.observation = observation
        self.from_prior_turn = True
        self.status = "current"
        self.refresh()

    def refresh(self) -> bool:
        """Check a current result and report whether it just became stale."""
        if self.observation is None:
            return False
        if not self._workspace_launch(self.observation.launch_cwd):
            self.observation = None
            self.status = "unconfirmed"
            return False
        revision = self.snapshot()
        if revision == self.observation.revision:
            return False
        self.observation = None
        self.status = "stale" if revision is not None else "unconfirmed"
        return self.status == "stale"
