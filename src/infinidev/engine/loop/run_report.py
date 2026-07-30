"""What a finished run has to say for itself.

Everything downstream of the loop — the review engine, the plan-fidelity
check, the post-loop objective re-verification, the hidden hand-off summary
the next chat turn reads — asks the engine the same kind of question: *what
changed, and against what plan?* None of it needs the engine, only the two
artefacts a run leaves behind: the ``FileChangeTracker`` and the final
``LoopState``.

Keeping these as free functions over those two artefacts means the reporting
is testable without booting a loop, and it keeps the engine's own surface
about *running*, which is the one thing only it can do.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from infinidev.engine._best_effort import best_effort

logger = logging.getLogger(__name__)


def changed_files_summary(tracker: Any | None) -> str:
    """A diff-per-file digest of the run, for the code reviewer.

    Empty string when nothing changed — the reviewer treats that as
    "nothing to review" rather than as an error.
    """
    if tracker is None:
        return ""
    paths = tracker.get_all_paths()
    if not paths:
        return ""

    parts: list[str] = []
    for path in paths:
        action = tracker.get_action(path)
        diff = tracker.get_diff(path)
        if diff:
            parts.append(f"### {path} ({action})\n```diff\n{diff}\n```")
        else:
            parts.append(f"### {path} ({action}, no diff)")
    return "\n\n".join(parts)


def has_file_changes(tracker: Any | None) -> bool:
    """Whether the run touched anything on disk."""
    return bool(tracker is not None and tracker.get_all_paths())


def file_change_reasons(tracker: Any | None) -> dict[str, list[str]]:
    """Path → why each edit was made, as recorded at edit time."""
    if tracker is None:
        return {}
    return {
        path: reasons
        for path in tracker.get_all_paths()
        if (reasons := tracker.get_reasons(path))
    }


def file_contents(tracker: Any | None) -> dict[str, str]:
    """Path → current content, for every file the run changed.

    Files that vanished or grew past the tracking limit are skipped rather
    than raising: a reviewer working from a partial set is still useful, a
    crashed finish path is not.
    """
    if tracker is None:
        return {}
    from infinidev.engine.tool_executor import MAX_TRACK_FILE_SIZE

    result: dict[str, str] = {}
    for path in tracker.get_all_paths():
        with best_effort("failed to read tracked file %s", path):
            if os.path.isfile(path) and os.path.getsize(path) <= MAX_TRACK_FILE_SIZE:
                with open(path, "r", encoding="utf-8", errors="replace") as handle:
                    result[path] = handle.read()
    return result


def plan_steps(state: Any | None) -> list[dict]:
    """The plan as executed, for the reviewer's plan-fidelity check.

    Under the chat-agent-first pipeline the engine really is seeded with a
    multi-step, user-approved plan, so this has something to check against;
    it used to return ``[]`` unconditionally, which quietly made that check
    dead code.
    """
    if state is None or not getattr(state, "plan", None):
        return []
    return [
        {
            "step": step.index,
            "title": step.title,
            "explanation": step.detail or step.explanation,
            "status": step.status,
        }
        for step in state.plan.steps
    ]


def objective_checks(state: Any | None) -> list[tuple[int, str, Any]]:
    """``(step_index, title, StepVerification)`` for every verifiable step.

    The post-loop review re-runs all of these together, which is what
    catches the regression the per-step gate structurally cannot see: step
    3's edit quietly breaking step 1's already-green check.
    """
    if state is None or not getattr(state, "plan", None):
        return []
    return [
        (step.index, step.title, verify)
        for step in state.plan.steps
        if (verify := getattr(step, "verify", None)) is not None
        and verify.is_executable
    ]


def work_summary(
    state: Any | None,
    tracker: Any | None,
    *,
    final_answer: str,
    status: str,
) -> str | None:
    """Distil the finished task into a hidden hand-off summary.

    Returns ``None`` when there is nothing worth recording or the feature is
    off. Never raises: this runs on the finish path, and a failed summary
    must not cost the user a completed task.
    """
    from infinidev.engine.loop.work_summary import build_work_summary

    try:
        return build_work_summary(
            state, tracker, final_answer=final_answer, status=status,
        )
    except Exception:
        logger.warning("build_work_summary failed", exc_info=True)
        return None
