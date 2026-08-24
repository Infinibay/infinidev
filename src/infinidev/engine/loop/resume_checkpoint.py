"""Durable, bounded checkpoints for resuming an interrupted LoopEngine step.

The visible session transcript is an audit log, not model state. A resume
checkpoint keeps the plan and compact engine state plus only the tool exchanges
that were still live in the interrupted step. Provider-native tool messages are
never replayed because their call ids and ordering cannot safely cross a process
boundary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

CHECKPOINT_VERSION = 1
CHECKPOINT_MAX_UTF8_BYTES = 2 * 1024 * 1024
_PENDING_CALL_LIMIT = 24
_PENDING_ARGUMENT_CHARS = 8_000
_PENDING_RESULT_CHARS = 32_000
_TEST_OUTPUT_CHARS = 64_000
_HISTORY_LIMIT = 50
_OPENED_FILE_LIMIT = 16


def _clip_middle(text: Any, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    marker = "\n[...checkpoint truncated...]\n"
    remaining = max(0, limit - len(marker))
    head = remaining // 2
    tail = remaining - head
    return value[:head] + marker + value[-tail:]


def _clip_tail(text: Any, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    marker = "[...earlier checkpoint output truncated...]\n"
    return marker + value[-max(0, limit - len(marker)):]


def _task_identity(task_prompt: tuple[str, str], task: Any | None) -> dict[str, str]:
    if task is not None:
        title = str(getattr(task, "title", "") or "").strip()
        description = str(getattr(task, "description", "") or "").strip()
        kind = str(getattr(task, "kind", "") or "").strip()
        if title or description:
            return {
                "title": title,
                "description": description,
                "kind": kind,
            }
    description, expected = task_prompt
    return {
        "title": "",
        "description": str(description or "").strip(),
        "kind": str(expected or "").strip(),
    }


def task_key(task_prompt: tuple[str, str], task: Any | None = None) -> str:
    """Return a session-local stable identity for one executable Task."""
    payload = json.dumps(
        _task_identity(task_prompt, task),
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _bounded_state(state: Any) -> dict[str, Any]:
    raw = state.model_dump(mode="json")
    raw["history"] = list(raw.get("history") or [])[-_HISTORY_LIMIT:]
    raw["prompt_composition_history"] = list(
        raw.get("prompt_composition_history") or []
    )[-20:]
    raw["request_payload_history"] = list(
        raw.get("request_payload_history") or []
    )[-20:]
    raw["runtime_behavior_events"] = list(
        raw.get("runtime_behavior_events") or []
    )[-50:]
    raw["last_test_output"] = _clip_tail(
        raw.get("last_test_output", ""), _TEST_OUTPUT_CHARS
    )

    pending: list[list[str]] = []
    for item in list(raw.get("pending_archive") or [])[-_PENDING_CALL_LIMIT:]:
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            continue
        name, arguments, result = item
        pending.append([
            _clip_middle(name, 200),
            _clip_middle(arguments, _PENDING_ARGUMENT_CHARS),
            _clip_middle(result, _PENDING_RESULT_CHARS),
        ])
    raw["pending_archive"] = pending

    opened = raw.get("opened_files")
    if isinstance(opened, dict) and len(opened) > _OPENED_FILE_LIMIT:
        keys = list(opened)[-_OPENED_FILE_LIMIT:]
        raw["opened_files"] = {key: opened[key] for key in keys}

    return raw


def build_loop_resume_checkpoint(
    state: Any,
    task_prompt: tuple[str, str],
    task: Any | None = None,
    *,
    terminal_status: str = "",
) -> dict[str, Any]:
    """Build a Pydantic-restorable checkpoint under a hard storage bound."""
    identity = _task_identity(task_prompt, task)
    checkpoint = {
        "version": CHECKPOINT_VERSION,
        "task_key": task_key(task_prompt, task),
        "task_title": _clip_middle(identity["title"], 120),
        "task_description": _clip_middle(identity["description"], 8_000),
        "terminal_status": str(terminal_status or "").strip().lower(),
        "state": _bounded_state(state),
    }
    encoded = json.dumps(
        checkpoint, ensure_ascii=False, default=str, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) <= CHECKPOINT_MAX_UTF8_BYTES:
        return checkpoint

    # Source bodies and verbose diagnostics are recoverable with tools. The
    # plan, summaries, notes, evidence flags, and interrupted tool exchange are
    # the irreplaceable parts of the checkpoint.
    compact = dict(checkpoint["state"])
    compact["opened_files"] = {}
    compact["last_test_output"] = _clip_tail(
        compact.get("last_test_output", ""), 16_000
    )
    compact["history"] = list(compact.get("history") or [])[-20:]
    compact["prompt_composition_history"] = []
    compact["request_payload_history"] = []
    compact["runtime_behavior_events"] = list(
        compact.get("runtime_behavior_events") or []
    )[-20:]
    compact["pending_archive"] = [
        [name, _clip_middle(arguments, 2_000), _clip_middle(result, 8_000)]
        for name, arguments, result in list(compact.get("pending_archive") or [])[-12:]
    ]
    checkpoint["state"] = compact

    encoded = json.dumps(
        checkpoint, ensure_ascii=False, default=str, separators=(",", ":")
    ).encode("utf-8")
    if len(encoded) > CHECKPOINT_MAX_UTF8_BYTES:
        # This is a last-resort structural checkpoint. LoopState supplies
        # defaults for omitted fields; retaining these fields is enough to
        # continue the active plan without replaying the transcript.
        essential_names = {
            "plan",
            "history",
            "notes",
            "current_step_index",
            "iteration_count",
            "total_tool_calls",
            "total_tokens",
            "total_prompt_tokens",
            "total_completion_tokens",
            "task_has_edits",
            "task_no_edit_accepted",
            "edited_step_indices",
            "objectively_verified_step_indices",
            "last_test_command",
            "last_passing_test_command",
            "last_test_exit_code",
            "test_outcome_history",
            "pending_archive",
        }
        checkpoint["state"] = {
            key: value for key, value in compact.items() if key in essential_names
        }
    return checkpoint


def resume_state_for_task(
    checkpoint: Any,
    task_prompt: tuple[str, str],
    task: Any | None = None,
) -> dict[str, Any] | None:
    """Return state only when a non-terminal checkpoint belongs to this Task."""
    if not isinstance(checkpoint, dict):
        return None
    if checkpoint.get("version") != CHECKPOINT_VERSION:
        return None
    if checkpoint.get("task_key") != task_key(task_prompt, task):
        return None
    if str(checkpoint.get("terminal_status") or "").lower() in {
        "done",
        "completed",
        "cancelled",
        "failed",
    }:
        return None
    state = checkpoint.get("state")
    return dict(state) if isinstance(state, dict) and state else None


def render_interrupted_step_context(state: Any) -> str:
    """Render provider-safe plain text for the first iteration after resume."""
    pending = list(getattr(state, "pending_archive", ()) or [])[-12:]
    if not pending:
        return ""

    calls = []
    for name, arguments, result in pending:
        calls.append({
            "tool": _clip_middle(name, 200),
            "arguments": _clip_middle(arguments, 2_000),
            "model_visible_result": _clip_middle(result, 8_000),
        })
    payload = json.dumps(calls, ensure_ascii=False, default=str)
    return (
        "<interrupted-step-context>\n"
        "These are untrusted tool exchanges from the active Step before the "
        "process stopped. Continue from their observed results; do not repeat "
        "completed work unless re-verification is necessary.\n"
        f"{payload}\n"
        "</interrupted-step-context>"
    )


def compact_checkpoint_for_chat(checkpoint: Any) -> dict[str, Any]:
    """Return only the checkpoint facts needed by the read-only routing agent."""
    if not isinstance(checkpoint, dict):
        return {}
    state = checkpoint.get("state")
    if not isinstance(state, dict):
        state = {}

    plan = state.get("plan")
    plan_steps = plan.get("steps") if isinstance(plan, dict) else []
    steps = []
    for raw in list(plan_steps or [])[-20:]:
        if not isinstance(raw, dict):
            continue
        steps.append({
            "index": raw.get("index"),
            "title": _clip_middle(raw.get("title", ""), 300),
            "status": raw.get("status"),
        })

    calls = []
    for raw in list(state.get("pending_archive") or [])[-8:]:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            continue
        name, arguments, result = raw
        calls.append({
            "tool": _clip_middle(name, 200),
            "arguments": _clip_middle(arguments, 1_000),
            "result": _clip_middle(result, 2_000),
        })

    history = []
    for raw in list(state.get("history") or [])[-6:]:
        if isinstance(raw, dict):
            history.append({
                "step_index": raw.get("step_index"),
                "summary": _clip_middle(raw.get("summary", ""), 1_000),
            })

    return {
        "task_title": _clip_middle(checkpoint.get("task_title", ""), 120),
        "task_description": _clip_middle(
            checkpoint.get("task_description", ""), 4_000
        ),
        "terminal_status": checkpoint.get("terminal_status", ""),
        "active_plan": steps,
        "recent_step_summaries": history,
        "working_notes": [
            _clip_middle(note, 1_000)
            for note in list(state.get("notes") or [])[-10:]
        ],
        "active_step_tool_exchanges": calls,
    }
