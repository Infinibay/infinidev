"""Running a user hook and turning its stdout into text the model reads.

Three rules govern everything here, and each one is a place a naive
implementation causes real damage:

**A hook can never fail the run.** It is the user's own command, run at
a point where the engine has already done real work — a non-zero exit, a
hang, a missing binary, a config typo. All of them cost the hook's
output and a log line, nothing more. This mirrors how the rest of the
engine treats optional machinery (working memory, ContextRank, Ken): the
feature degrades, the task completes.

**A hook can never hang the run.** Every command gets a deadline and is
killed at it. Without one, a hook that reads from stdin or waits on a
lock would freeze the loop between two steps with no way for the user to
tell what happened.

**A hook can never flood the context.** stdout goes straight into the
model's prompt, so it is capped. A hook that cats a log file should cost
a truncation notice, not the context window the task needs to finish.

The command runs through the shell, with the workspace as cwd, and gets
the event payload two ways: as JSON on stdin (the full structure) and as
``INFINIDEV_HOOK_*`` environment variables (the scalar fields, for the
one-liners people actually write). Neither is authoritative over the
other; they are the same data at two levels of ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Any

from infinidev.engine.subprocess_runner import run_captured
from infinidev.engine.user_hooks.config import HookSpec, get_hooks
from infinidev.engine.user_hooks.events import UserHookEvent

logger = logging.getLogger(__name__)

#: Ceiling on what one event may contribute to the prompt.
MAX_OUTPUT_CHARS = 8_000

_TRUNCATION_NOTICE = (
    "\n[… hook output truncated at {limit} characters — have the hook "
    "print less, or write the detail to a file and print the path.]"
)


@dataclass(frozen=True)
class HookOutput:
    """What an event's hooks produced, already capped and joined."""

    event: UserHookEvent
    text: str = ""
    ran: int = 0
    failed: int = 0
    labels: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        """Truthy only when there is text worth injecting."""
        return bool(self.text.strip())


def _truncate(text: str) -> str:
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return text[:MAX_OUTPUT_CHARS] + _TRUNCATION_NOTICE.format(
        limit=MAX_OUTPUT_CHARS
    )


def _env_for(payload: dict[str, Any], event: UserHookEvent) -> dict[str, str]:
    """Scalar payload fields as ``INFINIDEV_HOOK_*`` variables.

    Nested values are skipped rather than JSON-encoded: a hook that wants
    structure should read stdin, and an environment variable holding a
    serialised object is a trap in shell one-liners.
    """
    env = dict(os.environ)
    env["INFINIDEV_HOOK_EVENT"] = event.value
    for key, value in payload.items():
        if isinstance(value, bool):
            # Before the int branch: bool is a subclass of int, and
            # "True"/"False" is worse than "1"/"0" for `[ "$X" = 1 ]`.
            env[f"INFINIDEV_HOOK_{key.upper()}"] = "1" if value else "0"
        elif isinstance(value, (str, int, float)):
            env[f"INFINIDEV_HOOK_{key.upper()}"] = str(value)
    return env


def _run_one(
    spec: HookSpec, payload: dict[str, Any], workspace_path: str | None,
) -> str | None:
    """Execute one hook. Returns its output, or ``None`` when it produced none."""
    if spec.is_literal:
        return spec.prompt

    stdin_payload = ""
    try:
        stdin_payload = json.dumps(payload, default=str)
    except (TypeError, ValueError):
        logger.debug("hook payload not serialisable for %s", spec.event.value)

    try:
        completed = run_captured(
            spec.command,
            shell=True,
            cwd=workspace_path or None,
            timeout=spec.timeout,
            env=_env_for(payload, spec.event),
            input_text=stdin_payload,
        )
    except (OSError, ValueError) as exc:
        logger.warning(
            "hook %s (%s) could not be started: %s",
            spec.event.value, spec.label(), exc,
        )
        return None

    if completed.timed_out:
        logger.warning(
            "hook %s (%s) timed out after %.0fs — output discarded",
            spec.event.value, spec.label(), spec.timeout,
        )
        return None

    if completed.exit_code != 0:
        stderr = (completed.stderr or "").strip()
        logger.warning(
            "hook %s (%s) exited %d — output discarded%s",
            spec.event.value, spec.label(), completed.exit_code,
            f": {stderr[:400]}" if stderr else "",
        )
        return None

    return completed.stdout or ""


def run_hooks(
    event: UserHookEvent,
    payload: dict[str, Any] | None = None,
    *,
    workspace_path: str | None = None,
) -> HookOutput:
    """Run every hook bound to ``event`` and collect what they printed.

    Hooks run in declaration order and their outputs are concatenated.
    A hook that prints nothing contributes nothing — it is a perfectly
    good way to write a conditional hook, since the shell already knows
    how to stay quiet.
    """
    specs = get_hooks(event)
    if not specs:
        return HookOutput(event=event)

    payload = dict(payload or {})
    payload.setdefault("event", event.value)
    if workspace_path:
        payload.setdefault("workspace_path", workspace_path)

    chunks: list[str] = []
    labels: list[str] = []
    failed = 0
    for spec in specs:
        output = _run_one(spec, payload, workspace_path)
        if output is None:
            failed += 1
            continue
        stripped = output.strip()
        if not stripped:
            continue
        chunks.append(stripped)
        labels.append(spec.label())

    return HookOutput(
        event=event,
        text=_truncate("\n\n".join(chunks)),
        ran=len(specs),
        failed=failed,
        labels=labels,
    )
