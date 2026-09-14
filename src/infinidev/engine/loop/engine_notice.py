"""Corrective notices the engine owes the model after rejecting a closure.

``StepCompleteGate`` runs *inside* the inner loop, so it can answer the
``step_complete`` tool call directly: the model reads "your close was
overridden" in the tool result and corrects itself on the next turn.

Two gates live outside that loop, in ``LoopEngine.execute``:
``_enforce_edit_requirement`` and ``_enforce_step_effect``. They decide after
the inner loop has returned, and the ``messages`` list they could overwrite is
discarded and rebuilt from scratch on the next iteration. A refusal that only
flips ``StepResult`` therefore reaches the model as *nothing at all*: the next
prompt is byte-identical to the one that produced the rejected closure, so the
model repeats the same ``step_complete`` and the engine repeats the same
refusal. Measured on the MiniMax-M3 baseline, one documentation task spent ten
of its eleven iterations in exactly that cycle.

This module is the missing channel. A refusal queues an imperative notice that
``build_iteration_prompt`` renders once, above the plan, and consumes. The
notice names the observation that blocked the close and the exact call that
unblocks it, because "you have not edited anything" is only actionable if the
model knows which tool the engine is waiting for.

The per-Step refusal counter bounds the exchange. Production runs with
``TASK_MAX_ITERATIONS = 0``, so the iteration budget is not a bound; without
this counter a gate that can refuse forever would spin forever.
"""

from __future__ import annotations

import re
from typing import Any

#: Consecutive refusals for one Step before the engine stops negotiating and
#: closes the run. Three leaves room for a genuine wrong turn plus a
#: correction, without paying for the same rejected closure indefinitely.
MAX_CLOSURE_REFUSALS = 3

#: Handoff refusals are counted per Task, not per Step, and the bound is one.
#: The handoff is a wording defect in the deliverable, not a missing piece of
#: work, so a second round of it would be the engine arguing about prose.
MAX_HANDOFF_REFUSALS = 1

#: Tokens a final answer can use to name the check it ran. Conservative on
#: purpose: a false positive here costs the user a round, so a runner has to be
#: a whole word and a command has to follow it.
_COMMAND_HINT = re.compile(
    r"`[^`]*\b(?:pytest|python3?|uv|pip|npm|npx|yarn|pnpm|make|cargo|go|bash|sh"
    r"|ruff|mypy|flake8|black|node|deno|bundle|rake|gradle|mvn|dotnet|php|ruby)\b[^`]*`"
    r"|\b(?:pytest|python3?|uv|npm|npx|make|cargo|ruff|mypy|deno)\s+[-\w./]",
    re.IGNORECASE,
)


def names_a_verification_command(text: str) -> bool:
    """Whether *text* names a command the reader could run to check the work.

    Used to decide one thing only: the model said a check ran in
    ``evidence_summary`` and then handed the user an answer that names no
    command at all. Measured over 25 planning runs, 18 named one, so the
    engine stays quiet the other 7 rather than guessing at prose quality.
    """

    return bool(_COMMAND_HINT.search(str(text or "")))


_HEADER = '<engine-notice priority="critical" reason="closure-rejected">'


def _box(title: str, body: str) -> str:
    return f"{_HEADER}\n{title}\n\n{body}\n</engine-notice>"


def build_edit_requirement_notice(*, attempt: int) -> str:
    """Notice for a write Task that claimed completion before any edit."""
    escalation = ""
    if attempt >= 2:
        escalation = (
            f"\nThis is refusal {attempt} of {MAX_CLOSURE_REFUSALS} for this Step. "
            "One more identical close ends the run without the work.\n"
        )
    return _box(
        'Your step_complete(status="done") was REJECTED — the Task is still open.',
        (
            "Why: this Task changes the repository and the engine has observed no "
            "successful workspace edit. Reading, searching, planning and summarising "
            "do not satisfy a write Task.\n\n"
            "Do this now, in order:\n"
            "1. Call `edit_file` on an existing file, or `create_file` for a new one, "
            "with the actual change the Task asks for.\n"
            "2. Run the check that exercises it (test, script, or assertions).\n"
            "3. Call `step_complete` again, quoting the observed result in "
            "`evidence_summary`.\n\n"
            "If the requested behaviour genuinely already exists and needs no change, "
            "do not repeat the same close: re-run the call with `no_edit=true` and put "
            "the evidence that proves the behaviour is already present in "
            "`evidence_summary`."
            f"{escalation}"
        ),
    )


def build_step_effect_notice(
    *,
    step_index: int,
    step_title: str,
    attempt: int,
) -> str:
    """Notice for an implementation Step that produced no net workspace change."""
    escalation = ""
    if attempt >= 2:
        escalation = (
            f"\nThis is refusal {attempt} of {MAX_CLOSURE_REFUSALS} for Step "
            f"{step_index}. Repeating `step_complete` without new evidence will end "
            "the run as blocked.\n"
        )
    return _box(
        f"Your step_complete was REJECTED — Step {step_index} is still active.",
        (
            f'Step {step_index} ("{step_title}") is an implementation Step, and the '
            "workspace shows no net change since it became active. A Step whose own "
            "name promises a change has to produce one: reads, searches, notes and "
            "re-summaries cannot close it.\n\n"
            "Choose exactly one and act on it in your next tool call:\n"
            "- The change is real and still missing: make it now with `edit_file` "
            "(`create_file` for a new file), then verify and close.\n"
            "- The change already landed earlier in this Task and this Step is "
            "redundant: call `modify_step` to retitle it as the concrete remaining "
            "work, or `remove_step` it, and close with `status=\"continue\"`.\n"
            "- The Step is really verification or reporting, not a change: "
            "`modify_step` its title so it does not claim an edit, then close.\n"
            f"{escalation}"
        ),
    )


def build_handoff_notice(*, evidence: str, attempt: int) -> str:
    """Notice for a `done` whose final answer names no check the user can run.

    Not a claim that the work is wrong. The engine only fires this when the
    model itself wrote a check into ``evidence_summary`` and then left it out
    of the answer the user reads, so the notice quotes it back rather than
    asking for a judgment the engine cannot make.
    """
    quoted = evidence.strip()[:400] or "(none recorded)"
    return _box(
        'Your step_complete(status="done") was REJECTED — one more line, then it '
        "closes.",
        (
            "Why: it is not a defect in the deliverable, it is a gap in what the "
            "user can check. You reported this check in `evidence_summary`:\n\n"
            f"    {quoted}\n\n"
            "Your `final_answer` names no command at all, so the user has to take "
            "the result on trust.\n\n"
            "Call `step_complete` again with `status=\"done\"`, the same summary "
            "and the same evidence, and a `final_answer` that keeps the outcome "
            "first and then adds one `Verification:` line naming the exact command "
            "you ran and what it returned. If nothing was actually run, say what "
            "you did instead — a limitation stated is worth more than a claim."
            + (
                f"\nThis is handoff refusal {attempt} of {MAX_HANDOFF_REFUSALS}; "
                "the next close is accepted as it stands."
                if attempt >= MAX_HANDOFF_REFUSALS
                else ""
            )
        ),
    )


def note_handoff_refusal(state: Any) -> int:
    """Increment and return the per-Task handoff refusal count."""
    if state is None:
        return 1
    state.handoff_refusals = int(getattr(state, "handoff_refusals", 0) or 0) + 1
    return state.handoff_refusals


def clear_handoff_refusals(state: Any) -> None:
    if state is not None:
        state.handoff_refusals = 0


def queue_engine_notice(state: Any, text: str) -> None:
    """Store *text* as the notice the next prompt build will render."""
    if state is None or not text:
        return
    state.pending_engine_notice = text


def drain_engine_notice(state: Any) -> str:
    """Pop the queued notice, or "" when there is none.

    Called by ``build_iteration_prompt`` once per iteration. Idempotent.
    """
    text = str(getattr(state, "pending_engine_notice", "") or "")
    if text:
        state.pending_engine_notice = ""
    return text


def note_closure_refusal(state: Any, step_index: int) -> int:
    """Increment and return the consecutive-refusal count for *step_index*."""
    if state is None:
        return 1
    counters = getattr(state, "effect_refusals_by_step", None)
    if counters is None:
        return 1
    counters[step_index] = int(counters.get(step_index, 0)) + 1
    return counters[step_index]


def clear_closure_refusals(state: Any, step_index: int) -> None:
    """Forget the refusal count once a Step closes or the plan moves on."""
    counters = getattr(state, "effect_refusals_by_step", None)
    if counters is not None:
        counters.pop(step_index, None)


__all__ = [
    "MAX_CLOSURE_REFUSALS",
    "MAX_HANDOFF_REFUSALS",
    "build_edit_requirement_notice",
    "build_handoff_notice",
    "build_step_effect_notice",
    "clear_closure_refusals",
    "clear_handoff_refusals",
    "names_a_verification_command",
    "note_handoff_refusal",
    "drain_engine_notice",
    "note_closure_refusal",
    "queue_engine_notice",
]
