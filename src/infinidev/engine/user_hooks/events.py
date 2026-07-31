"""The six points where a user's own command can enter the run.

These are *user* hooks — commands declared in ``.infinidev/hooks.json`` —
and they are a different thing from :mod:`infinidev.engine.hooks`, whose
``HookEvent`` names in-process Python callbacks the engine itself
registers. The two never mix: an in-process hook can rewrite a tool call,
a user hook can only contribute text.

The lifecycle is symmetric — a task and a step each get one start hook and
*two* end hooks — and the pair at each end is the whole point of the
design:

``*_end_instruction``
    Fires when the model claims it is finished, and its output is handed
    back as work still to do. The step or task stays open for one more
    pass. This is the "run a deep review before you call it done" slot.
    Deliberately **once** per step / per task: a hook that fired again on
    the retry would be a loop with no exit, since the model's second
    ``step_complete`` looks exactly like its first.

``*_end_summary``
    Fires when the step or task really closes, and its output is written
    where the summarisation cannot evict it — onto the ``ActionRecord``
    for a step, onto the hidden work summary for a task.

That distinction is the reason there are two and not one. Everything a
step does is compacted into a ~50-token summary at the step boundary
(see ``engine/loop/step_manager.py``), so an instruction injected
mid-step is *supposed* to evaporate: it was scaffolding, it did its job,
and carrying it forward would just crowd the prompt. A summary hook's
output is the opposite — it is meant to be read by later steps, so it
rides on the record that survives.
"""

from __future__ import annotations

from enum import Enum


class UserHookEvent(str, Enum):
    """Every event a user hook can be bound to, as written in the config."""

    TASK_START = "task_start"
    STEP_START = "step_start"
    STEP_END_INSTRUCTION = "step_end_instruction"
    STEP_END_SUMMARY = "step_end_summary"
    TASK_END_INSTRUCTION = "task_end_instruction"
    TASK_END_SUMMARY = "task_end_summary"

    @classmethod
    def parse(cls, raw: str) -> "UserHookEvent | None":
        """Return the event named ``raw``, or ``None`` if it isn't one.

        Config comes from a hand-edited file, so an unknown key is a typo,
        not a crash. Callers log it and move on.
        """
        try:
            return cls(str(raw).strip())
        except ValueError:
            return None


#: Events whose output is injected as work to do rather than as a note.
INSTRUCTION_EVENTS = frozenset({
    UserHookEvent.TASK_END_INSTRUCTION,
    UserHookEvent.STEP_END_INSTRUCTION,
})
