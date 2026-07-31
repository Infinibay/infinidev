"""How hook output is worded when it reaches the model.

Wording is not cosmetic here. The engine's prompt is XML-structured and
the model is trained by every other block in it to treat a tag as a kind
of content — so hook output arrives in a tag of its own, and the tag says
who wrote it. Two failure modes this avoids:

- Text pasted bare into a tool result reads as engine output, and the
  model may argue with it or try to "fix" it. Attributed text reads as an
  instruction from the user's configuration, which is what it is.
- An instruction that does not say *what happens next* leaves the model
  guessing whether it should re-call ``step_complete``. Every instruction
  block ends by telling it exactly that, because the alternative is a
  step that either stalls or closes without doing the work.

Hook text is the user's own, so it is never reformatted or truncated
mid-sentence here — only wrapped.
"""

from __future__ import annotations

from infinidev.engine.user_hooks.events import UserHookEvent

_STEP_INSTRUCTION = (
    "step_complete HELD — a configured hook has additional work for this "
    "step before it can close.\n\n"
    "<hook-instruction>\n{text}\n</hook-instruction>\n\n"
    "Carry out the instruction above, then call step_complete again. This "
    "hook fires once per step: your next step_complete will be honoured on "
    "its own merits."
)

_TASK_INSTRUCTION = (
    "<hook-instruction>\n{text}\n</hook-instruction>\n\n"
    "The work above was requested by a configured end-of-task hook, not by "
    "the user directly. Carry it out against the changes already made in "
    "this session."
)

_CONTEXT_BLOCK = '<hook-output event="{event}">\n{text}\n</hook-output>'


def step_instruction(text: str) -> str:
    """Body that replaces the ``step_complete`` tool result to hold a step."""
    return _STEP_INSTRUCTION.format(text=text.strip())


def task_instruction(text: str) -> str:
    """Input for the re-entered turn when a task-end hook asks for more work."""
    return _TASK_INSTRUCTION.format(text=text.strip())


def context_block(event: UserHookEvent, text: str) -> str:
    """Attributed block for hook output that is context rather than work."""
    return _CONTEXT_BLOCK.format(event=event.value, text=text.strip())
