"""Result shape returned by the chat agent orchestrator.

A chat-agent turn ends in exactly one of two states: the agent
produced a conversational reply (``kind="respond"``) and the turn is
over, OR the agent decided real work is needed (``kind="escalate"``)
and the pipeline must continue to the planner.

Keeping the two states in a single tagged dataclass makes the
pipeline branch exhaustive — you can't forget to handle one of them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from infinidev.engine.orchestration.escalation_packet import EscalationPacket


@dataclass(frozen=True)
class ChatAgentResult:
    """Output of a single chat-agent turn.

    Invariants enforced at construction time:
      * ``kind="respond"`` → ``reply`` is non-empty, ``escalation`` is None
      * ``kind="escalate"`` → ``escalation`` is not None

    ``streamed`` is True when the chat agent ran in streaming mode and
    already emitted plain text incrementally via
    ``hooks.notify_stream_chunk``. On the ``respond`` path the pipeline
    uses it to decide whether to call ``hooks.notify`` for the reply
    (double-render otherwise). It is *also* consulted on the
    ``escalate`` path: when True, the pipeline calls
    ``hooks.notify_stream_end`` to finalize a streaming bubble that was
    opened by plain-text content emitted before the ``escalate`` tool
    fired (otherwise that bubble stays stuck in raw-markdown mode).
    """

    kind: Literal["respond", "escalate"]
    reply: str = ""
    escalation: EscalationPacket | None = None
    streamed: bool = False
    # Populated only by the exception-fallback path — the raw traceback
    # that would otherwise be invisible to the user. The UI renders it
    # inside a collapsed widget so it doesn't clutter the chat unless the
    # user explicitly expands it.
    error_traceback: str | None = None

    def __post_init__(self) -> None:
        if self.kind == "respond":
            if not self.reply.strip():
                raise ValueError(
                    "ChatAgentResult(kind='respond') requires non-empty reply"
                )
            if self.escalation is not None:
                raise ValueError(
                    "ChatAgentResult(kind='respond') must not carry an escalation"
                )
        elif self.kind == "escalate":
            if self.escalation is None:
                raise ValueError(
                    "ChatAgentResult(kind='escalate') requires an escalation packet"
                )
        else:
            raise ValueError(f"Invalid kind: {self.kind!r}")
