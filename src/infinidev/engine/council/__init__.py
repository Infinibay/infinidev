"""Council — an opt-in multi-agent deliberation phase.

The council sits between the chat agent's ``escalate`` and the analyst
planner. When the user asks for it (or the chat agent flags a complex
design/research task), the orchestrator (a *moderator* agent) spins up
N short-lived subagents, each with its own persona, objective, and
isolated tool context. They debate on a shared **channel** of threads
(a blackboard) over a few rounds, and the moderator synthesises a
:class:`DesignBrief` that enriches the escalation handoff.

Design constraints (see the design discussion that produced this):

  * **Design/research only.** The council never writes code. Its tools
    are read-only exploration plus channel-posting. Execution stays
    mono-agent (planner → developer), exactly as before.
  * **Moderator assigns roles.** Personas and per-member objectives are
    generated per task by the moderator, not drawn from a fixed table.
    ``personas.py`` is a starter palette the moderator references.
  * **Convergence is judged, with a hard cap.** The moderator decides
    each round whether the debate converged; ``COUNCIL_MAX_ROUNDS`` is
    the runaway guard.
  * **User approval is conditional.** The brief carries
    ``user_decision_required`` — the pipeline only interrupts the user
    when the council hits a genuine product fork it must not decide
    alone.

The public entry point is :func:`run_council`.
"""

from __future__ import annotations

from infinidev.engine.council.brief import (
    CouncilRoster,
    DesignBrief,
    MemberAssignment,
)
from infinidev.engine.council.channel import Channel, Message, Thread

__all__ = [
    "run_council",
    "Channel",
    "Thread",
    "Message",
    "DesignBrief",
    "CouncilRoster",
    "MemberAssignment",
]


def run_council(*args, **kwargs):
    """Lazy proxy to :func:`engine.council.runner.run_council`.

    Importing ``runner`` eagerly would pull in litellm and the tool
    registry at package-import time, which slows cold start and risks
    the same circular-import trap the pipeline guards against. The proxy
    keeps ``from infinidev.engine.council import run_council`` cheap.
    """
    from infinidev.engine.council.runner import run_council as _impl

    return _impl(*args, **kwargs)
