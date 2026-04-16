"""Handoff payload from the chat agent to the analyst planner.

The chat agent is read-only by design — when it concludes that the
user's conversation is pointing at real work, it calls the
``escalate`` tool and returns an ``EscalationPacket``. The packet is
the single source of truth the planner reads when deciding what to
break into steps. No parallel enriched-prompt strings, no
``specification`` dicts — this dataclass is the contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class EscalationPacket:
    """Data passed from the chat agent to the analyst planner.

    Attributes:
        user_request: The user's original message that triggered
            escalation (verbatim, no paraphrasing).
        understanding: One or two sentences, written by the chat agent,
            summarising what the user wants in its own words. The
            planner uses this to validate that the handoff isn't
            confused.
        suggested_flow: v1 restricts this to ``"develop"``; future
            versions may allow other targets like ``"sysadmin"``.
        opened_files: Files the chat agent already read during its
            conversational turn. The planner SHOULD NOT re-open these
            — the point of the handoff is to avoid redundant I/O.
        user_visible_preview: Short "voy a implementar X" message the
            pipeline shows to the user via ``notify`` before the
            planner runs. Avoids dead-air while the planner thinks.
        user_signal: The literal user text the chat agent interpreted
            as approval (e.g. "sí, dale"). Kept for audit; helps debug
            false-positive escalations later.
    """

    user_request: str
    understanding: str
    suggested_flow: Literal["develop"] = "develop"
    opened_files: list[str] = field(default_factory=list)
    user_visible_preview: str = ""
    user_signal: str = ""
