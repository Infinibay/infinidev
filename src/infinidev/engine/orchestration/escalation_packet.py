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
from typing import Any, Literal


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
    # Images the user attached in the chat turn that triggered the
    # escalation. Forwarded so the planner and the developer loop can
    # inject them into their own initial multimodal user message.
    # Typed loosely (``list``) to avoid an import cycle with
    # ``engine.multimodal``.
    attachments: list = field(default_factory=list)
    # ── Council (multi-agent deliberation) ──────────────────────────────
    # When the chat agent detects the user asked for a multi-agent
    # debate ("usá varios subagentes", "que debatan", "armá un consejo")
    # — or judged a design/research task complex enough — it sets
    # ``council_requested``. The pipeline then runs the council phase
    # (engine/council/) before the planner. ``council_focus`` narrows
    # the deliberation: "design", "research", or "both".
    council_requested: bool = False
    council_focus: Literal["design", "research", "both"] = "design"
    # The synthesised DesignBrief, attached AFTER the council runs (the
    # chat agent never sets this). Typed loosely to avoid importing
    # engine.council at packet-definition time.
    design_brief: Any | None = None
    # The GroundedSpec produced by the spec-elaboration loop, attached
    # BEFORE the council/planner run (the chat agent never sets this).
    # Turns the vague request into a grounded spec the planner builds on.
    # Typed loosely (``engine.analysis.grounded_spec.GroundedSpec``) to
    # avoid an import at packet-definition time.
    grounded_spec: Any | None = None
