"""Tools exclusive to the council tiers.

Like the chat-agent and planner terminators, these are schema-level
pseudo-tools: the council orchestrator reads the LLM's tool_call args
directly and never routes them through normal dispatch. ``_run`` returns
a JSON acknowledgement as a safe fallback.

Two tiers:
  * ``council_member`` — ``channel_post`` (contribute to the debate) and
    ``conclude`` (leave the council). Plus read-only exploration tools.
  * ``council_moderator`` — ``seed_council`` (assign personas/objectives
    and open threads), ``council_verdict`` (judge convergence each
    round), ``synthesize_brief`` (emit the final DesignBrief). Plus
    read-only exploration tools.
"""

from infinidev.tools.council.channel_tools import ChannelPostTool, ConcludeTool
from infinidev.tools.council.moderator_tools import (
    CouncilVerdictTool,
    SeedCouncilTool,
    SynthesizeBriefTool,
)

COUNCIL_MEMBER_TOOLS = [ChannelPostTool, ConcludeTool]
COUNCIL_MODERATOR_TOOLS = [SeedCouncilTool, CouncilVerdictTool, SynthesizeBriefTool]

__all__ = [
    "ChannelPostTool",
    "ConcludeTool",
    "SeedCouncilTool",
    "CouncilVerdictTool",
    "SynthesizeBriefTool",
    "COUNCIL_MEMBER_TOOLS",
    "COUNCIL_MODERATOR_TOOLS",
]
