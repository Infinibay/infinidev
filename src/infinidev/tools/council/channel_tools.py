"""Member-tier council terminators: ``channel_post`` and ``conclude``.

A council member ends each round-turn by either contributing to the
debate (``channel_post``) or signalling it is done participating
(``conclude``). Read-only exploration tools (file reads, code intel,
web) do NOT end the turn — only these two do. That keeps each member's
per-round cost bounded to "explore a little, then say one thing".
"""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class ChannelPostInput(BaseModel):
    message: str = Field(
        ...,
        description=(
            "Your contribution this round, in the user's language. Be "
            "concrete: a proposal, a critique of a specific other "
            "message, a research finding, or a refinement. Reference "
            "other messages by their id when you respond to them."
        ),
    )
    thread_id: str = Field(
        "",
        description=(
            "The thread to post into (e.g. 't-2'). Leave empty AND set "
            "new_thread_title to open a brand-new thread instead."
        ),
    )
    new_thread_title: str = Field(
        "",
        description=(
            "If set, opens a new thread with this title and posts your "
            "message as its opener. Use when your point doesn't belong "
            "in any existing thread. Mutually exclusive with thread_id."
        ),
    )
    parent_id: str = Field(
        "",
        description=(
            "Optional id of the specific message you are replying to "
            "(e.g. 'm-7'), for threaded back-and-forth."
        ),
    )
    refs: list[str] = Field(
        default_factory=list,
        description=(
            "Files or symbols you cite as evidence (e.g. "
            "['engine/loop/engine.py', 'LoopEngine.execute']). Grounds "
            "your claim so others can verify it."
        ),
    )


class ChannelPostTool(InfinibayBaseTool):
    is_read_only: bool = True  # posts to an in-memory channel, never the FS
    name: str = "channel_post"
    description: str = (
        "Post your contribution to the shared council channel and end "
        "your turn for this round. Call this EXACTLY once per round, "
        "after any read-only exploration. Either post into an existing "
        "thread (thread_id) or open a new one (new_thread_title)."
    )
    args_schema: Type[BaseModel] = ChannelPostInput

    def _run(
        self,
        message: str,
        thread_id: str = "",
        new_thread_title: str = "",
        parent_id: str = "",
        refs: list | None = None,
    ) -> str:
        # Schema-level terminator: the council runner reads tool_call
        # args directly. This _run is the safe fallback.
        return json.dumps({
            "kind": "channel_post",
            "message": message,
            "thread_id": thread_id,
            "new_thread_title": new_thread_title,
            "parent_id": parent_id,
            "refs": refs or [],
        })


class ConcludeInput(BaseModel):
    final_position: str = Field(
        ...,
        description=(
            "Your settled stance on the question, in 1-3 sentences. This "
            "is what the moderator weighs when synthesising the design "
            "brief. State it even if it agrees with the emerging consensus."
        ),
    )
    confidence: str = Field(
        "medium",
        description="How sure you are: 'low', 'medium', or 'high'.",
    )


class ConcludeTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "conclude"
    description: str = (
        "Signal that you have nothing more to add and are leaving the "
        "debate with a final position. Use this once the discussion is "
        "converging and your point has been made — do not keep posting "
        "restatements. The moderator records your final_position."
    )
    args_schema: Type[BaseModel] = ConcludeInput

    def _run(self, final_position: str, confidence: str = "medium") -> str:
        return json.dumps({
            "kind": "conclude",
            "final_position": final_position,
            "confidence": confidence,
        })


__all__ = ["ChannelPostTool", "ConcludeTool"]
