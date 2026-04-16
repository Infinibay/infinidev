"""EscalateTool — terminator that hands the turn off to the planner."""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class EscalateInput(BaseModel):
    understanding: str = Field(
        ...,
        description=(
            "1-2 sentences in your own words describing what the user "
            "wants. The planner reads this to confirm the handoff isn't "
            "confused. Do NOT paraphrase back to the user — just state "
            "the intent as you understand it."
        ),
    )
    user_visible_preview: str = Field(
        "",
        description=(
            "Short message (1 sentence, user's language) shown via the "
            "UI immediately after escalation, before the planner runs. "
            "Example: 'Voy a implementar X — arranco con el análisis.' "
            "Prevents dead-air while the planner thinks. Empty string "
            "skips the preview."
        ),
    )
    opened_files: list[str] = Field(
        default_factory=list,
        description=(
            "List of file paths you read during this turn. The planner "
            "will NOT re-open these — the handoff's point is to avoid "
            "redundant I/O. Include every path you called read_file / "
            "get_symbol_code / code_search on."
        ),
    )
    user_signal: str = Field(
        "",
        description=(
            "The exact user text you interpreted as approval (e.g. "
            "'sí dale', 'implementá eso', 'fix it'). Kept for audit. "
            "Empty when the user's first message was itself a direct "
            "action request (no prior chat to approve from)."
        ),
    )
    suggested_flow: str = Field(
        "develop",
        description=(
            "Flow target for the planner. v1 only accepts 'develop'. "
            "Future versions may add 'sysadmin'. Default is 'develop'."
        ),
    )


class EscalateTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "escalate"
    description: str = (
        "Hand off this turn to the planner for real work. Use this when "
        "the user has clearly asked for execution (action verbs like "
        "'fix', 'implement', 'refactor', 'arreglá', 'implementá') or "
        "has approved a proposal you made in a previous turn. After "
        "this call the planner will write a detailed plan and the "
        "developer will execute it. Do NOT escalate on ambiguous "
        "acknowledgements ('ok', 'suena bien') — ask to confirm first. "
        "If the user just wants a conversational answer, call `respond` "
        "instead."
    )
    args_schema: Type[BaseModel] = EscalateInput

    def _run(
        self,
        understanding: str,
        user_visible_preview: str = "",
        opened_files: list[str] | None = None,
        user_signal: str = "",
        suggested_flow: str = "develop",
    ) -> str:
        # As with RespondTool, the orchestrator reads tool_call args
        # directly. This _run is the safe fallback if the tool ever
        # hits the normal dispatch path.
        return json.dumps({
            "kind": "escalate",
            "understanding": understanding,
            "user_visible_preview": user_visible_preview,
            "opened_files": opened_files or [],
            "user_signal": user_signal,
            "suggested_flow": suggested_flow,
        })
