"""RespondTool — terminator for the chat agent's conversational replies."""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class RespondInput(BaseModel):
    message: str = Field(
        ...,
        description=(
            "The reply to show the user. 1-3 sentences, natural chat "
            "register. Match the user's language (Spanish or English). "
            "Do not prefix with 'Claro,' or 'Sure,' — just answer."
        ),
    )
    language: str | None = Field(
        None,
        description=(
            "Optional language hint ('es' or 'en'). The chat agent "
            "usually infers this from the user's message; set explicitly "
            "only if the conversation mixes languages."
        ),
    )


class RespondTool(InfinibayBaseTool):
    is_read_only: bool = True
    name: str = "respond"
    description: str = (
        "End the turn with a conversational reply to the user. Use this "
        "when the user's message can be answered without triggering "
        "analysis or code editing — greetings, project questions that "
        "only need a couple of file reads, opinion questions you can "
        "answer now. After this call the turn is over; the pipeline "
        "will NOT run the analyst or developer. If execution is "
        "clearly wanted (the user said 'do it', 'implement that', "
        "approved your proposal), call `escalate` instead."
    )
    args_schema: Type[BaseModel] = RespondInput

    def _run(self, message: str, language: str | None = None) -> str:
        # The chat_agent orchestrator reads the tool_call args directly
        # from the LLM response and treats `respond` as a terminator —
        # this _run is only reached if the tool is dispatched through
        # the normal path (which should not happen). Returning a
        # structured ack keeps the contract clean.
        return json.dumps({"kind": "respond", "message": message, "language": language})
