"""Tool that lets the agent send a message to the user without ending the task."""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class SendMessageInput(BaseModel):
    message: str = Field(
        ..., description="The message to display to the user in the chat."
    )


class SendMessageTool(InfinibayBaseTool):
    name: str = "send_message"
    description: str = (
        "Send a message to the user without ending the current task. "
        "Use this to share progress updates, intermediate results, ask "
        "clarifying questions, or show findings while you continue working. "
        "The task loop keeps running after this call."
    )
    args_schema: Type[BaseModel] = SendMessageInput

    def _run(self, message: str) -> str:
        # The actual delivery is handled by the loop engine which intercepts
        # this tool call and emits a loop_user_message event.  The tool
        # itself just returns an acknowledgment so the LLM knows it worked.
        return json.dumps({"status": "delivered"})
