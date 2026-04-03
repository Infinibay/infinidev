"""Tool that lets the agent send a message to the user without ending the task."""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class SendMessageInput(BaseModel):
    message: str = Field(
        ..., description="The message to display to the user in the chat."
    )


