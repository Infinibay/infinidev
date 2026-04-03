"""Tool that lets the agent send a message to the user without ending the task."""

import json
from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


from infinidev.tools.chat.send_message_input import SendMessageInput
from infinidev.tools.chat.send_message_tool import SendMessageTool
