"""Tools exclusive to the chat agent tier.

These are pseudo-tools: they are schema-level terminators rather than
regular dispatched tools. The chat_agent orchestrator parses the LLM's
tool_call args directly from the LiteLLM response (mirroring how
step_complete is handled) and constructs a ChatAgentResult from them.
_run returns a JSON acknowledgement so that if the loop ever does
dispatch them through the normal path, nothing blows up.
"""

from infinidev.tools.chat_agent.respond_tool import RespondTool
from infinidev.tools.chat_agent.escalate_tool import EscalateTool

__all__ = ["RespondTool", "EscalateTool"]
