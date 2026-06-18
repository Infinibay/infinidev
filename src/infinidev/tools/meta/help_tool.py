"""Meta-tool that provides detailed help and examples for all tools."""

from typing import Type

from pydantic import BaseModel

from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.meta.help_input import HelpInput
from infinidev.tools.meta.help_content import HELP_CONTENT, _CATEGORY_INDEX


class HelpTool(InfinibayBaseTool):
    name: str = "help"
    description: str = "Get detailed help and examples for any tool."
    args_schema: Type[BaseModel] = HelpInput

    def _run(self, context: str | None = None) -> str:
        if context is not None:
            context = context.strip().lower()

        # Direct match
        if context in HELP_CONTENT:
            return HELP_CONTENT[context]

        # Try matching as category index key
        if context in _CATEGORY_INDEX:
            return HELP_CONTENT.get(context, f"No help available for category: {context}")

        # Fuzzy: search for context as substring in keys
        matches = [k for k in HELP_CONTENT if k and context and context in k]
        if len(matches) == 1:
            return HELP_CONTENT[matches[0]]
        if len(matches) > 1:
            return f"Multiple matches for '{context}': {', '.join(matches)}. Be more specific."

        available = sorted(k for k in HELP_CONTENT if k is not None)
        return (
            f"No help found for '{context}'.\n"
            f"Available topics: {', '.join(available)}"
        )
