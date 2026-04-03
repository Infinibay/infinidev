"""Meta-tool that provides detailed help and examples for all tools."""

from typing import Type

from pydantic import BaseModel, Field

from infinidev.tools.base.base_tool import InfinibayBaseTool


class HelpInput(BaseModel):
    context: str | None = Field(
        default=None,
        description="Tool name or category to get help for. Omit for overview.",
    )


