"""Prompt-building sub-package.

Pieces of the system/iteration prompt that are self-contained enough
to live outside ``loop/context.py`` and be tested in isolation.
"""

from infinidev.engine.loop.prompt.tools_section import (
    build_tools_prompt_section,
)

__all__ = ["build_tools_prompt_section"]
