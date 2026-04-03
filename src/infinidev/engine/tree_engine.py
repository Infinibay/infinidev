"""Backward-compatibility re-export — actual code in engine.tree.engine."""
from infinidev.engine.tree.engine import *  # noqa: F401,F403
from infinidev.engine.tree.engine import (  # noqa: F401  — explicit re-exports for mock.patch
    TreeEngine,
    _call_llm,
    _parse_text_tool_calls,
    _ManualToolCall,
)
# Re-export execute_tool_call so tests can patch it at this module path
from infinidev.engine.loop_tools import execute_tool_call  # noqa: F401
