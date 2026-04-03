"""Backward-compatibility re-export — actual code in engine.loop.engine.

All imports of the form ``from infinidev.engine.loop_engine import X``
continue to work via this stub.
"""

from infinidev.engine.loop.engine import LoopEngine  # noqa: F401
from infinidev.engine.loop.execution_context import ExecutionContext  # noqa: F401
from infinidev.engine.loop.llm_caller import (  # noqa: F401
    LLMCaller, LLMCallResult, ClassifiedCalls,
)
from infinidev.engine.loop.tool_processor import ToolProcessor  # noqa: F401
from infinidev.engine.loop.loop_guard import LoopGuard  # noqa: F401
from infinidev.engine.loop.step_manager import StepManager  # noqa: F401
from infinidev.engine.loop.step_summarizer import (  # noqa: F401
    _summarize_step, _synthesize_final,
)
from infinidev.engine.loop.model_context import _get_model_max_context  # noqa: F401
