"""Stochastic primitives — reusable building blocks for behavior checkers.

These helpers let a :class:`StochasticChecker` compose its judgement
from cheap deterministic/heuristic operations (regex, fuzzy string,
embedding similarity, tool-call inspection, result error detection,
diff inspection) instead of delegating to an LLM.

Every primitive that signals a behavior returns a
:class:`Confidence` so checkers can combine multiple sources and map
the result to an integer delta via :func:`confidence_to_delta`.
"""

from infinidev.engine.behavior.primitives.scoring import (
    Confidence,
    combine,
    confidence_to_delta,
)
from infinidev.engine.behavior.primitives.text import (
    fuzzy_ratio,
    keyword_presence,
    regex_scan,
)
from infinidev.engine.behavior.primitives.embeddings import (
    cosine_sim,
    embed,
    max_cosine_sim,
)
from infinidev.engine.behavior.primitives.tool_inspect import (
    NormalizedCall,
    detect_shell_hack,
    filter_by_name,
    iterate_messages_tool_calls,
    normalize_tool_calls,
    step_complete_status,
)
from infinidev.engine.behavior.primitives.result_inspect import (
    detect_error,
    immediately_preceding_tool_error,
    iterate_tool_results,
    last_tool_error,
    was_acknowledged,
)
from infinidev.engine.behavior.primitives.diff_inspect import (
    FileOp,
    biggest_op,
    count_line_delta,
    parse_file_ops,
    scan_for_todo_markers,
)

__all__ = [
    "Confidence",
    "combine",
    "confidence_to_delta",
    "fuzzy_ratio",
    "keyword_presence",
    "regex_scan",
    "cosine_sim",
    "embed",
    "max_cosine_sim",
    "NormalizedCall",
    "detect_shell_hack",
    "filter_by_name",
    "iterate_messages_tool_calls",
    "normalize_tool_calls",
    "step_complete_status",
    "detect_error",
    "immediately_preceding_tool_error",
    "iterate_tool_results",
    "last_tool_error",
    "was_acknowledged",
    "FileOp",
    "biggest_op",
    "count_line_delta",
    "parse_file_ops",
    "scan_for_todo_markers",
]
