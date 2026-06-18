"""Modular tool call parsing for multiple LLM output formats.

Each format is a ToolCallFormat subclass with a `detect()` and `parse()` method.
Formats are tried in priority order (lowest number first). To add a new format,
subclass ToolCallFormat and register it with @register_format.

Built-in formats:
- Gemma4Format:        <|tool_call>call:name{args}<tool_call|>
- QwenFormat:          <tool_call>{...}</tool_call>
- QwenPipeFormat:      <|tool_call|>{...}<|/tool_call|>
- MistralFormat:       [TOOL_CALLS] [...]
- LlamaFormat:         <|python_tag|> {...}
- FunctionCallFormat:  <function_call>{...}</function_call>
- ToolTagFormat:       <tool>{...}</tool>
- AttrFunctionFormat:  <function=name>{...}</function>
- ManualJsonFormat:    {"tool_calls": [...]}  / bare JSON / markdown blocks
- SearchReplaceFormat: <<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE

Also provides:
- safe_json_loads() with automatic JSON repair
- ManualToolCall wrapper for uniform tool call handling
- parse_step_complete_args() for step_complete tool results
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

# Shared, dependency-free normalization helpers. ``safe_json_loads`` is
# re-exported here because other modules import it from this path
# (e.g. loop/step_summarizer, review_engine, plan_validator).
from infinidev.engine.formats._normalize import (  # noqa: F401
    _WRAPPER_KEYS,
    _extract_calls_from_array,
    _extract_calls_from_fragments,
    _fix_jslike_json,
    _normalize_call_list,
    _normalize_single_call,
    safe_json_loads,
)

logger = logging.getLogger(__name__)


# ── ManualToolCall ────────────────────────────────────────────────────────

from infinidev.engine.formats.manual_tool_call import ManualToolCall
from infinidev.engine.formats.tool_call_format import (
    ToolCallFormat, register_format, get_registered_formats, _FORMATS,
)
# Import all format classes to trigger @register_format registration
from infinidev.engine.formats.gemma4_format import Gemma4Format  # noqa: F401
from infinidev.engine.formats.hermes_xml_format import HermesXmlFormat  # noqa: F401
from infinidev.engine.formats.qwen_format import QwenFormat  # noqa: F401
from infinidev.engine.formats.qwen_pipe_format import QwenPipeFormat  # noqa: F401
from infinidev.engine.formats.mistral_format import MistralFormat  # noqa: F401
from infinidev.engine.formats.llama_format import LlamaFormat  # noqa: F401
from infinidev.engine.formats.function_call_format import FunctionCallFormat  # noqa: F401
from infinidev.engine.formats.tool_tag_format import ToolTagFormat  # noqa: F401
from infinidev.engine.formats.attr_function_format import AttrFunctionFormat  # noqa: F401
from infinidev.engine.formats.manual_json_format import ManualJsonFormat  # noqa: F401
from infinidev.engine.formats.search_replace_format import SearchReplaceFormat  # noqa: F401


# ── Preprocessing ────────────────────────────────────────────────────────

# Thinking section tags to strip before parsing
_THINKING_RE = re.compile(
    r"<(?:thinking|think|thoughts|\|thinking\|)>.*?</(?:thinking|think|thoughts|\|thinking\|)>",
    re.DOTALL | re.IGNORECASE,
)

# Hallucinated tool output tags to strip
_HALLUCINATED_OUTPUT_RE = re.compile(
    r"<(?:tool[_-]?output|tool[_-]?result|tool[_-]?response|observation)>"
    r".*?"
    r"</(?:tool[_-]?output|tool[_-]?result|tool[_-]?response|observation)>",
    re.DOTALL | re.IGNORECASE,
)

# Gemma 4 noise tokens
_GEMMA4_NOISE_RE = re.compile(
    r"<\|?tool_response\|?>|<\|?channel\|?>|<\|?channel>|<channel\|>|<eos>"
)


def _preprocess(text: str) -> str:
    """Strip thinking sections, hallucinated outputs, and noise tokens."""
    text = _THINKING_RE.sub("", text)
    text = _HALLUCINATED_OUTPUT_RE.sub("", text)
    text = _GEMMA4_NOISE_RE.sub("", text)
    return text


# ══════════════════════════════════════════════════════════════════════════
# Built-in formats (ordered by priority)
# ══════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════════

def parse_text_tool_calls(content: str) -> list[dict[str, Any]] | None:
    """Parse tool calls from model text when native FC is unavailable.

    Tries each registered format in priority order. Returns a list of
    dicts with "name" and "arguments" keys, or None if nothing found.
    """
    if not content or not content.strip():
        return None

    cleaned = _preprocess(content)

    for fmt_cls in _FORMATS:
        fmt = fmt_cls()
        if fmt.detect(cleaned):
            result = fmt.parse(cleaned)
            if result:
                return result
            # detect() matched but parse() yielded nothing — a likely silent
            # tool-call loss on a weak model. Log once so it is debuggable.
            logger.debug(
                "format %r detected but parsed no tool calls from: %.200s",
                fmt.name, cleaned,
            )

    return None


def get_registered_formats() -> list[dict[str, Any]]:
    """Return info about all registered formats (for debugging/help)."""
    return [
        {"name": cls.name, "priority": cls.priority, "doc": cls.__doc__ or ""}
        for cls in _FORMATS
    ]


# ── Step complete parser ──────────────────────────────────────────────────

def parse_step_complete_args(arguments: str | dict[str, Any]) -> "StepResult":
    """Parse step_complete tool call arguments into a StepResult."""
    from infinidev.engine.loop.models import StepOperation, StepResult

    if isinstance(arguments, str):
        try:
            args = safe_json_loads(arguments) if arguments.strip() else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = arguments or {}

    if not isinstance(args, dict):
        args = {}

    # next_steps is ignored — plan management is now done via add_step/modify_step/remove_step.
    # We still parse it for backward compat so existing LLM behavior doesn't crash.

    raw_answer = args.get("final_answer")
    if raw_answer is not None and not isinstance(raw_answer, str):
        raw_answer = json.dumps(raw_answer)

    return StepResult(
        summary=args.get("summary", "Step completed (no summary provided)"),
        status=args.get("status", "continue"),
        next_steps=[],
        final_answer=raw_answer,
    )
