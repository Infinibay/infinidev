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
import re
from abc import ABC, abstractmethod
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None


# ── JSON utilities ────────────────────────────────────────────────────────

def safe_json_loads(text: str) -> Any:
    """Parse JSON with automatic repair for malformed model output."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        if _repair_json is not None:
            try:
                repaired = _repair_json(text, return_objects=True)
                if repaired is not None:
                    return repaired
            except Exception:
                pass
        raise


def _fix_jslike_json(args_str: str) -> dict[str, Any]:
    """Convert JS-like object syntax to a Python dict.

    Many models emit tool args with unquoted keys and/or single-quoted
    values. This normalizes to valid JSON.
    """
    args_str = args_str.strip()
    try:
        result = safe_json_loads(args_str)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    fixed = re.sub(
        r"'((?:[^'\\]|\\.)*)'",
        lambda m: '"' + m.group(1).replace('"', '\\"').replace("\\'", "'") + '"',
        args_str,
    )
    fixed = re.sub(r'(?<=[{\[,])\s*([a-zA-Z_]\w*)\s*:', r' "\1":', fixed)
    fixed = re.sub(r'^\{\s*([a-zA-Z_]\w*)\s*:', r'{ "\1":', fixed)

    try:
        result = safe_json_loads(fixed)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, TypeError):
        pass

    return {}


# ── Normalization helpers ────────────────────────────────────────────────

def _normalize_call_list(raw: list) -> list[dict[str, Any]] | None:
    """Normalize a list of raw tool call dicts into [{name, arguments}, ...]."""
    calls: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            call = _normalize_single_call(item)
            if call:
                calls.append(call)
    return calls if calls else None


def _normalize_single_call(obj: dict) -> dict[str, Any] | None:
    """Normalize a single tool call dict.

    Handles variants:
    - {"name": "x", "arguments": {...}}
    - {"function": {"name": "x", "arguments": {...}}}
    - {"function": "x", "arguments": {...}}
    - {"name": "x", "parameters": {...}}
    """
    name = obj.get("name")
    arguments = obj.get("arguments") or obj.get("parameters") or {}

    if not name and "function" in obj:
        func = obj["function"]
        if isinstance(func, dict):
            name = func.get("name")
            arguments = func.get("arguments") or func.get("parameters") or {}
        elif isinstance(func, str):
            name = func

    if not name or not isinstance(name, str):
        return None

    return {"name": name, "arguments": arguments}


def _extract_calls_from_fragments(fragments: list[str]) -> list[dict[str, Any]] | None:
    """Parse JSON tool call objects from text fragments."""
    calls: list[dict[str, Any]] = []
    for frag in fragments:
        frag = frag.strip()
        if not frag:
            continue
        parsed = None
        try:
            parsed = safe_json_loads(frag)
        except (json.JSONDecodeError, TypeError):
            brace = frag.find("{")
            if brace == -1:
                continue
            depth = 0
            for i, ch in enumerate(frag[brace:], start=brace):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            parsed = safe_json_loads(frag[brace : i + 1])
                        except (json.JSONDecodeError, TypeError):
                            pass
                        break
            if parsed is None:
                continue

        if not isinstance(parsed, dict):
            continue

        call = _normalize_single_call(parsed)
        if call:
            calls.append(call)

    return calls if calls else None


def _extract_calls_from_array(text: str) -> list[dict[str, Any]] | None:
    """Parse a JSON array of tool call objects."""
    try:
        arr = safe_json_loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(arr, list):
        return None
    return _normalize_call_list(arr)


# ── ManualToolCall ────────────────────────────────────────────────────────

from infinidev.engine.formats.manual_tool_call import ManualToolCall
from infinidev.engine.formats.tool_call_format import (
    ToolCallFormat, register_format, get_registered_formats, _FORMATS,
)
# Import all format classes to trigger @register_format registration
from infinidev.engine.formats.gemma4_format import Gemma4Format  # noqa: F401
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

    raw_next_steps = args.get("next_steps", [])
    next_steps: list[StepOperation] = []
    if isinstance(raw_next_steps, list):
        for item in raw_next_steps:
            if isinstance(item, dict) and "op" in item and "index" in item:
                try:
                    next_steps.append(StepOperation(
                        op=item["op"],
                        index=item["index"],
                        description=item.get("description", ""),
                    ))
                except Exception:
                    pass

    raw_answer = args.get("final_answer")
    if raw_answer is not None and not isinstance(raw_answer, str):
        raw_answer = json.dumps(raw_answer)

    return StepResult(
        summary=args.get("summary", "Step completed (no summary provided)"),
        status=args.get("status", "continue"),
        next_steps=next_steps,
        final_answer=raw_answer,
    )
