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

class ManualToolCall:
    """Lightweight stand-in for native tool call objects in manual TC mode."""

    __slots__ = ("id", "function")

    class _Function:
        __slots__ = ("name", "arguments")

        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = self._Function(name, arguments)


# ══════════════════════════════════════════════════════════════════════════
# Format registry
# ══════════════════════════════════════════════════════════════════════════

_FORMATS: list[type["ToolCallFormat"]] = []


def register_format(cls: type["ToolCallFormat"]) -> type["ToolCallFormat"]:
    """Class decorator to register a tool call format."""
    _FORMATS.append(cls)
    _FORMATS.sort(key=lambda c: c.priority)
    return cls


class ToolCallFormat(ABC):
    """Base class for tool call format parsers.

    Subclass and decorate with @register_format to add a new format.

    Attributes:
        name:     Human-readable format name (for logging).
        priority: Lower = tried first. Native token formats should be 10-50,
                  generic JSON formats 100+, fallbacks 200+.
    """

    name: str = "unknown"
    priority: int = 100

    @abstractmethod
    def detect(self, text: str) -> bool:
        """Return True if this format's markers are present in the text."""
        ...

    @abstractmethod
    def parse(self, text: str) -> list[dict[str, Any]] | None:
        """Extract tool calls from text. Return None if parsing fails."""
        ...


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

@register_format
class Gemma4Format(ToolCallFormat):
    """Gemma 4: <|tool_call>call:func_name{args}<tool_call|>"""

    name = "gemma4"
    priority = 10

    _PREFIX = re.compile(r"<\|tool_call>call:([a-z_]\w*)")

    def detect(self, text: str) -> bool:
        return "<|tool_call>call:" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []

        for m in self._PREFIX.finditer(text):
            name = m.group(1)
            rest = text[m.end():]

            brace_start = rest.find("{")
            if brace_start == -1:
                calls.append({"name": name, "arguments": {}})
                continue

            # Brace-depth match (handles nested objects/arrays)
            depth = 0
            end_idx = -1
            in_sq = in_dq = False
            for i, ch in enumerate(rest[brace_start:]):
                if in_sq:
                    if ch == "'" and (i == 0 or rest[brace_start + i - 1] != "\\"):
                        in_sq = False
                    continue
                if in_dq:
                    if ch == '"' and (i == 0 or rest[brace_start + i - 1] != "\\"):
                        in_dq = False
                    continue
                if ch == "'":
                    in_sq = True
                elif ch == '"':
                    in_dq = True
                elif ch in "{[":
                    depth += 1
                elif ch in "}]":
                    depth -= 1
                    if depth == 0:
                        end_idx = brace_start + i
                        break

            if end_idx == -1:
                args_str = rest[brace_start:]
                cut = args_str.find("<tool_call|>")
                if cut != -1:
                    args_str = args_str[:cut]
            else:
                args_str = rest[brace_start : end_idx + 1]

            calls.append({"name": name, "arguments": _fix_jslike_json(args_str)})

        return calls if calls else None


@register_format
class QwenFormat(ToolCallFormat):
    """Qwen / GLM: <tool_call>{...}</tool_call>"""

    name = "qwen"
    priority = 20

    _RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)

    def detect(self, text: str) -> bool:
        return "<tool_call>" in text and "</tool_call>" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None


@register_format
class QwenPipeFormat(ToolCallFormat):
    """Qwen pipe-delimited: <|tool_call|>{...}<|/tool_call|>"""

    name = "qwen_pipe"
    priority = 21

    _RE = re.compile(r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>", re.DOTALL)

    def detect(self, text: str) -> bool:
        return "<|tool_call|>" in text and "<|/tool_call|>" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None


@register_format
class MistralFormat(ToolCallFormat):
    """Mistral: [TOOL_CALLS] [{...}, ...]"""

    name = "mistral"
    priority = 30

    _RE = re.compile(r"\[TOOL_CALLS\]\s*(\[.*?\])", re.DOTALL)

    def detect(self, text: str) -> bool:
        return "[TOOL_CALLS]" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        m = self._RE.search(text)
        return _extract_calls_from_array(m.group(1)) if m else None


@register_format
class LlamaFormat(ToolCallFormat):
    """Llama: <|python_tag|> followed by JSON"""

    name = "llama"
    priority = 40

    _RE = re.compile(r"<\|python_tag\|>\s*(.*)", re.DOTALL)

    def detect(self, text: str) -> bool:
        return "<|python_tag|>" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        m = self._RE.search(text)
        return _extract_calls_from_fragments([m.group(1)]) if m else None


@register_format
class FunctionCallFormat(ToolCallFormat):
    """Generic: <function_call>{...}</function_call>"""

    name = "function_call"
    priority = 50

    _RE = re.compile(r"<function_?call>\s*(.*?)\s*</function_?call>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        low = text.lower()
        return "<functioncall>" in low or "<function_call>" in low

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None


@register_format
class ToolTagFormat(ToolCallFormat):
    """Fine-tuned / small models: <tool>{...}</tool> or <tools>{...}</tools>"""

    name = "tool_tag"
    priority = 51

    _RE = re.compile(r"<tools?>\s*(.*?)\s*</tools?>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        low = text.lower()
        return ("<tool>" in low or "<tools>") and ("</tool>" in low or "</tools>" in low)

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        return _extract_calls_from_fragments(matches) if matches else None


@register_format
class AttrFunctionFormat(ToolCallFormat):
    """Attribute-style: <function=tool_name>{"arg": "val"}</function>"""

    name = "attr_function"
    priority = 52

    _RE = re.compile(r"<function=([a-z_]\w*)>\s*(.*?)\s*</function>", re.DOTALL | re.IGNORECASE)

    def detect(self, text: str) -> bool:
        return "<function=" in text.lower()

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        matches = self._RE.findall(text)
        if not matches:
            return None
        calls = []
        for name, args_str in matches:
            try:
                args = safe_json_loads(args_str)
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            except (json.JSONDecodeError, TypeError):
                calls.append({"name": name, "arguments": {}})
        return calls if calls else None


@register_format
class ManualJsonFormat(ToolCallFormat):
    """Manual-mode JSON: {"tool_calls": [...]}, bare JSON, or markdown blocks."""

    name = "manual_json"
    priority = 100

    def detect(self, text: str) -> bool:
        return "{" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        candidates: list[str] = []

        # Markdown code blocks
        code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
        candidates.extend(code_blocks)
        candidates.append(text.strip())

        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate:
                continue

            brace_start = candidate.find("{")
            if brace_start == -1:
                continue

            depth = 0
            for i, ch in enumerate(candidate[brace_start:], start=brace_start):
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = candidate[brace_start : i + 1]
                        try:
                            parsed = safe_json_loads(json_str)
                            if isinstance(parsed, dict):
                                if "tool_calls" in parsed:
                                    calls = _normalize_call_list(parsed["tool_calls"])
                                    if calls:
                                        return calls
                                if "name" in parsed:
                                    calls = _normalize_call_list([parsed])
                                    if calls:
                                        return calls
                        except (json.JSONDecodeError, TypeError):
                            pass
                        continue

        return None


@register_format
class SearchReplaceFormat(ToolCallFormat):
    """Aider-style SEARCH/REPLACE blocks → edit_file tool calls."""

    name = "search_replace"
    priority = 200

    _RE = re.compile(
        r"(?:^([^\n<>]+\.[\w]+)\n)?"
        r"<{4,}\s*SEARCH"
        r"(?:[@\s]+([^\n]*\.[\w]+))?"
        r"(?:[@\s]*(\d+)(?:-\d+)?)?\s*\n"
        r"(.*?)\n"
        r"={4,}\s*\n"
        r"(.*?)\n"
        r">{4,}\s*REPLACE",
        re.DOTALL | re.MULTILINE,
    )

    def detect(self, text: str) -> bool:
        return "SEARCH" in text and "REPLACE" in text and "=====" in text

    def parse(self, text: str) -> list[dict[str, Any]] | None:
        calls: list[dict[str, Any]] = []
        for m in self._RE.finditer(text):
            path = m.group(1) or m.group(2) or ""
            old_string = m.group(4)
            new_string = m.group(5)
            if old_string is not None and new_string is not None:
                args: dict[str, Any] = {"old_string": old_string, "new_string": new_string}
                if path:
                    args["path"] = path.strip()
                calls.append({"name": "edit_file", "arguments": args})
        return calls if calls else None


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
