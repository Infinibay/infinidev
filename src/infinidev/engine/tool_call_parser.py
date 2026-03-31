"""Tool call parsing for multiple LLM output formats.

Supports 9+ formats:
1. Manual-mode JSON: {"tool_calls": [{"name": ..., "arguments": ...}]}
2. Qwen/GLM: <tool_call>{...}</tool_call>
3. Qwen pipe-delimited: <|tool_call|>...<|/tool_call|>
4. Mistral: [TOOL_CALLS] [...]
5. Llama: <|python_tag|> function calls
6. <function_call>/<functioncall> wrappers
7. Markdown code blocks with JSON
8. Bare JSON objects
9. SEARCH/REPLACE blocks (Aider-style diffs)

Also provides:
- safe_json_loads() with automatic JSON repair
- ManualToolCall wrapper for uniform tool call handling
- parse_step_complete_args() for step_complete tool results
"""

from __future__ import annotations

import json
import re
from typing import Any

try:
    from json_repair import repair_json as _repair_json
except ImportError:
    _repair_json = None


# ── JSON utilities ────────────────────────────────────────────────────────

def safe_json_loads(text: str) -> Any:
    """Parse JSON with automatic repair for malformed model output.

    Tries standard json.loads first, then falls back to json_repair
    if available. This handles common LLM JSON issues like trailing
    commas, unquoted keys, truncated strings, etc.
    """
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


# ── ManualToolCall ────────────────────────────────────────────────────────

class ManualToolCall:
    """Lightweight stand-in for native tool call objects in manual TC mode.

    Mirrors the attribute structure of litellm/OpenAI tool call objects
    so the rest of the pipeline (dispatch, logging) works unchanged.
    """

    __slots__ = ("id", "function")

    class _Function:
        __slots__ = ("name", "arguments")

        def __init__(self, name: str, arguments: str) -> None:
            self.name = name
            self.arguments = arguments

    def __init__(self, id: str, name: str, arguments: str) -> None:
        self.id = id
        self.function = self._Function(name, arguments)


# ── Main parser ───────────────────────────────────────────────────────────

def parse_text_tool_calls(content: str) -> list[dict[str, Any]] | None:
    """Parse tool calls from model text when native FC is unavailable.

    Returns a list of dicts with "name" and "arguments" keys,
    or None if no valid tool calls found.
    """
    if not content or not content.strip():
        return None

    # Strip thinking sections (various model formats)
    cleaned = re.sub(
        r"<(?:thinking|think|thoughts|\|thinking\|)>.*?</(?:thinking|think|thoughts|\|thinking\|)>",
        "",
        content,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Strip hallucinated tool outputs — small models often generate fake results
    cleaned = re.sub(
        r"<(?:tool[_-]?output|tool[_-]?result|tool[_-]?response|observation)>.*?</(?:tool[_-]?output|tool[_-]?result|tool[_-]?response|observation)>",
        "",
        cleaned,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # ── 1. Native model tool-call tokens ─────────────────────────────
    # Try these first — they're unambiguous signals of tool use intent.

    # Qwen / GLM: <tool_call>{...}</tool_call>  (one or more)
    tc_tag_matches = re.findall(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        cleaned, re.DOTALL,
    )
    if tc_tag_matches:
        calls = _extract_calls_from_fragments(tc_tag_matches)
        if calls:
            return calls

    # Qwen pipe-delimited: <|tool_call|>{...}<|/tool_call|>
    tc_pipe_matches = re.findall(
        r"<\|tool_call\|>\s*(.*?)\s*<\|/tool_call\|>",
        cleaned, re.DOTALL,
    )
    if tc_pipe_matches:
        calls = _extract_calls_from_fragments(tc_pipe_matches)
        if calls:
            return calls

    # Mistral: [TOOL_CALLS] [{...}, ...]
    mistral_match = re.search(
        r"\[TOOL_CALLS\]\s*(\[.*?\])",
        cleaned, re.DOTALL,
    )
    if mistral_match:
        calls = _extract_calls_from_array(mistral_match.group(1))
        if calls:
            return calls

    # Llama: <|python_tag|> followed by JSON (function call format)
    python_tag_match = re.search(
        r"<\|python_tag\|>\s*(.*)",
        cleaned, re.DOTALL,
    )
    if python_tag_match:
        calls = _extract_calls_from_fragments([python_tag_match.group(1)])
        if calls:
            return calls

    # Generic: <function_call>{...}</function_call> or <functioncall>{...}</functioncall>
    fc_matches = re.findall(
        r"<function_?call>\s*(.*?)\s*</function_?call>",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if fc_matches:
        calls = _extract_calls_from_fragments(fc_matches)
        if calls:
            return calls

    # <tool>{...}</tool> or <tools>{...}</tools> (fine-tuned / small models)
    tool_tag_matches = re.findall(
        r"<tools?>\s*(.*?)\s*</tools?>",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if tool_tag_matches:
        calls = _extract_calls_from_fragments(tool_tag_matches)
        if calls:
            return calls

    # <function=tool_name>{"arg": "val"}</function> (attribute-style)
    attr_matches = re.findall(
        r"<function=([a-z_]\w*)>\s*(.*?)\s*</function>",
        cleaned, re.DOTALL | re.IGNORECASE,
    )
    if attr_matches:
        calls = []
        for name, args_str in attr_matches:
            try:
                args = safe_json_loads(args_str)
                calls.append({"name": name, "arguments": args if isinstance(args, dict) else {}})
            except (json.JSONDecodeError, TypeError):
                calls.append({"name": name, "arguments": {}})
        if calls:
            return calls

    # ── 2. Our manual-mode JSON: {"tool_calls": [...]} ───────────────
    # Check markdown code blocks first, then bare text.
    json_candidates: list[str] = []

    # Match ```json ... ``` or ``` ... ```
    code_blocks = re.findall(r"```(?:json)?\s*\n?(.*?)\n?```", cleaned, re.DOTALL)
    json_candidates.extend(code_blocks)

    # Also try the raw cleaned text (model might output bare JSON)
    json_candidates.append(cleaned.strip())

    for candidate in json_candidates:
        candidate = candidate.strip()
        if not candidate:
            continue

        # Try to find a JSON object in the candidate
        brace_start = candidate.find("{")
        if brace_start == -1:
            continue

        # Find the matching closing brace
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
                            # {"tool_calls": [...]} wrapper
                            if "tool_calls" in parsed:
                                calls = _normalize_call_list(parsed["tool_calls"])
                                if calls:
                                    return calls
                            # Bare tool call object: {"name": "...", "arguments": {...}}
                            if "name" in parsed:
                                calls = _normalize_call_list([parsed])
                                if calls:
                                    return calls
                    except (json.JSONDecodeError, TypeError):
                        pass
                    continue  # try next brace pair in the candidate

    # ── 9. SEARCH/REPLACE blocks (Aider-style diffs) ────────────────
    sr_calls = _parse_search_replace_blocks(cleaned)
    if sr_calls:
        return sr_calls

    return None


def _parse_search_replace_blocks(text: str) -> list[dict[str, Any]] | None:
    """Parse SEARCH/REPLACE blocks into edit_file tool calls.

    Supports formats:
    - ``<<<<<<< SEARCH`` ... ``=======`` ... ``>>>>>>> REPLACE``
    - ``<<<<<<< SEARCH@path`` or ``<<<<<<< SEARCH path``
    - File path on the line before the block
    """
    pattern = re.compile(
        r"(?:^([^\n<>]+\.[\w]+)\n)?"            # optional file path on preceding line
        r"<{4,}\s*SEARCH"                         # <<<<<<< SEARCH
        r"(?:[@\s]+([^\n]*\.[\w]+))?"             # optional @path or path after SEARCH
        r"(?:[@\s]*(\d+)(?:-\d+)?)?\s*\n"        # optional @linenum or @start-end
        r"(.*?)\n"                                # old code (captured)
        r"={4,}\s*\n"                             # =======
        r"(.*?)\n"                                # new code (captured)
        r">{4,}\s*REPLACE",                       # >>>>>>> REPLACE
        re.DOTALL | re.MULTILINE,
    )

    calls: list[dict[str, Any]] = []
    for m in pattern.finditer(text):
        path = m.group(1) or m.group(2) or ""
        old_string = m.group(4)
        new_string = m.group(5)

        if old_string is not None and new_string is not None:
            args: dict[str, Any] = {
                "old_string": old_string,
                "new_string": new_string,
            }
            if path:
                args["path"] = path.strip()
            calls.append({"name": "edit_file", "arguments": args})

    return calls if calls else None


# ── Fragment / array parsers ──────────────────────────────────────────────

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
            # Try to extract first JSON object from fragment
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
    - {"function": "x", "arguments": {...}}  (Llama-style)
    - {"name": "x", "parameters": {...}}
    """
    name = obj.get("name")
    arguments = obj.get("arguments") or obj.get("parameters") or {}

    # Nested "function" key (OpenAI-style wrapper)
    if not name and "function" in obj:
        func = obj["function"]
        if isinstance(func, dict):
            name = func.get("name")
            arguments = func.get("arguments") or func.get("parameters") or {}
        elif isinstance(func, str):
            # {"function": "read_file", "arguments": {...}}
            name = func

    if not name or not isinstance(name, str):
        return None

    return {"name": name, "arguments": arguments}


# ── Step complete parser ──────────────────────────────────────────────────

def parse_step_complete_args(arguments: str | dict[str, Any]) -> "StepResult":
    """Parse step_complete tool call arguments into a StepResult."""
    from infinidev.engine.loop_models import StepOperation, StepResult

    if isinstance(arguments, str):
        try:
            args = safe_json_loads(arguments) if arguments.strip() else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
    else:
        args = arguments or {}

    # Parse next_steps into StepOperation objects
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

    # Coerce final_answer to string (model may pass dict/list instead of string)
    raw_answer = args.get("final_answer")
    if raw_answer is not None and not isinstance(raw_answer, str):
        raw_answer = json.dumps(raw_answer)

    return StepResult(
        summary=args.get("summary", "Step completed (no summary provided)"),
        status=args.get("status", "continue"),
        next_steps=next_steps,
        final_answer=raw_answer,
    )
