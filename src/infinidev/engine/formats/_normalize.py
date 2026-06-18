"""Shared JSON + tool-call normalization helpers (dependency-free leaf).

Every format adapter parses model output into raw dicts and then funnels
them through these helpers to produce the uniform ``{"name", "arguments"}``
shape. Keeping the helpers here (rather than copy-pasted into each adapter)
guarantees a single source of truth — most importantly for the mis-nested
sibling-param rescue in ``_normalize_single_call``.

This module imports ONLY ``json``, ``re``, and ``json_repair`` so it stays a
leaf with no dependency on any other format module (no circular imports).
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


_WRAPPER_KEYS = frozenset({
    "name", "arguments", "parameters", "function", "id", "type", "index",
    "tool_call_id", "role", "content",
})


def _normalize_single_call(obj: dict) -> dict[str, Any] | None:
    """Normalize a single tool call dict.

    Handles variants:
    - {"name": "x", "arguments": {...}}
    - {"function": {"name": "x", "arguments": {...}}}
    - {"function": "x", "arguments": {...}}
    - {"name": "x", "parameters": {...}}

    Mis-nesting rescue (Bug #12): small models sometimes emit a parameter
    as a SIBLING of "arguments" instead of inside it, e.g.::

        {"name": "add_step",
         "arguments": {"title": "..."},
         "expected_output": "..."}     # ← should be inside arguments

    We rescue any unknown top-level key that is not part of the standard
    wrapper schema (``_WRAPPER_KEYS``) by folding it into ``arguments``,
    so the tool actually receives the parameter the model intended.
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

    # Rescue mis-nested params: any non-wrapper top-level key gets folded
    # into arguments unless arguments already has that key (don't clobber).
    if isinstance(arguments, dict):
        for k, v in obj.items():
            if k in _WRAPPER_KEYS:
                continue
            if k not in arguments:
                arguments[k] = v

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
