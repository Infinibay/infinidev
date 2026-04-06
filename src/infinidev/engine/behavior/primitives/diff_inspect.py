"""File-edit inspection: parse create_file/replace_lines/symbol ops."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from infinidev.engine.behavior.primitives.scoring import Confidence
from infinidev.engine.behavior.primitives.tool_inspect import NormalizedCall


@dataclass
class FileOp:
    tool: str
    path: str
    lines_added: int
    lines_removed: int
    content: str   # the added/replacement content (for marker scanning)


_EDIT_TOOLS = {
    "create_file",
    "replace_lines",
    "edit_symbol",
    "add_symbol",
    "remove_symbol",
}


def parse_file_ops(calls: Iterable[NormalizedCall]) -> list[FileOp]:
    """Extract :class:`FileOp` records from tool calls.

    Uses heuristics on the arguments — the goal is cheap signal for
    checkers, not byte-exact diff reconstruction.
    """
    ops: list[FileOp] = []
    for c in calls:
        if c.name not in _EDIT_TOOLS:
            continue
        args = c.args
        path = (
            args.get("file_path")
            or args.get("path")
            or args.get("filename")
            or ""
        )
        content = (
            args.get("content")
            or args.get("new_content")
            or args.get("replacement")
            or args.get("code")
            or ""
        )
        if not isinstance(content, str):
            content = str(content)
        if c.name == "create_file":
            added = content.count("\n") + (1 if content else 0)
            removed = 0
        elif c.name == "remove_symbol":
            added = 0
            removed = max(1, content.count("\n") + 1)
        elif c.name == "replace_lines":
            added = content.count("\n") + (1 if content else 0)
            start = args.get("start_line") or args.get("start")
            end = args.get("end_line") or args.get("end")
            try:
                removed = max(0, int(end) - int(start) + 1)
            except (TypeError, ValueError):
                removed = added
        else:  # edit_symbol / add_symbol
            added = content.count("\n") + (1 if content else 0)
            removed = added  # approximate
        ops.append(
            FileOp(
                tool=c.name,
                path=str(path),
                lines_added=added,
                lines_removed=removed,
                content=content,
            )
        )
    return ops


def count_line_delta(ops: Iterable[FileOp]) -> tuple[int, int]:
    """Return ``(total_added, total_removed)`` across all *ops*."""
    a = r = 0
    for op in ops:
        a += op.lines_added
        r += op.lines_removed
    return a, r


def biggest_op(ops: Iterable[FileOp]) -> FileOp | None:
    ops = list(ops)
    if not ops:
        return None
    return max(ops, key=lambda o: max(o.lines_added, o.lines_removed))


_TODO_PATTERNS: dict[str, re.Pattern[str]] = {
    "todo": re.compile(r"\b(TODO|FIXME|XXX|HACK)\b"),
    "placeholder": re.compile(
        r"\b(placeholder|not implemented|implement me|unimplemented)\b",
        re.IGNORECASE,
    ),
    "stub_body": re.compile(r"^\s*pass\s*(#.*)?$", re.MULTILINE),
    "ellipsis_stub": re.compile(r"^\s*\.\.\.\s*$", re.MULTILINE),
    "raise_notimpl": re.compile(r"raise\s+NotImplementedError"),
    "left_as_exercise": re.compile(
        r"(left\s+as\s+(an?\s+)?exercise|you\s+can\s+add|fill\s+in)",
        re.IGNORECASE,
    ),
}


def scan_for_todo_markers(content: str) -> Confidence:
    """Scan edited file content for lazy-work markers."""
    if not content:
        return Confidence.none()
    from infinidev.engine.behavior.primitives.text import regex_scan

    return regex_scan(content, _TODO_PATTERNS, per_hit_weight=0.4)
