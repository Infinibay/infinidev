"""What to hand a model when a tool result does not fit.

The four loops that are not the LoopEngine — chat agent, planner, council,
spec elaborator — used to cap a result at N characters and append
``...[truncated]``. Measured against one project's history, that was the
dominant cause of wasted work:

    files over the cap    2 distinct,  29 reads  = 14.5 reads per file
    files under the cap   2 distinct,   3 reads  =  1.5 reads per file

A 22 KB HANDOFF_PROMPT.md was read 21 times. The notice said content was
missing and never said how to get it, so the only move that looked reasonable
was to call again — and the same bytes came back.

Truncation is worse than it looks. The 8 000 characters the model DID get end
mid-word and stay in the transcript forever, so it reasons from a third of a
file believing it read the file, and concludes something is absent when it was
in the 68% that got cut.

So an oversized FILE READ is refused instead, and the refusal carries the
file's outline — headings for prose, definitions for code, with line numbers.
That is the part that matters: a bare refusal makes the model guess at a
range it has no way to know, and guessing costs the same round-trips the
truncation did. Choosing from a table of contents costs one. The outline for
that 432-line handoff is 776 characters, 3% of the file.

Results that cannot be paginated — a diff, a search, a command's output —
still get trimmed, because there is no range to ask for. They get an honest
notice instead of a dead end.
"""

from __future__ import annotations

import json
import re

# ``read_file`` and ``partial_read`` emit "%6d\tcontent" per line.
_NUMBERED_LINE = re.compile(r"^\s*(\d+)\t", re.M)

# Tools that take a line offset, and the argument that carries it.
_PAGINATED: dict[str, str] = {
    "read_file": "offset",
    "partial_read": "start_line",
}

# Marker the UI keys off to show a size refusal differently from a failure.
# It is deliberately a stable literal: the renderer matches it, not a regex.
OVERSIZED_ERROR = "file too large to read in one call"

# What counts as a structural line, by extension. Anchored at column 0 or at
# a small indent so a nested local function does not crowd out the class.
_OUTLINE_PATTERNS: list[tuple[tuple[str, ...], re.Pattern[str]]] = [
    ((".md", ".markdown", ".rst", ".txt"), re.compile(r"^#{1,6} \S")),
    ((".py",), re.compile(r"^ {0,4}(?:async def|def|class) \w")),
    (
        (".ts", ".tsx", ".js", ".jsx", ".mjs"),
        re.compile(
            r"^ {0,2}(?:export\s+)?(?:default\s+)?"
            r"(?:async\s+)?(?:function|class|const|interface|type|enum)\s+\w"
        ),
    ),
    ((".go",), re.compile(r"^(?:func|type) \w")),
    ((".rs",), re.compile(r"^ {0,4}(?:pub )?(?:async )?(?:fn|struct|enum|impl|trait) \w")),
    ((".rb",), re.compile(r"^ {0,2}(?:def|class|module) \w")),
    ((".java", ".kt", ".cs"), re.compile(r"^ {0,4}(?:public|private|protected|internal)\s+\w")),
    ((".sh", ".bash"), re.compile(r"^(?:function )?\w+\s*\(\)\s*\{")),
]

_MAX_OUTLINE_ENTRIES = 60


def _outline_pattern(path: str) -> re.Pattern[str] | None:
    lower = path.lower()
    for exts, pattern in _OUTLINE_PATTERNS:
        if lower.endswith(exts):
            return pattern
    return None


def _strip_line_numbers(text: str) -> list[tuple[int, str]]:
    """Turn ``read_file`` output back into ``(line_number, content)`` pairs."""
    out: list[tuple[int, str]] = []
    for raw in text.splitlines():
        head, tab, rest = raw.partition("\t")
        if tab and head.strip().isdigit():
            out.append((int(head.strip()), rest))
    return out


def _build_outline(path: str, numbered: list[tuple[int, str]]) -> list[str]:
    """Structural lines with their line numbers, capped so it stays small."""
    pattern = _outline_pattern(path)
    if pattern is None:
        return []
    hits = [f"{n:>6}  {line.rstrip()}" for n, line in numbered if pattern.match(line)]
    if len(hits) > _MAX_OUTLINE_ENTRIES:
        dropped = len(hits) - _MAX_OUTLINE_ENTRIES
        hits = hits[:_MAX_OUTLINE_ENTRIES]
        hits.append(f"        ... and {dropped} more, not listed")
    return hits


def _args_dict(raw_args: str | dict) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        try:
            parsed = json.loads(raw_args)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {}


def _path_from(raw_args: str | dict) -> str:
    args = _args_dict(raw_args)
    return str(args.get("file_path") or args.get("path") or "")


def _refuse_file_read(
    result: str, *, max_chars: int, tool_name: str, path: str,
) -> str:
    """Refuse the read and hand back a map of the file instead."""
    numbered = _strip_line_numbers(result)
    total_lines = numbered[-1][0] if numbered else result.count("\n") + 1
    offset_arg = _PAGINATED[tool_name]

    # A chunk that comfortably fits, rounded so the number reads as advice
    # rather than as a computed constant the model tries to hit exactly.
    chars_per_line = max(1, len(result) // max(1, total_lines))
    safe_lines = max(50, (max_chars // chars_per_line) // 50 * 50)

    parts = [
        json.dumps({
            "error": OVERSIZED_ERROR,
            "file_path": path,
            "lines": total_lines,
            "characters": len(result),
            "limit_characters": max_chars,
        }),
        "",
        f"NOTHING was read. {path} is {len(result):,} characters over "
        f"{total_lines} lines, and the limit for one call is {max_chars:,}.",
        "",
        f"Read the part you need: {tool_name}(file_path={path!r}, "
        f"{offset_arg}=<first line>, limit={safe_lines}).",
    ]

    outline = _build_outline(path, numbered)
    if outline:
        parts += ["", "The file's structure, so you can pick a range:", ""]
        parts += outline
    else:
        parts += [
            "",
            f"Start at {offset_arg}=1 and work forward in blocks of "
            f"{safe_lines} lines.",
        ]
    return "\n".join(parts)


def _trim_unpaginated(result: str, *, max_chars: int) -> str:
    """Cut a result there is no way to request a range of."""
    hidden = len(result) - max_chars
    return (
        f"{result[:max_chars]}\n"
        f"...[TRUNCATED. {hidden:,} of {len(result):,} characters were cut "
        f"and are NOT shown above.]\n"
        f"Calling this tool again with the same arguments returns this same "
        f"truncated text. Narrow the request instead."
    )


def handle_oversized_result(
    result: str,
    *,
    max_chars: int,
    tool_name: str = "",
    tool_args: str | dict = "",
) -> str:
    """Return what the model should receive for *result*.

    Under the limit, *result* itself. Over it, a refusal with an outline for a
    file read, or a trimmed result with an honest notice for anything else.
    """
    if len(result) <= max_chars:
        return result

    path = _path_from(tool_args)
    if tool_name in _PAGINATED and path and _NUMBERED_LINE.search(result):
        return _refuse_file_read(
            result, max_chars=max_chars, tool_name=tool_name, path=path,
        )
    return _trim_unpaginated(result, max_chars=max_chars)


def is_oversized_refusal(result: str) -> bool:
    """True when *result* is a size refusal, for the UI to render it apart.

    Checks the parsed error rather than the substring so a file whose content
    happens to quote the marker cannot masquerade as one.
    """
    if not result or not result.startswith("{"):
        return False
    first_line = result.split("\n", 1)[0]
    try:
        parsed = json.loads(first_line)
    except (ValueError, TypeError):
        return False
    return isinstance(parsed, dict) and parsed.get("error") == OVERSIZED_ERROR


class DuplicateCallGuard:
    """Refuse a tool call this run already made with the same arguments.

    The refusal above is advice, and advice is what a model ignores when it is
    stuck. This is the part that terminates: the second identical call never
    reaches the tool, and the model reads a refusal instead of bytes it has.
    One guard per run — a repeat in a later turn is a legitimate re-read.
    """

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], int] = {}

    @staticmethod
    def _key(tool_name: str, tool_args: str | dict) -> tuple[str, str]:
        """Normalize arguments so key order cannot disguise a repeat."""
        if isinstance(tool_args, str):
            try:
                parsed = json.loads(tool_args)
            except (ValueError, TypeError):
                return (tool_name, tool_args.strip())
            if isinstance(parsed, dict):
                return (tool_name, json.dumps(parsed, sort_keys=True))
            return (tool_name, str(parsed))
        if isinstance(tool_args, dict):
            return (tool_name, json.dumps(tool_args, sort_keys=True))
        return (tool_name, str(tool_args))

    def refusal_for(self, tool_name: str, tool_args: str | dict) -> str | None:
        """Text to hand back INSTEAD of running the tool, or ``None``."""
        key = self._key(tool_name, tool_args)
        first_seen = self._seen.get(key)
        self._seen[key] = (first_seen or 0) + 1
        if first_seen is None:
            return None

        detail = (
            f"You already called {tool_name} with these exact arguments "
            f"earlier in this turn. Its result is in this conversation "
            f"above — scroll up and use it. Running it again returns "
            f"identical bytes."
        )
        lines = [json.dumps({"error": "duplicate call", "detail": detail})]
        if tool_name in _PAGINATED and (path := _path_from(tool_args)):
            arg = _PAGINATED[tool_name]
            lines.append(
                f"If you need a part of {path} you have not seen, pass "
                f"{arg}= the line you want to start at."
            )
        else:
            lines.append(
                "If you need something you have not seen, change the "
                "arguments. If you have what you need, act on it."
            )
        return "\n".join(lines)

