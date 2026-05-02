"""ToolCallWidget — renders every tool call in the chat with its full payload.

Replaces the old "format_tool_chat_message" approach (which dropped most
tools on the floor and showed a 1-line summary for the survivors).  Each
tool gets a structured render: its arguments and its full result, always
expanded — no accordion, no collapse.

Visual format (claude-code / opencode style):

    ▸ tool_name ──────────────────────────────────────────
    ▌
    ▌  language  python
    ▌
    ▌  code
    ▌    print("hello")
    ▌    print("line2")
    ▌
    ▌  ✓ exit 0
    ▌
    ▌  stdout
    ▌    hello
    ▌    line2
    ▌

The vertical accent bar groups the tool block visually; section labels
(in dim italic) provide hierarchy without piling on color.

The widget receives messages of the shape:

    {
        "type": "tool_call",
        "tool_name": str,
        "args": dict,
        "result": str,           # full tool output
        "error": str,            # extracted error string (if any)
        "exec_data": dict | None # parsed exec envelope (execute_command/code_interpreter)
    }

Per-tool formatters live in ``_TOOL_FORMATTERS`` and return a list of
"sections". A *section* is one of:

    {"kind": "kv",      "key": str, "value": str}     # inline:  key  value
    {"kind": "label",   "text": str}                  # standalone italic label
    {"kind": "block",   "lines": list[str], "style": str}  # indented body
    {"kind": "status",  "icon": str, "text": str, "ok": bool}  # ✓/✗ chip
    {"kind": "spacer"}                                # blank divider

The widget post-processes sections into rendered lines, prepending the
accent bar uniformly so formatters never need to know about decoration.
Tools without a formatter get a generic key:value dump.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

import time as _time

import infinidev.ui.controls.message_widgets as _mw
from infinidev.ui.controls.message_widgets import (
    RenderResult, register,
    COPY_ICON, COPY_ICON_STYLE,
    COPY_OK_ICON, COPY_OK_STYLE,
    COPY_FAIL_ICON, COPY_FAIL_STYLE,
)
from infinidev.ui.theme import TEXT, TEXT_MUTED


# ── Style tokens ─────────────────────────────────────────────────────────
# Restrained palette. One accent for the tool block (the vertical bar +
# header marker). Secondary information in dim. Output bodies use a soft
# off-white. Errors are the only saturated colour.

_TC_BAR_FG     = "#7c70a8"          # vertical accent bar (muted purple)
_TC_HEADER_FG  = "#b3a5db bold"     # tool name in header (slightly brighter)
_TC_RULE_FG    = TEXT_MUTED          # the trailing ── after the tool name
_TC_LABEL_FG   = TEXT_MUTED + " italic"  # section labels: language / code / stdout
_TC_KEY_FG     = TEXT_MUTED          # inline arg keys
_TC_VAL_FG     = TEXT                # arg values, primary content
_TC_BODY_FG    = "#bcbcbc"          # general body text
_TC_OUT_FG     = "#a8b8c8"          # stdout-like output (soft blue-gray)
_TC_ERR_FG     = "#d48a8a"          # stderr-like / partial errors
_TC_FAIL_FG    = "#ff5577 bold"     # hard error
_TC_OK_FG      = "#5fd07f"          # success marker (✓)


# ── Section types ────────────────────────────────────────────────────────
# Formatters return a list of these. The widget renders them inside the
# accent-bar frame so formatters never need to know about decoration.

@dataclass
class _Section:
    kind: str
    key: str = ""
    value: str = ""
    text: str = ""
    lines: list[str] | None = None
    style: str = _TC_VAL_FG
    icon: str = ""
    ok: bool = True


def _kv(key: str, value: Any) -> _Section:
    return _Section(kind="kv", key=key, value=_stringify(value, max_len=240))


def _label(text: str) -> _Section:
    return _Section(kind="label", text=text)


def _block(text: str, style: str = _TC_VAL_FG) -> _Section:
    if isinstance(text, str):
        lines = text.split("\n")
    else:
        lines = [str(text)]
    return _Section(kind="block", lines=lines, style=style)


def _status(text: str, ok: bool) -> _Section:
    return _Section(kind="status", text=text, ok=ok)


def _spacer() -> _Section:
    return _Section(kind="spacer")


# ── Helpers ──────────────────────────────────────────────────────────────


_MAX_BODY_LINES = 200       # cap any tool body so a runaway output doesn't blow the viewport
_BAR_GUTTER     = "  "       # 2-space gutter after the bar
_INNER_INDENT   = "  "       # indent inside the gutter for block content


def _truncate(s: str, w: int) -> str:
    if len(s) <= w:
        return s
    return s[: max(1, w - 1)] + "…"


def _stringify(v: Any, max_len: int = 200) -> str:
    """Render an arbitrary arg value as a readable string."""
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        s = str(v)
    else:
        try:
            s = json.dumps(v, ensure_ascii=False)
        except Exception:
            s = repr(v)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"


def _try_parse_exec(result: str) -> dict | None:
    """Parse an execute_command / code_interpreter JSON envelope."""
    if not result:
        return None
    s = result.strip()
    if not s.startswith("{"):
        return None
    try:
        d = json.loads(s)
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    if "exit_code" not in d and "stdout" not in d and "stderr" not in d:
        return None
    return d


def _try_parse_json(s: Any) -> Any:
    """Best-effort JSON parse — returns dict/list or None.

    Used by the generic fallback to detect tools that return a raw
    JSON envelope and pretty-print it instead of dumping the whole
    thing as one wall of text.
    """
    if not isinstance(s, str):
        return None
    stripped = s.strip()
    if not stripped or not (stripped.startswith("{") or stripped.startswith("[")):
        return None
    try:
        return json.loads(stripped)
    except Exception:
        return None


def _format_size(n: Any) -> str:
    """Human-readable byte size: 842 B / 1.2 KB / 5.3 MB."""
    try:
        n = int(n)
    except (TypeError, ValueError):
        return ""
    if n < 0:
        return ""
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    if n < 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    return f"{n / (1024 * 1024 * 1024):.1f} GB"


# ── Per-tool formatters ──────────────────────────────────────────────────
# Each formatter receives (args, result, error, width) and returns a list
# of _Section objects. The widget composes them inside the accent frame.

SectionList = list[_Section]


def _fmt_default(tool_name: str, args: dict, result: str, error: str, width: int) -> SectionList:
    """Generic fallback: dump every arg as kv, then result.

    If the result is a JSON dict/list we render it as structured
    sections (top-level keys → kv rows; nested values → indented
    block) so tools that wrap their output in `_success({...})` don't
    leak raw `{"file_path": "...", "entries": [...]}` text.
    """
    sections: SectionList = []
    if args:
        for k, v in args.items():
            val = _stringify(v, max_len=width * 2)
            if "\n" in val or len(val) > 80:
                sections.append(_label(k))
                sections.append(_block(val, _TC_VAL_FG))
            else:
                sections.append(_kv(k, val))

    if result and not error:
        if sections:
            sections.append(_spacer())
        parsed = _try_parse_json(result)
        if isinstance(parsed, dict):
            sections.extend(_render_json_dict(parsed))
        elif isinstance(parsed, list):
            sections.extend(_render_json_list(parsed))
        else:
            sections.append(_label("result"))
            sections.append(_block(result, _TC_OUT_FG))
    return sections


def _render_json_dict(d: dict, *, max_list_items: int = 30) -> SectionList:
    """Render a top-level JSON dict as kv rows + nested blocks.

    Lists of dicts → one line per item, formatted as compact
    `key=value  key=value` rather than a multi-line JSON dump.
    """
    out: SectionList = []
    for k, v in d.items():
        if isinstance(v, list):
            n = len(v)
            out.append(_kv(k, f"{n} item(s)"))
            if n == 0:
                continue
            shown = v[:max_list_items]
            if all(isinstance(x, (str, int, float, bool)) or x is None for x in shown):
                # Scalar list — bullet block
                out.append(_block("\n".join(_stringify(x, 240) for x in shown), _TC_OUT_FG))
            elif all(isinstance(x, dict) for x in shown):
                # List of dicts — render each as a compact key=value line
                lines = []
                for item in shown:
                    pairs = "  ".join(
                        f"{ik}={_stringify(iv, 80)}" for ik, iv in item.items()
                    )
                    lines.append(pairs)
                out.append(_block("\n".join(lines), _TC_OUT_FG))
            else:
                # Mixed — fall back to indented JSON
                out.append(_block(json.dumps(shown, indent=2, default=str), _TC_OUT_FG))
            if n > max_list_items:
                out.append(_label(f"… +{n - max_list_items} more"))
        elif isinstance(v, dict):
            out.append(_label(k))
            # Render inner dict as nested kv (one level deep) — keeps it compact
            for ik, iv in v.items():
                if isinstance(iv, (dict, list)):
                    out.append(_kv(f"  {ik}", _stringify(iv, 240)))
                else:
                    out.append(_kv(f"  {ik}", _stringify(iv, 240)))
        else:
            out.append(_kv(k, _stringify(v, 240)))
    return out


def _smart_result(result: str) -> SectionList:
    """Render a tool result intelligently: JSON dict/list → structured;
    plain text → block. Used by formatters that don't care about a
    specific schema but want JSON output to look reasonable.
    """
    parsed = _try_parse_json(result)
    if isinstance(parsed, dict):
        return _render_json_dict(parsed)
    if isinstance(parsed, list):
        return _render_json_list(parsed)
    return [_block(result, _TC_OUT_FG)]


def _render_json_list(items: list, *, max_items: int = 30) -> SectionList:
    """Render a top-level JSON list."""
    out: SectionList = [_kv("count", str(len(items)))]
    if not items:
        return out
    out.append(_spacer())
    shown = items[:max_items]
    for item in shown:
        if isinstance(item, dict):
            pairs = "  ".join(f"{k}={_stringify(v, 60)}" for k, v in item.items())
            out.append(_block(pairs, _TC_OUT_FG))
        else:
            out.append(_block(_stringify(item, 200), _TC_OUT_FG))
    if len(items) > max_items:
        out.append(_label(f"… +{len(items) - max_items} more"))
    return out


def _fmt_read_file(args, result, error, width) -> SectionList:
    # Args-only: file content would clutter the chat. The agent
    # consumes the body internally; the user just needs to see *what*
    # was read.
    path = args.get("path") or args.get("file_path") or "?"
    return [_kv("path", path)]


def _fmt_partial_read(args, result, error, width) -> SectionList:
    # Args-only: the bytes are food for the agent.
    path = args.get("path") or args.get("file_path") or "?"
    start = args.get("start_line") or args.get("line_start") or args.get("start")
    end = args.get("end_line") or args.get("line_end") or args.get("end")
    label = f"{path}:{start}-{end}" if start is not None and end is not None else path
    return [_kv("range", label)]


def _fmt_create_file(args, result, error, width) -> SectionList:
    path = args.get("path") or args.get("file_path") or "?"
    content = args.get("content") or ""
    s: SectionList = [_kv("path", path)]
    if content:
        s.append(_spacer())
        s.append(_label("content"))
        body = content if len(content) < 8000 else content[:8000] + "\n…"
        s.append(_block(body, _TC_VAL_FG))
    return s


def _fmt_replace_lines(args, result, error, width) -> SectionList:
    path = args.get("path") or args.get("file_path") or "?"
    ls = args.get("line_start") or args.get("start_line") or "?"
    le = args.get("line_end") or args.get("end_line") or "?"
    new_lines = args.get("new_lines") or args.get("new_content") or ""
    s: SectionList = [_kv("at", f"{path}:{ls}-{le}")]
    if new_lines:
        s.append(_spacer())
        s.append(_label("new_lines"))
        s.append(_block(new_lines, _TC_VAL_FG))
    return s


def _fmt_list_directory(args, result, error, width) -> SectionList:
    # Args-only: a full listing is noise in the chat. Show what was
    # requested; the agent has the entries.
    path = args.get("path") or args.get("directory") or "."
    s: SectionList = [_kv("path", path)]
    if args.get("recursive"):
        s.append(_kv("recursive", "true"))
    if args.get("pattern"):
        s.append(_kv("pattern", str(args.get("pattern"))))
    return s


def _fmt_code_search(args, result, error, width) -> SectionList:
    q = args.get("query") or args.get("pattern") or args.get("search_query") or ""
    path = args.get("file_path") or args.get("path") or ""
    s: SectionList = [_kv("query", q)]
    if path:
        s.append(_kv("path", str(path)))
    if not (result and not error):
        return s
    parsed = _try_parse_json(result)
    if isinstance(parsed, dict) and "match_count" in parsed:
        s.append(_kv("matches", str(parsed.get("match_count", 0))))
        if parsed.get("truncated"):
            s.append(_label("(truncated)"))
    else:
        s.append(_kv("matches", str(len(result.splitlines()))))
    return s


def _fmt_glob(args, result, error, width) -> SectionList:
    p = args.get("pattern") or args.get("glob_pattern") or ""
    s: SectionList = [_kv("pattern", p)]
    if not (result and not error):
        return s

    parsed = _try_parse_json(result)
    if isinstance(parsed, dict) and "matches" in parsed:
        matches = parsed.get("matches") or []
        match_count = parsed.get("match_count", len(matches))
        truncated = parsed.get("truncated", False)
        s.append(_kv("matches", str(match_count)))
        if truncated:
            s.append(_label("(truncated)"))
        if matches:
            s.append(_spacer())
            s.append(_block("\n".join(str(m) for m in matches), _TC_OUT_FG))
        else:
            s.append(_label("(no matches)"))
    else:
        # Plain text fallback
        lines = [l for l in result.splitlines() if l.strip()]
        s.append(_kv("paths", str(len(lines))))
        s.append(_spacer())
        s.append(_block(result, _TC_OUT_FG))
    return s


def _fmt_shell_like(args, result, error, width, *, code_key: str | None = None) -> SectionList:
    """Shared renderer for execute_command and code_interpreter."""
    cmd_or_code = args.get(code_key) if code_key else (args.get("command") or args.get("cmd") or "")
    language = args.get("language") or args.get("lang") or ""
    s: SectionList = []

    if language:
        s.append(_kv("language", language))

    if cmd_or_code:
        text = str(cmd_or_code)
        if "\n" in text or len(text) > 70:
            s.append(_label("code" if code_key else "command"))
            s.append(_block(text, _TC_VAL_FG))
        else:
            s.append(_kv("$", text))

    env = _try_parse_exec(result)
    if env is not None:
        exit_code = env.get("exit_code")
        stdout = (env.get("stdout") or "").rstrip("\n")
        stderr = (env.get("stderr") or "").rstrip("\n")
        killed = env.get("killed_reason")
        s.append(_spacer())
        if exit_code == 0 and not killed:
            s.append(_status("✓ exit 0", ok=True))
        elif killed:
            s.append(_status(f"✗ killed: {_truncate(str(killed), 80)}", ok=False))
        elif exit_code is not None:
            s.append(_status(f"✗ exit {exit_code}", ok=False))
        if stdout:
            s.append(_label("stdout"))
            s.append(_block(stdout, _TC_OUT_FG))
        if stderr:
            s.append(_label("stderr"))
            s.append(_block(stderr, _TC_ERR_FG))
        if not stdout and not stderr and not killed:
            s.append(_label("(no output)"))
    elif result and not error:
        s.append(_spacer())
        s.append(_label("result"))
        s.append(_block(result, _TC_OUT_FG))

    return s


def _fmt_execute_command(args, result, error, width) -> SectionList:
    return _fmt_shell_like(args, result, error, width)


def _fmt_code_interpreter(args, result, error, width) -> SectionList:
    return _fmt_shell_like(args, result, error, width, code_key="code")


def _fmt_get_symbol_code(args, result, error, width) -> SectionList:
    # Args-only: the symbol body is for the agent.
    name = args.get("name") or args.get("symbol_name") or "?"
    return [_kv("symbol", name)]


def _fmt_list_symbols(args, result, error, width) -> SectionList:
    # Args-only: a long symbol list is noise in the chat.
    path = args.get("file_path") or args.get("path") or "?"
    return [_kv("file", path)]


def _fmt_search_symbols(args, result, error, width) -> SectionList:
    q = args.get("query") or ""
    s: SectionList = [_kv("query", q)]
    if result and not error:
        s.append(_spacer())
        s.extend(_smart_result(result))
    return s


def _fmt_find_references(args, result, error, width) -> SectionList:
    name = args.get("name") or "?"
    s: SectionList = [_kv("symbol", name)]
    if result and not error:
        # Result may be plain text (one ref per line, header on top)
        # or JSON. _smart_result handles both.
        if not _try_parse_json(result):
            n = max(0, len(result.splitlines()) - 1)  # minus header
            s.append(_kv("references", str(n)))
        s.append(_spacer())
        s.extend(_smart_result(result))
    return s


def _fmt_edit_symbol(args, result, error, width) -> SectionList:
    name = args.get("name") or args.get("symbol_name") or "?"
    new_code = args.get("new_code") or args.get("code") or ""
    s: SectionList = [_kv("symbol", name)]
    if new_code:
        s.append(_spacer())
        s.append(_label("new_code"))
        s.append(_block(new_code, _TC_VAL_FG))
    return s


def _fmt_add_symbol(args, result, error, width) -> SectionList:
    name = args.get("name") or "?"
    file_path = args.get("file_path") or args.get("path") or "?"
    code = args.get("code") or ""
    s: SectionList = [
        _kv("symbol", name),
        _kv("file", file_path),
    ]
    if code:
        s.append(_spacer())
        s.append(_label("code"))
        s.append(_block(code, _TC_VAL_FG))
    return s


def _fmt_remove_symbol(args, result, error, width) -> SectionList:
    return [_kv("symbol", args.get("name") or "?")]


def _fmt_project_structure(args, result, error, width) -> SectionList:
    # Args-only: the tree itself is for the agent.
    path = args.get("path") or args.get("directory") or "."
    return [_kv("path", path)]


def _fmt_git_diff(args, result, error, width) -> SectionList:
    target = args.get("file_path") or args.get("branch") or ""
    s: SectionList = []
    if target:
        s.append(_kv("target", target))
    if result and not error:
        if s:
            s.append(_spacer())
        # git_diff is always plain text — skip JSON heuristic
        s.append(_block(result, _TC_OUT_FG))
    return s


def _fmt_git_simple(args, result, error, width) -> SectionList:
    primary = (
        args.get("branch_name")
        or args.get("name")
        or args.get("message")
        or args.get("branch")
        or ""
    )
    s: SectionList = []
    if primary:
        s.append(_kv("arg", primary))
    if result and not error:
        if s:
            s.append(_spacer())
        s.extend(_smart_result(result))
    return s


def _fmt_record_finding(args, result, error, width) -> SectionList:
    # Args-only: the persisted body is in args; the result is just
    # a "stored OK / id=N" envelope which is noise.
    title = args.get("title") or "?"
    content = args.get("content") or args.get("body") or ""
    s: SectionList = [_kv("title", title)]
    if content:
        s.append(_spacer())
        s.append(_label("content"))
        s.append(_block(content, _TC_VAL_FG))
    return s


def _fmt_search_findings(args, result, error, width) -> SectionList:
    # Search → the matches ARE the value. Keep result.
    q = args.get("query") or ""
    s: SectionList = [_kv("query", q)]
    if result and not error:
        s.append(_spacer())
        s.extend(_smart_result(result))
    return s


def _fmt_read_findings(args, result, error, width) -> SectionList:
    # Args-only: bulk read used for agent context, not user-facing.
    q = args.get("query") or ""
    return [_kv("query", q)] if q else []


def _fmt_web(args, result, error, width) -> SectionList:
    # Args-only: web result is consumed by the agent. The user
    # already sees the URL/query — that's what they wanted to know.
    target = args.get("query") or args.get("url") or ""
    return [_kv("target", target)]


_TOOL_FORMATTERS: dict[str, Callable[[dict, str, str, int], SectionList]] = {
    "read_file": _fmt_read_file,
    "partial_read": _fmt_partial_read,
    "create_file": _fmt_create_file,
    "write_file": _fmt_create_file,
    "replace_lines": _fmt_replace_lines,
    "list_directory": _fmt_list_directory,
    "code_search": _fmt_code_search,
    "glob": _fmt_glob,
    "execute_command": _fmt_execute_command,
    "code_interpreter": _fmt_code_interpreter,
    "get_symbol_code": _fmt_get_symbol_code,
    "list_symbols": _fmt_list_symbols,
    "search_symbols": _fmt_search_symbols,
    "find_references": _fmt_find_references,
    "find_definition": _fmt_find_references,
    "edit_symbol": _fmt_edit_symbol,
    "add_symbol": _fmt_add_symbol,
    "remove_symbol": _fmt_remove_symbol,
    "project_structure": _fmt_project_structure,
    "git_diff": _fmt_git_diff,
    "git_branch": _fmt_git_simple,
    "git_commit": _fmt_git_simple,
    "git_status": _fmt_git_simple,
    "git_push": _fmt_git_simple,
    "record_finding": _fmt_record_finding,
    "search_findings": _fmt_search_findings,
    "read_findings": _fmt_read_findings,
    "search_knowledge": _fmt_search_findings,
    "web_search": _fmt_web,
    "web_fetch": _fmt_web,
}


# ── Widget ───────────────────────────────────────────────────────────────


LineList = list[list[tuple[str, str]]]


def _serialize_for_copy(msg: dict[str, Any]) -> str:
    """Build a clean plain-text representation of a tool call for the clipboard.

    Preference order: a sensible per-tool format (so pasting into chat
    or a doc reads naturally), then a JSON-y fallback. The output
    deliberately strips the bar/header chrome and ANSI markers — what
    the user wants to share is the *content*, not the rendering.
    """
    tool_name = msg.get("tool_name", "?")
    args = msg.get("args") or {}
    if not isinstance(args, dict):
        args = {}
    result = msg.get("result") or ""
    error = msg.get("error") or ""

    parts: list[str] = [f"{tool_name}"]

    # Per-tool tweaks where it improves the paste
    if tool_name in ("execute_command",):
        cmd = args.get("command") or args.get("cmd") or ""
        if cmd:
            parts.append(f"$ {cmd}")
    elif tool_name == "code_interpreter":
        lang = args.get("language") or args.get("lang") or ""
        code = args.get("code") or ""
        if lang:
            parts.append(f"language: {lang}")
        if code:
            parts.append(f"```\n{code}\n```")
    else:
        # Generic: show every arg as `key: value`
        for k, v in args.items():
            sval = _stringify(v, max_len=10_000)
            if "\n" in sval:
                parts.append(f"{k}:\n{sval}")
            else:
                parts.append(f"{k}: {sval}")

    # Result: prefer parsed exec envelope when present
    env = _try_parse_exec(result)
    if env is not None:
        ec = env.get("exit_code")
        sout = (env.get("stdout") or "").rstrip("\n")
        serr = (env.get("stderr") or "").rstrip("\n")
        if ec is not None:
            parts.append(f"exit {ec}")
        if sout:
            parts.append(f"stdout:\n{sout}")
        if serr:
            parts.append(f"stderr:\n{serr}")
    elif result:
        parts.append(f"result:\n{result}")

    if error:
        parts.append(f"error: {error}")

    return "\n".join(parts).strip()


def _wrap_text(text: str, content_width: int) -> list[str]:
    """Hard-wrap one logical line to the given visual width."""
    out: list[str] = []
    if not text:
        return [""]
    while len(text) > content_width:
        out.append(text[:content_width])
        text = text[content_width:]
    out.append(text)
    return out


class ToolCallWidget:
    """Renders a `tool_call` chat message — every tool, every parameter,
    framed by a soft accent bar with header rule."""

    msg_type = "tool_call"
    group_label = "Tools"

    def render(self, msg: dict[str, Any], width: int) -> RenderResult:
        tool_name = msg.get("tool_name", "?")
        args = msg.get("args") or {}
        if not isinstance(args, dict):
            args = {}
        result = msg.get("result") or ""
        error = msg.get("error") or ""

        # ── 1. Compose sections via per-tool formatter ────────────────
        formatter = _TOOL_FORMATTERS.get(tool_name)
        try:
            sections = formatter(args, result, error, width) if formatter else _fmt_default(tool_name, args, result, error, width)
        except Exception:
            sections = _fmt_default(tool_name, args, result, error, width)

        # ── 2. Header line: "▸ tool_name ───────  [⧉]"  ───────────────
        # The copy button anchors to the right edge; the horizontal
        # rule stretches to fill what's left between the tool name
        # and the button. Recently-copied messages show a transient
        # ✓/✗ feedback icon (handled by the shared _copy_highlight).
        marker_style = _TC_BAR_FG if not error else _TC_FAIL_FG
        marker_text = "▸ " if not error else "✗ "

        now = _time.monotonic()
        hl = _mw._copy_highlight.get(id(msg))
        if hl and (now - hl[0]) < _mw._COPY_FEEDBACK_DURATION:
            copy_label, copy_style = (COPY_OK_ICON, COPY_OK_STYLE) if hl[1] else (COPY_FAIL_ICON, COPY_FAIL_STYLE)
        else:
            copy_label, copy_style = COPY_ICON, COPY_ICON_STYLE

        head_used = len(marker_text) + len(tool_name) + 1 + len(copy_label)
        rule_len = max(0, width - head_used)
        lines: LineList = [[
            (marker_style + " bold", marker_text),
            (_TC_HEADER_FG, tool_name),
            (" ", " "),
            (_TC_RULE_FG, "─" * rule_len),
            (copy_style, copy_label),
        ]]
        clickable: dict[int, Callable[[], None]] = {}

        def _copy_msg(m=msg):
            try:
                from infinidev.ui.clipboard import copy_to_clipboard
                ok = copy_to_clipboard(_serialize_for_copy(m))
            except Exception:
                ok = False
            _mw._copy_highlight[id(m)] = (_time.monotonic(), ok)
            cb = getattr(_mw, "_copy_feedback", None)
            if cb:
                try:
                    cb(ok)
                except Exception:
                    pass
        clickable[0] = _copy_msg

        # ── 3. Render each section, prepending the accent bar ─────────
        bar = (_TC_BAR_FG, "▌")
        gutter = ("", _BAR_GUTTER)  # 2 spaces after the bar
        content_width = max(20, width - 1 - len(_BAR_GUTTER))  # bar + gutter

        def _bar_line(*frags: tuple[str, str]) -> list[tuple[str, str]]:
            return [bar, gutter, *frags]

        # Top breathing room
        lines.append(_bar_line(("", "")))

        for sec in sections:
            if sec.kind == "spacer":
                lines.append(_bar_line(("", "")))
                continue

            if sec.kind == "kv":
                key_text = f"{sec.key}"
                # 2-space gap between key and value, key in dim italic.
                head = [
                    (_TC_LABEL_FG, key_text),
                    ("", "  "),
                    (_TC_VAL_FG, sec.value),
                ]
                # Wrap if too long.
                visible = sum(len(t) for _, t in head)
                if visible <= content_width:
                    lines.append(_bar_line(*head))
                else:
                    # Key + value spill — render key on one line, value below
                    lines.append(_bar_line((_TC_LABEL_FG, key_text)))
                    for chunk in _wrap_text(sec.value, content_width - len(_INNER_INDENT)):
                        lines.append(_bar_line(("", _INNER_INDENT), (_TC_VAL_FG, chunk)))
                continue

            if sec.kind == "label":
                lines.append(_bar_line((_TC_LABEL_FG, sec.text)))
                continue

            if sec.kind == "block":
                inner_width = max(10, content_width - len(_INNER_INDENT))
                raw_lines = sec.lines or []
                shown = raw_lines[:_MAX_BODY_LINES]
                for raw in shown:
                    for chunk in _wrap_text(raw, inner_width):
                        lines.append(_bar_line(("", _INNER_INDENT), (sec.style, chunk)))
                if len(raw_lines) > _MAX_BODY_LINES:
                    hidden = len(raw_lines) - _MAX_BODY_LINES
                    lines.append(_bar_line((_TC_LABEL_FG, f"… {hidden} more line(s) (full output in result)")))
                continue

            if sec.kind == "status":
                style = _TC_OK_FG if sec.ok else _TC_FAIL_FG
                lines.append(_bar_line((style, sec.text)))
                continue

        # ── 4. Hard error footer (if not already shown by formatter) ──
        if error:
            lines.append(_bar_line(("", "")))
            for chunk in _wrap_text(error, content_width - 2):
                lines.append(_bar_line((_TC_FAIL_FG, "✗ " + chunk)))

        # ── 5. Closing bar + chat separator ───────────────────────────
        lines.append(_bar_line(("", "")))
        lines.append([("", "")])

        return RenderResult(lines=lines, clickable_offsets=clickable)

    def render_group_header(self, count: int, collapsed: bool, width: int) -> RenderResult:
        # Tool calls are NOT grouped (see message_groups.NEVER_GROUP_TYPES).
        # Fallback flat header for safety.
        label = f"{self.group_label} ({count})"
        pad = " " * max(0, width - len(label) - 1)
        return RenderResult(lines=[[(_TC_HEADER_FG, f" {label}{pad}")]])


register(ToolCallWidget())
