"""Detailed trace logging for the loop engine.

Activated by setting the env var ``INFINIDEV_TRACE_FILE`` to a file path.
When inactive every function is a no-op so production has zero overhead.

Captured per task run:
  * task description, expected output, model id, key settings
  * each iteration's full XML user prompt
  * each LLM response: reasoning_content (thinking), content, tool calls (with args)
  * the plan after every step transition
  * each tool execution: name, args, brief result/error
  * final result and high-level stats

The output format is plain text with banner separators so a human can scroll
through and see the model's chain of thought across the whole run.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Any

_LOCK = threading.Lock()
_HANDLE = None
_PATH: str | None = None
_START_TS: float | None = None


def _is_enabled() -> bool:
    return os.environ.get("INFINIDEV_TRACE_FILE") not in (None, "")


def _open() -> None:
    global _HANDLE, _PATH, _START_TS
    if _HANDLE is not None or not _is_enabled():
        return
    _PATH = os.environ["INFINIDEV_TRACE_FILE"]
    os.makedirs(os.path.dirname(os.path.abspath(_PATH)) or ".", exist_ok=True)
    _HANDLE = open(_PATH, "a", encoding="utf-8", buffering=1)
    _START_TS = time.time()


def _ts() -> str:
    if _START_TS is None:
        return "+0.0s"
    return f"+{time.time() - _START_TS:6.1f}s"


def _w(text: str) -> None:
    if not _is_enabled():
        return
    try:
        from infinidev.engine.static_analysis_timer import measure as _sa_measure
        _ctx = _sa_measure("trace_log")
    except Exception:
        from contextlib import nullcontext
        _ctx = nullcontext()
    with _ctx:
        with _LOCK:
            _open()
            if _HANDLE is None:
                return
            try:
                _HANDLE.write(text)
                _HANDLE.write("\n")
            except Exception as exc:  # pragma: no cover - tracing must never raise
                print(f"[trace_log] write failed: {exc}", file=sys.stderr)


def _banner(char: str, label: str) -> str:
    bar = char * 80
    return f"\n{bar}\n{_ts()}  {label}\n{bar}"


def _truncate(s: str, n: int = 6000) -> str:
    if s is None:
        return ""
    s = str(s)
    if len(s) <= n:
        return s
    return s[:n] + f"\n…[truncated {len(s) - n} chars]"


# ── public API ───────────────────────────────────────────────────────────

def trace_run_start(*, model: str, task: str, expected: str, settings_snapshot: dict | None = None) -> None:
    if not _is_enabled():
        return
    _w(_banner("=", f"RUN START  model={model}"))
    _w(f"TASK:\n{_truncate(task)}")
    if expected:
        _w(f"\nEXPECTED OUTPUT:\n{_truncate(expected, 1500)}")
    if settings_snapshot:
        _w("\nSETTINGS:")
        for k, v in settings_snapshot.items():
            _w(f"  {k} = {v}")


def trace_iteration_prompt(iteration: int, system_prompt: str, user_prompt: str) -> None:
    if not _is_enabled():
        return
    _w(_banner("─", f"ITER {iteration}  PROMPT (user role)"))
    _w(_truncate(user_prompt, 12000))


def trace_llm_response(
    iteration: int,
    *,
    reasoning: str | None,
    content: str | None,
    tool_calls: list[Any] | None,
) -> None:
    if not _is_enabled():
        return
    _w(_banner("·", f"ITER {iteration}  LLM RESPONSE"))
    if reasoning:
        _w(f"[THINKING]\n{_truncate(reasoning, 8000)}")
    if content:
        _w(f"\n[CONTENT]\n{_truncate(content, 4000)}")
    if tool_calls:
        _w(f"\n[TOOL CALLS] ({len(tool_calls)})")
        for i, tc in enumerate(tool_calls):
            name = getattr(tc, "name", None)
            args = None
            fn = getattr(tc, "function", None)
            if fn is not None:
                name = name or getattr(fn, "name", None)
                args = getattr(fn, "arguments", None)
            if name is None and isinstance(tc, dict):
                name = tc.get("name") or (tc.get("function") or {}).get("name")
                args = (tc.get("function") or {}).get("arguments") or tc.get("arguments")
            try:
                if isinstance(args, str):
                    parsed = json.loads(args)
                    args_str = json.dumps(parsed, indent=2, ensure_ascii=False)
                else:
                    args_str = json.dumps(args, indent=2, ensure_ascii=False, default=str)
            except Exception:
                args_str = str(args)
            _w(f"  ({i + 1}) {name}")
            _w(_truncate(args_str, 2000))
    if not (reasoning or content or tool_calls):
        _w("[empty response]")


def trace_tool_result(iteration: int, tool_name: str, args: Any, result: Any) -> None:
    if not _is_enabled():
        return
    try:
        if isinstance(args, str):
            args_p = json.loads(args)
        else:
            args_p = args
        args_str = json.dumps(args_p, ensure_ascii=False)[:300]
    except Exception:
        args_str = str(args)[:300]
    _w(f"\n[TOOL RESULT] {tool_name}({args_str})")
    _w(_truncate(str(result), 2000))


def trace_plan(iteration: int, plan: Any) -> None:
    if not _is_enabled():
        return
    steps = getattr(plan, "steps", None) or []
    _w(_banner("·", f"ITER {iteration}  PLAN ({len(steps)} steps)"))
    for s in steps:
        idx = getattr(s, "index", "?")
        title = getattr(s, "title", "")
        status = getattr(s, "status", "")
        marker = {"done": "●", "active": "○", "pending": "◌", "blocked": "✗"}.get(status, "·")
        _w(f"  {marker} [{status:8}] {idx}. {title}")


def trace_step_done(iteration: int, status: str, summary: str, tool_calls: int) -> None:
    if not _is_enabled():
        return
    _w(_banner("·", f"ITER {iteration}  STEP DONE  status={status}  tool_calls={tool_calls}"))
    if summary:
        _w(_truncate(summary, 1500))


def trace_run_end(status: str, iterations: int, total_tools: int, final_result: str) -> None:
    if not _is_enabled():
        return
    _w(_banner("=", f"RUN END  status={status}  iterations={iterations}  tools={total_tools}"))
    if final_result:
        _w(_truncate(final_result, 4000))
    with _LOCK:
        global _HANDLE
        if _HANDLE is not None:
            try:
                _HANDLE.flush()
                _HANDLE.close()
            except Exception:
                pass
            _HANDLE = None
