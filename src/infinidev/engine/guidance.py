"""Reactive guidance system for small models that get stuck.

Most models don't need this. Frontier and 30B+ local models plan and
edit fine on their own — injecting guidance for them is just token
waste. The system is designed to fire **only** when a model has
demonstrated a clear stuck-pattern that pre-baked advice can resolve,
and only when ``is_small`` is true.

Three pieces, one file:

  * **Detector** — pure functions that scan the most recent step's
    messages and the running ``LoopState`` for a small set of well-
    defined stuck-patterns (no LLM call, ~O(messages) work).
  * **Library** — a small dict of :class:`GuidanceEntry` objects keyed
    by pattern name. Each entry is a short title + body + concrete
    example, capped at ~250 tokens.
  * **Hook helpers** — :func:`maybe_queue_guidance` is called by the
    engine after each step; :func:`drain_pending_guidance` is called by
    the prompt builder before rendering the next iteration.

Hard guarantees:
  * Never delivers the same entry twice in one task (``guidance_given``).
  * Never delivers more than ``LOOP_GUIDANCE_MAX_PER_TASK`` entries.
  * Never fires for non-small models.
  * Never costs an LLM call. Detection is regex + counter math.

The patterns and library are intentionally small (≈8 entries). Adding
more is a one-line change to ``_LIBRARY``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState
    from infinidev.engine.loop.execution_context import ExecutionContext


# ── Data ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GuidanceEntry:
    """A single piece of pre-baked how-to advice for a stuck-pattern."""

    key: str
    title: str
    body: str
    example: str = ""

    def render(self) -> str:
        """Format the entry as a self-contained ``<guidance>`` XML block."""
        parts = [f"<guidance pattern=\"{self.key}\">", f"## {self.title}", "", self.body]
        if self.example:
            parts.extend(["", "Concrete example:", "```", self.example, "```"])
        parts.append("</guidance>")
        return "\n".join(parts)


# ── Library ───────────────────────────────────────────────────────────────

_LIBRARY: dict[str, GuidanceEntry] = {
    "stuck_on_planning": GuidanceEntry(
        key="stuck_on_planning",
        title="How to write a concrete plan step",
        body=(
            "Each step you create with add_step must name a FILE PATH, a "
            "FUNCTION (with parens), and the SPECIFIC change. Vague titles "
            "like 'implement the feature' get flagged as warnings and the "
            "model that wrote them tends to drift. Set expected_output "
            "to a short, verifiable check the step must pass."
        ),
        example=(
            "BAD:  add_step(title='Implement the auth module')\n"
            "GOOD: add_step(\n"
            "        title='Add JWT exp check to verify_token() in auth.py:52',\n"
            "        expected_output='pytest tests/test_auth.py::test_expired passes',\n"
            "      )"
        ),
    ),
    "stuck_on_edit": GuidanceEntry(
        key="stuck_on_edit",
        title="How to edit existing files in this project",
        body=(
            "Use replace_lines for surgical edits — it takes a file_path "
            "plus start_line, end_line, and the replacement content. "
            "Read the file first to find the exact line numbers; do not "
            "guess. The pre-write syntax check will reject the edit if "
            "the result has invalid Python — fix the indentation and "
            "retry, do not work around it. For brand-new files use "
            "create_file. For renaming a function, use edit_symbol."
        ),
        example=(
            "1. read_file('src/auth.py')   # see verify_token at line 42-58\n"
            "2. replace_lines(\n"
            "     file_path='src/auth.py',\n"
            "     start_line=52, end_line=52,\n"
            "     content='    return payload[\"exp\"] is not None\\n',\n"
            "   )\n"
            "3. execute_command('pytest tests/test_auth.py -v')"
        ),
    ),
    "stuck_on_tests": GuidanceEntry(
        key="stuck_on_tests",
        title="When pytest keeps failing, READ the traceback",
        body=(
            "If the test runs but the assertion fails, the exit_code tells "
            "you nothing useful — the actual error message is in stdout. "
            "Read it. Look for the line starting with 'E ' (the assertion "
            "failure or exception) and the file:line above it (where the "
            "test failed). Then add_note the failure mode before editing "
            "again. Patching blindly without reading the traceback is the "
            "main reason a small model loops on the same broken edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_x.py::test_foo -v 2>&1 | tail -40')\n"
            "2. add_note('test_foo expects status=200 but got 404 — handler missing route')\n"
            "3. read_file('src/handler.py')  # find the route table\n"
            "4. replace_lines(...)           # add the route\n"
            "5. execute_command('pytest tests/test_x.py::test_foo -v')"
        ),
    ),
    "reread_loop": GuidanceEntry(
        key="reread_loop",
        title="Stop re-reading the same file",
        body=(
            "You have already read this file recently. Its content is in "
            "<opened-files> in your prompt. Re-reading wastes tokens and "
            "the file has not changed since the last read. Either edit it, "
            "or use add_note to save what you found and move on. If you "
            "need to see only a specific range, use read_file with offset "
            "and limit instead of fetching the whole file again."
        ),
        example=(
            "Wrong: read_file('src/auth.py') × 3\n"
            "Right:\n"
            "  read_file('src/auth.py')\n"
            "  add_note('verify_token at auth.py:42, missing exp check')\n"
            "  replace_lines(file_path='src/auth.py', start_line=52, ...)"
        ),
    ),
    "unknown_tool": GuidanceEntry(
        key="unknown_tool",
        title="You are calling tools that don't exist",
        body=(
            "The error 'Unknown tool: X' means X is not in the registered "
            "toolset for this run. Do NOT keep retrying it. Call the help "
            "tool to see what is available, or pick from these common "
            "ones: read_file, replace_lines, edit_file, create_file, "
            "code_search, glob, list_directory, execute_command, "
            "add_note, add_step, modify_step, step_complete."
        ),
        example=(
            "Wrong: search_in_files(...)        # not a real tool\n"
            "Right: code_search(query='verify_token', path='src/')"
        ),
    ),
    "vague_steps": GuidanceEntry(
        key="vague_steps",
        title="Your steps are too vague to act on",
        body=(
            "Several of your add_step calls did not name a file path, "
            "function, or line — that is why the tool returned a warning "
            "and you cannot make progress. Use modify_step on each vague "
            "step to add a concrete file:line and the specific change. "
            "Then set expected_output to one short verifiable sentence."
        ),
        example=(
            "modify_step(\n"
            "  index=2,\n"
            "  title='Add count_records() to minidb.py:24 returning len(self.tables)',\n"
            "  expected_output='pytest tests/test_count.py::test_basic passes',\n"
            ")"
        ),
    ),
    "text_only_iters": GuidanceEntry(
        key="text_only_iters",
        title="You must call tools, not just write text",
        body=(
            "Your last responses contained text only and no tool calls. "
            "The loop cannot make progress without tool calls. Pick the "
            "single most useful next action right now (read a file, edit "
            "a file, run a command) and emit it as a tool call. If you "
            "are unsure what to do, call read_file on the most relevant "
            "file or list_directory on the workspace."
        ),
        example=(
            "Wrong:\n"
            "  'I should now read the auth module to check the token logic.'\n"
            "Right (actual tool call, not narration):\n"
            "  read_file('src/auth.py')"
        ),
    ),
    "stuck_on_search": GuidanceEntry(
        key="stuck_on_search",
        title="Stop searching, start reading",
        body=(
            "You have run multiple searches without opening any of the "
            "results. The next step is to pick the most promising hit "
            "and read_file on it. If the searches all returned nothing, "
            "the term you are searching for does not exist in this "
            "codebase — try a synonym or use list_directory to discover "
            "the structure first."
        ),
        example=(
            "code_search(query='auth', path='src/') -> 5 hits\n"
            "read_file('src/auth/handlers.py')   # actually read the top hit\n"
            "add_note('auth flow lives in src/auth/handlers.py:42 verify()')"
        ),
    ),
}


# ── Detector ──────────────────────────────────────────────────────────────

# Patterns that the detector recognises in tool error responses.
_UNKNOWN_TOOL_RE = re.compile(r"Unknown tool[: ]", re.IGNORECASE)
_NO_PLAN_CTX_RE = re.compile(r"No active plan context", re.IGNORECASE)
_VAGUE_WARN_RE = re.compile(r"Vague step title", re.IGNORECASE)


def _tool_calls_in_messages(messages: list[dict]) -> list[tuple[str, str]]:
    """Extract (tool_name, raw_arguments) tuples from assistant messages."""
    out: list[tuple[str, str]] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            args = fn.get("arguments") or tc.get("arguments") or ""
            if isinstance(args, dict):
                args = json.dumps(args)
            out.append((str(name), str(args)))
    return out


def _tool_results(messages: list[dict]) -> list[str]:
    """Extract content of tool-role messages (the tool result strings)."""
    return [
        str(m.get("content") or "")
        for m in messages
        if m.get("role") == "tool"
    ]


def _has_text_only_iters(state: "LoopState", n: int = 2) -> bool:
    """True iff the model produced ``n`` consecutive iterations with no
    tool calls — measured from the running text-only counter on the
    LoopGuard, which the engine increments after each step."""
    counter = getattr(state, "text_only_iterations", 0)
    return counter >= n


def _has_unknown_tool_loop(messages: list[dict]) -> bool:
    """True iff the model called a non-existent tool 2+ times."""
    results = _tool_results(messages)
    return sum(1 for r in results if _UNKNOWN_TOOL_RE.search(r)) >= 2


def _has_pytest_loop_no_read(messages: list[dict]) -> bool:
    """True iff there are 3+ pytest invocations and 0 reads of any test
    or implementation file in between (model is patching blind)."""
    calls = _tool_calls_in_messages(messages)
    pytest_runs = sum(
        1 for name, args in calls
        if name == "execute_command" and "pytest" in args.lower()
    )
    if pytest_runs < 3:
        return False
    # Did the model read_file at least once between pytest runs?
    saw_pytest = False
    saw_read_after_pytest = False
    for name, _ in calls:
        if name == "execute_command":
            if saw_pytest:
                # New pytest run with no read in between
                return not saw_read_after_pytest
            saw_pytest = True
            saw_read_after_pytest = False
        elif name in ("read_file", "partial_read"):
            saw_read_after_pytest = True
    return False


def _has_repeated_edit_errors(messages: list[dict]) -> bool:
    """True iff 3+ edit calls (replace_lines/edit_file) returned errors."""
    calls = _tool_calls_in_messages(messages)
    results = _tool_results(messages)
    if not results:
        return False
    # Pair tool calls with their results 1:1, in order
    edit_errors = 0
    result_idx = 0
    for name, _ in calls:
        if result_idx >= len(results):
            break
        if name in ("replace_lines", "edit_file", "multi_edit_file", "create_file"):
            r = results[result_idx]
            if r.startswith('{"error"'):
                edit_errors += 1
        result_idx += 1
    return edit_errors >= 3


def _has_vague_step_spam(messages: list[dict]) -> bool:
    """True iff 3+ ``add_step`` results carried the vague-title warning."""
    return sum(1 for r in _tool_results(messages) if _VAGUE_WARN_RE.search(r)) >= 3


def _has_reread_loop(messages: list[dict], state: "LoopState") -> bool:
    """True iff the same file path was read 3+ times in this step."""
    counts: dict[str, int] = {}
    for name, args in _tool_calls_in_messages(messages):
        if name not in ("read_file", "partial_read"):
            continue
        try:
            payload = json.loads(args) if args else {}
        except Exception:
            continue
        path = payload.get("file_path") or payload.get("path") or ""
        if path:
            counts[path] = counts.get(path, 0) + 1
    return any(c >= 3 for c in counts.values())


def _has_stuck_on_search(messages: list[dict]) -> bool:
    """True iff 4+ search/glob/list_directory calls with 0 read_file."""
    search_calls = 0
    read_calls = 0
    for name, _ in _tool_calls_in_messages(messages):
        if name in ("code_search", "glob", "list_directory", "search_symbols"):
            search_calls += 1
        elif name in ("read_file", "partial_read"):
            read_calls += 1
    return search_calls >= 4 and read_calls == 0


# Order matters: we return the first matching pattern, and the order
# encodes priority (most-specific / highest-confidence first).
_DETECTORS: list[tuple[str, Any]] = [
    ("text_only_iters", lambda m, s: _has_text_only_iters(s)),
    ("unknown_tool",    lambda m, s: _has_unknown_tool_loop(m)),
    ("vague_steps",     lambda m, s: _has_vague_step_spam(m)),
    ("reread_loop",     lambda m, s: _has_reread_loop(m, s)),
    ("stuck_on_tests",  lambda m, s: _has_pytest_loop_no_read(m)),
    ("stuck_on_edit",   lambda m, s: _has_repeated_edit_errors(m)),
    ("stuck_on_search", lambda m, s: _has_stuck_on_search(m)),
    # NB: stuck_on_planning has no automatic detector — it's only
    # delivered explicitly via maybe_queue_guidance(force_key="stuck_on_planning")
    # because vague_steps already covers the same ground.
]


def detect_stuck_pattern(messages: list[dict], state: "LoopState") -> str | None:
    """Run all detectors in priority order and return the first match.

    *messages* should be the slice of the message buffer that belongs
    to the just-finished step (e.g. ``messages[step_messages_start:]``).
    Returns the matching pattern key or None.
    """
    for key, fn in _DETECTORS:
        try:
            if fn(messages, state):
                return key
        except Exception:
            # A detector failure must never crash the loop.
            continue
    return None


# ── Public hook API ───────────────────────────────────────────────────────

def maybe_queue_guidance(
    state: "LoopState",
    messages: list[dict],
    *,
    is_small: bool,
    max_per_task: int,
) -> str | None:
    """Detect a stuck-pattern and queue the matching guidance entry.

    Returns the key that was queued (for logging) or None. Safe to call
    every step — short-circuits on non-small models, on tasks that have
    already received the per-task quota, and when the next prompt
    already has a guidance queued.
    """
    if not is_small:
        return None
    if state.pending_guidance:
        return None
    if len(state.guidance_given) >= max_per_task:
        return None

    key = detect_stuck_pattern(messages, state)
    if not key or key in state.guidance_given:
        return None

    entry = _LIBRARY.get(key)
    if not entry:
        return None

    state.pending_guidance = entry.render()
    state.guidance_given.append(key)
    return key


def drain_pending_guidance(state: "LoopState") -> str:
    """Pop and return the queued guidance text, or "" if none.

    Called by ``build_iteration_prompt`` exactly once per iteration.
    Idempotent: a second call returns "" until something queues again.
    """
    txt = state.pending_guidance
    state.pending_guidance = ""
    return txt


__all__ = [
    "GuidanceEntry",
    "detect_stuck_pattern",
    "maybe_queue_guidance",
    "drain_pending_guidance",
]
