"""Stuck-pattern detectors for the guidance system.

Each detector is a pure function over the just-finished step's
message buffer (and optionally the running ``LoopState``). It returns
``bool`` for "did this pattern fire?" and never raises.

The full set of detectors is registered in :data:`_DETECTORS` in
priority order. :func:`detect_stuck_pattern` walks the list and
returns the first key that matches, or ``None``.

Adding a new detector:
  1. Write a ``_has_<name>`` function here.
  2. Add it to ``_DETECTORS`` at the appropriate priority.
  3. Add a matching :class:`GuidanceEntry` in ``library.py`` with the
     same key.
"""

from __future__ import annotations

import json
import re
from typing import Any, TYPE_CHECKING

from infinidev.engine.guidance.test_runners import (
    is_test_command,
    test_outcome_fingerprint,
)

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState


# ── Compiled regexes (module-level for fast reuse) ───────────────────────

_UNKNOWN_TOOL_RE = re.compile(r"Unknown tool[: ]", re.IGNORECASE)
_VAGUE_WARN_RE = re.compile(r"Vague step title", re.IGNORECASE)

# Patterns that look like a tool call rendered as text instead of as a
# real function call. Each one matches a JSON shape the model would
# normally emit ONLY through the function-calling channel.
_MALFORMED_TC_RES: tuple[re.Pattern[str], ...] = (
    # OpenAI-style: {"tool_calls": [{"name": "...", "arguments": ...}]}
    re.compile(r'"tool_calls"\s*:\s*\[\s*\{\s*"name"\s*:'),
    # Bare-call style: {"name": "...", "arguments": {...}}
    re.compile(r'\{\s*"name"\s*:\s*"[a-z_][a-z0-9_]*"\s*,\s*"arguments"\s*:'),
    # function field variant
    re.compile(r'"function"\s*:\s*\{\s*"name"\s*:'),
    # XML-tag variants emitted by some chat templates
    re.compile(r"<tool_call>|<function_call>|<\|tool_call_begin\|>|<\|tool_calls_section_begin\|>"),
)


# ── Message-buffer helpers ───────────────────────────────────────────────

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


# ── Individual detectors ─────────────────────────────────────────────────

def _has_malformed_tool_call(messages: list[dict]) -> bool:
    """True iff the model emitted text that LOOKS like a tool call but
    didn't actually call any tool.

    Catches the small-model failure mode where the model writes
    ``{"tool_calls": [...]}`` or ``{"name": "x", "arguments": {...}}``
    as plain text inside its content/thinking instead of emitting it
    through the function-calling channel. The engine already prints
    ``No function call detected (retry N/3)`` for this case but the
    model gets no actionable feedback. The matching guidance entry
    explains the difference between writing JSON and calling a tool.

    Fires when ANY assistant message in the step has content matching
    one of the malformed-tool-call regex patterns AND that same
    message has no real ``tool_calls``. We require at least one
    occurrence (not 2+) because a single attempt is already a clear
    sign the model doesn't understand the tool-calling API.
    """
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        if msg.get("tool_calls"):
            continue  # had real tool calls — fine
        content = str(msg.get("content") or "")
        if not content:
            continue
        for pattern in _MALFORMED_TC_RES:
            if pattern.search(content):
                return True
    return False


def _has_first_test_run(state: "LoopState") -> bool:
    """True iff the model has just run a test runner for the first time
    and we haven't yet introduced ``tail_test_output``.

    This is a *proactive* detector — unlike the rest of the detectors
    which fire when something is going wrong, this one fires on a
    benign event (the first time ``last_test_output`` is non-empty)
    so the model learns about the structured-failure tool BEFORE it
    gets stuck. The ``maybe_queue_guidance`` quota and the
    ``guidance_given`` dedup ensure it never delivers more than once
    per task and never crowds out higher-priority guidance later.
    """
    return bool(getattr(state, "last_test_output", "")) and \
        "first_test_run" not in getattr(state, "guidance_given", [])


def _has_text_only_iters(state: "LoopState", n: int = 2) -> bool:
    """True iff the model produced ``n`` consecutive iterations with no
    tool calls — measured from the running text-only counter on the
    LoopGuard."""
    counter = getattr(state, "text_only_iterations", 0)
    return counter >= n


def _has_unknown_tool_loop(messages: list[dict]) -> bool:
    """True iff the model called a non-existent tool 2+ times."""
    results = _tool_results(messages)
    return sum(1 for r in results if _UNKNOWN_TOOL_RE.search(r)) >= 2


def _has_test_loop_no_read(messages: list[dict], state: "LoopState | None" = None) -> bool:
    """True iff there are 3+ FAILING test runs and 0 reads in between.

    The "all runs failed" requirement avoids false positives when the
    model is making progress. The runner-agnostic check uses
    :func:`is_test_command` and :func:`test_outcome_fingerprint`.
    """
    calls = _tool_calls_in_messages(messages)
    results = _tool_results(messages)

    test_runs = sum(
        1 for name, args in calls
        if name == "execute_command" and is_test_command(args, state)
    )
    if test_runs < 3:
        return False

    failing_runs = 0
    for r in results:
        fp = test_outcome_fingerprint(r)
        if fp and ("failed" in fp or "error" in fp):
            failing_runs += 1
    if failing_runs < 3:
        return False

    saw_test = False
    saw_read_after_test = False
    for name, args in calls:
        if name == "execute_command" and is_test_command(args, state):
            if saw_test:
                return not saw_read_after_test
            saw_test = True
            saw_read_after_test = False
        elif name in ("read_file", "partial_read"):
            saw_read_after_test = True
    return False


def _has_same_test_output_loop(messages: list[dict]) -> bool:
    """True iff the last 3 test runs returned identical pass/fail counts.

    Stronger signal than :func:`_has_test_loop_no_read`: if the OUTCOME
    didn't change after 3 attempts, the model is editing the wrong
    thing regardless of how many files it read in between.
    """
    fingerprints: list[str] = []
    for msg in messages:
        if msg.get("role") != "tool":
            continue
        content = str(msg.get("content") or "")
        fp = test_outcome_fingerprint(content)
        if fp:
            fingerprints.append(fp)
    if len(fingerprints) < 3:
        return False
    return len(set(fingerprints[-3:])) == 1


def _has_repeated_edit_errors(messages: list[dict]) -> bool:
    """True iff 3+ edit calls (replace_lines/edit_file) returned errors."""
    calls = _tool_calls_in_messages(messages)
    results = _tool_results(messages)
    if not results:
        return False
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


def _normalize_step_title(title: str) -> str:
    """Lowercase + strip non-essential punctuation + collapse whitespace.

    Used as the input to similarity matching so cosmetic differences
    (capitalisation, trailing dots, double spaces) don't hide real
    duplicates.
    """
    s = title.lower()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _has_duplicate_steps(state: "LoopState | None") -> bool:
    """True iff the current plan has 2+ near-duplicate step titles.

    Uses ``difflib.SequenceMatcher.ratio()`` over normalised titles
    with a 0.78 threshold — high enough to ignore legitimately similar
    parallel steps (different filenames pull the ratio down) but low
    enough to catch the typical replanning bug where 3-4 variants of
    "Read test files" end up in the plan together.

    Considers only ``pending``/``active`` steps.
    """
    if state is None or not getattr(state, "plan", None):
        return False
    steps = [
        s for s in state.plan.steps
        if getattr(s, "status", "") in ("pending", "active")
    ]
    if len(steps) < 3:
        return False

    from difflib import SequenceMatcher
    titles = [_normalize_step_title(s.title) for s in steps]
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            if not titles[i] or not titles[j]:
                continue
            ratio = SequenceMatcher(None, titles[i], titles[j]).ratio()
            if ratio >= 0.78:
                return True
    return False


# ── Registry ─────────────────────────────────────────────────────────────

# Order encodes priority: the first matching detector wins.
# duplicate_steps fires early because a noisy plan poisons every
# downstream signal — clean it before reasoning about the rest.
# same_test_output_loop runs before stuck_on_tests because it's a
# stronger "no progress" signal independent of read activity.
_DETECTORS: list[tuple[str, Any]] = [
    # text_only, unknown_tool, malformed_tool_call are highest priority
    # — they signal a broken loop that nothing else can recover from.
    # malformed_tool_call sits BEFORE text_only because it's a strict
    # subset (text-only with a JSON-shaped fragment) and the more
    # specific guidance is more useful.
    ("malformed_tool_call",   lambda m, s: _has_malformed_tool_call(m)),
    ("text_only_iters",       lambda m, s: _has_text_only_iters(s)),
    ("unknown_tool",          lambda m, s: _has_unknown_tool_loop(m)),
    # first_test_run is a *proactive* introduction — fires once when the
    # model first runs a test command, so it knows about
    # tail_test_output BEFORE getting stuck.
    ("first_test_run",        lambda m, s: _has_first_test_run(s)),
    ("duplicate_steps",       lambda m, s: _has_duplicate_steps(s)),
    ("vague_steps",           lambda m, s: _has_vague_step_spam(m)),
    ("reread_loop",           lambda m, s: _has_reread_loop(m, s)),
    ("same_test_output_loop", lambda m, s: _has_same_test_output_loop(m)),
    ("stuck_on_tests",        lambda m, s: _has_test_loop_no_read(m, s)),
    ("stuck_on_edit",         lambda m, s: _has_repeated_edit_errors(m)),
    ("stuck_on_search",       lambda m, s: _has_stuck_on_search(m)),
    # NB: stuck_on_planning has no automatic detector — it's only
    # delivered explicitly via ``maybe_queue_guidance(force_key=...)``.
]


def detect_stuck_pattern(messages: list[dict], state: "LoopState") -> str | None:
    """Run all detectors in priority order and return the first match.

    *messages* should be the slice of the message buffer that belongs
    to the just-finished step (e.g. ``messages[step_messages_start:]``).
    A detector failure never crashes the loop — it's caught and skipped.
    """
    for key, fn in _DETECTORS:
        try:
            if fn(messages, state):
                return key
        except Exception:
            continue
    return None


__all__ = ["detect_stuck_pattern"]
