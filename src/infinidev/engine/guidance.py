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
        title="When tests keep failing, READ the failure output",
        body=(
            "exit_code tells you nothing useful — the actual error message "
            "is in stdout/stderr. READ it. For each test framework the "
            "key signal is in a different place:\n"
            "  • pytest:   look for lines starting with 'E   ' and the "
            "'FAILED' summary at the bottom\n"
            "  • jest/vitest: look for the '●' or 'FAIL' block above each "
            "failed test, plus the 'Expected'/'Received' diff\n"
            "  • mocha:    look for the '✗' / 'AssertionError' lines\n"
            "  • cargo test: look for 'thread ... panicked at' and the "
            "expected vs actual values\n"
            "  • go test:  look for '--- FAIL: TestX' and the lines below it\n"
            "  • dotnet/mvn: look for the '[xUnit.net]'/'<<< FAILURE!' lines\n"
            "After reading, add_note the EXACT failure mode (file:line + "
            "what was expected vs actual), THEN open the relevant file, "
            "THEN edit. Patching blindly without reading is the main "
            "reason a small model loops on the same broken edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_x.py::test_foo -v 2>&1 | tail -40')\n"
            "2. add_note('test_foo: expected 200, got 404 at handler.py:52 — route missing')\n"
            "3. read_file('src/handler.py')\n"
            "4. replace_lines(...)\n"
            "5. execute_command('pytest tests/test_x.py::test_foo -v')"
        ),
    ),
    "same_test_output_loop": GuidanceEntry(
        key="same_test_output_loop",
        title="Your edits are not changing the test outcome — switch tactics",
        body=(
            "You have run the test runner 3+ times and the pass/fail count "
            "is IDENTICAL each time. Your edits are not affecting the "
            "failing test. This means EITHER (a) you are editing the wrong "
            "file or wrong line, OR (b) the bug is somewhere you haven't "
            "looked yet. STOP editing and do a diagnostic step: "
            "1) isolate ONE failing test (e.g. `pytest path::name -v`, "
            "`jest -t 'name'`, `cargo test name`, `go test -run TestName`),"
            " 2) capture the FULL failure output (not just the exit code), "
            "3) add_note the EXACT file:line where the error is raised "
            "and what was expected vs actual, "
            "4) read THAT file at THAT line. Only then edit."
        ),
        example=(
            "1. execute_command('pytest tests/test_foo.py::test_one -v --tb=long 2>&1 | tail -60')\n"
            "2. add_note('TypeError at minidb.py:92 inside _parse_values: quote_char is None')\n"
            "3. read_file('minidb.py')   # focus on _parse_values around line 92\n"
            "4. modify_step(index=N, expected_output='_parse_values handles empty values without crashing')\n"
            "5. replace_lines('minidb.py', start_line=88, end_line=95, content=...)"
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


def _has_test_loop_no_read(messages: list[dict], state: "LoopState | None" = None) -> bool:
    """True iff there are 3+ FAILING test runs and 0 reads in between.

    Generalises across all runners that ``is_test_command`` recognises:
    pytest, jest, vitest, mocha, cargo test, go test, dotnet test, mvn,
    etc. The "all runs failed" requirement avoids false positives when
    the model is making progress and the failure count is changing —
    that case is either healthy iteration or, if the count is stuck,
    ``same_test_output_loop`` catches it instead.
    """
    calls = _tool_calls_in_messages(messages)
    results = _tool_results(messages)

    test_runs = sum(
        1 for name, args in calls
        if name == "execute_command" and is_test_command(args, state)
    )
    if test_runs < 3:
        return False

    # Require at least 3 results that look like a *failing* test run.
    failing_runs = 0
    for r in results:
        fp = test_outcome_fingerprint(r)
        if fp and ("failed" in fp or "error" in fp):
            failing_runs += 1
    if failing_runs < 3:
        return False

    # Did the model read_file at least once between consecutive test runs?
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


# Backwards-compat alias for any code that imported the old name.
_has_pytest_loop_no_read = _has_test_loop_no_read


# ── Test runner detection (multi-language) ────────────────────────────────
#
# Detection of "the model just ran tests" must work across runners and
# languages. We split the problem in two:
#
#   1. is_test_command(args)        — was this execute_command a test run?
#   2. test_outcome_fingerprint(out)— stable hash of the run's outcome
#
# Both functions are intentionally permissive: they aim for HIGH RECALL
# (don't miss real test runs) and acceptable precision. False positives
# in the fingerprint are harmless because the same_test_output_loop
# detector also requires the fingerprint to repeat 3 times in a row.

# Substrings that indicate a test runner invocation. Matched anywhere
# in the command line. Add more here as new runners come up.
_TEST_RUNNER_TOKENS: tuple[str, ...] = (
    # Python
    "pytest", "py.test", "python -m unittest", "nose2", "trial ",
    # JavaScript / TypeScript
    "jest", "vitest", "mocha", "ava ", "tap ", "tape ", "node --test",
    "npm test", "npm run test", "yarn test", "pnpm test", "bun test",
    # Rust
    "cargo test", "cargo nextest",
    # Go
    "go test",
    # .NET
    "dotnet test",
    # JVM
    "mvn test", "mvn verify", "gradle test", "gradlew test", "./gradlew test",
    # Ruby
    "rspec", "rake test", "minitest",
    # PHP
    "phpunit", "pest ",
    # Elixir
    "mix test",
    # Swift
    "swift test", "xcodebuild test",
    # C/C++
    "ctest", "make test", "make check",
)


def _user_test_command_tokens() -> tuple[str, ...]:
    """Read user-declared test-command tokens from settings.

    The setting ``INFINIBAY_LOOP_CUSTOM_TEST_COMMANDS`` accepts a
    comma-separated list of substrings (e.g. ``bash test.sh,make
    integration``). Each substring is added on top of
    ``_TEST_RUNNER_TOKENS`` so the detector recognises project-specific
    test runners (custom shell scripts, Makefile targets, etc.) without
    losing the built-in defaults.
    """
    try:
        from infinidev.config.settings import settings
        raw = getattr(settings, "LOOP_CUSTOM_TEST_COMMANDS", "") or ""
    except Exception:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _runtime_test_command_tokens(state: "LoopState | None" = None) -> tuple[str, ...]:
    """Read agent-declared test-command tokens from the running LoopState.

    The agent can call ``declare_test_command(command_pattern)`` mid-task
    to teach the detector about a project-specific test runner it just
    discovered (e.g. a custom shell wrapper). Tokens are stored on the
    LoopState in ``custom_test_commands`` so they survive iterations
    without polluting global config.
    """
    if state is None:
        return ()
    tokens = getattr(state, "custom_test_commands", None) or []
    return tuple(t.lower() for t in tokens if t)


def is_test_command(args_str: str, state: "LoopState | None" = None) -> bool:
    """True iff the execute_command arguments look like a test runner call.

    Three sources contribute, in priority order (any match wins):
      1. Built-in ``_TEST_RUNNER_TOKENS`` — pytest, jest, cargo, etc.
      2. User-declared tokens via ``LOOP_CUSTOM_TEST_COMMANDS`` setting.
      3. Agent-declared tokens via ``declare_test_command`` tool, stored
         on the LoopState (passed in as *state*).

    The agent and user contributions never conflict with the defaults
    — they are additive. This means a project that uses both pytest and
    a custom ``bash integration.sh`` script gets both detected without
    extra work.
    """
    s = args_str.lower()
    if any(token in s for token in _TEST_RUNNER_TOKENS):
        return True
    for token in _user_test_command_tokens():
        if token in s:
            return True
    for token in _runtime_test_command_tokens(state):
        if token in s:
            return True
    return False


# A keyword that, when paired with a number nearby, signals a test
# outcome. Order doesn't matter — we collect all (number, keyword)
# pairs from the output and use their multiset as the fingerprint.
_OUTCOME_KEYWORDS: tuple[str, ...] = (
    "passed", "failed", "errors", "error",
    "passing", "failing", "skipped", "pending",
    "ok", "fail", "pass",
    "total",  # jest summary line
    "tests run", "failures",  # mvn / surefire
    "successes", "ignored",   # cargo test
)

# Compiled in advance: matches "<int> <keyword>" or "<keyword>: <int>"
# in either order, case-insensitive, allowing punctuation between.
_NUMBER_KEYWORD_RE = re.compile(
    r"(?:(\d+)\s*(?:[,;:|]\s*)?\s*("
    + "|".join(re.escape(k) for k in _OUTCOME_KEYWORDS)
    + r"))|(?:("
    + "|".join(re.escape(k) for k in _OUTCOME_KEYWORDS)
    + r")\s*:\s*(\d+))",
    re.IGNORECASE,
)


def test_outcome_fingerprint(content: str) -> str | None:
    """Extract a stable fingerprint of a test run's outcome.

    Works across pytest, jest, vitest, mocha, cargo test, go test,
    dotnet test, mvn, etc. Returns a sorted, normalised string like
    ``"1 failed, 2 passed"`` or ``"3 passed"`` — identical outcomes
    on different runs produce the same string. Returns None when the
    content doesn't look like test output at all.
    """
    if not content:
        return None
    lower = content.lower()
    # Cheap pre-filter: skip clearly-not-test content.
    if not any(k in lower for k in ("passed", "failed", "test", "ok", "fail", "error")):
        return None

    # Collect (number:int, keyword:str) pairs from the entire output.
    pairs: dict[str, int] = {}
    for m in _NUMBER_KEYWORD_RE.finditer(content):
        if m.group(1) and m.group(2):
            num, kw = m.group(1), m.group(2)
        elif m.group(3) and m.group(4):
            kw, num = m.group(3), m.group(4)
        else:
            continue
        try:
            n = int(num)
        except ValueError:
            continue
        kw_norm = kw.lower().strip()
        # Normalise synonyms so different runners merge cleanly.
        if kw_norm in ("fail", "failing", "failures"):
            kw_norm = "failed"
        elif kw_norm in ("pass", "passing", "successes", "ok"):
            kw_norm = "passed"
        elif kw_norm == "errors":
            kw_norm = "error"
        # Take the LAST occurrence per keyword — runners often print
        # progress (e.g. "0 passed" then "3 passed") and the final
        # value is the authoritative one.
        pairs[kw_norm] = n

    if not pairs:
        return None
    # Drop "total"=N when other counters explain everything — keeps
    # fingerprints stable across runners that do/don't include totals.
    keep_order = ("failed", "error", "passed", "skipped", "pending", "ignored", "total")
    parts = [f"{pairs[k]} {k}" for k in keep_order if k in pairs]
    if not parts:
        return None
    return ", ".join(parts)


# Backwards-compat alias used elsewhere in the file. Older code paths
# expect the pytest-specific name; the new generic implementation
# handles pytest fine.
_pytest_outcome_fingerprint = test_outcome_fingerprint


def _has_same_test_output_loop(messages: list[dict]) -> bool:
    """True iff the last 3 test runs returned identical pass/fail counts.

    Works across any runner ``test_outcome_fingerprint`` recognises
    (pytest, jest, cargo, go test, mvn, etc.). This is a stronger
    signal than ``stuck_on_tests``: it doesn't matter whether the
    model read files in between — if the OUTCOME of the test runner
    didn't change after 3 attempts, the model is editing the wrong
    thing and needs a different tactic. T3v2 (glm-4.7-flash on minidb)
    was the canonical example: model kept editing minidb.py while
    ``1 failed, 2 passed`` stayed identical across iterations because
    the failing test exposed a bug the model misread.
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
    # Same outcome 3+ times in a row at the tail end of the buffer.
    return len(set(fingerprints[-3:])) == 1


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
    ("text_only_iters",       lambda m, s: _has_text_only_iters(s)),
    ("unknown_tool",          lambda m, s: _has_unknown_tool_loop(m)),
    ("vague_steps",           lambda m, s: _has_vague_step_spam(m)),
    ("reread_loop",           lambda m, s: _has_reread_loop(m, s)),
    # same_test_output_loop runs BEFORE stuck_on_tests because it's a
    # stronger signal: identical outcome regardless of read activity.
    ("same_test_output_loop", lambda m, s: _has_same_test_output_loop(m)),
    ("stuck_on_tests",        lambda m, s: _has_test_loop_no_read(m, s)),
    ("stuck_on_edit",         lambda m, s: _has_repeated_edit_errors(m)),
    ("stuck_on_search",       lambda m, s: _has_stuck_on_search(m)),
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
