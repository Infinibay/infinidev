"""Multi-language test-runner detection and outcome fingerprinting.

Two reusable primitives used by the stuck-pattern detectors:

  * :func:`is_test_command` — was this ``execute_command`` invocation
    a test runner call? Recognises 16+ runners across Python, JS/TS,
    Rust, Go, .NET, JVM, Ruby, PHP, Elixir, Swift, C/C++. Three sources
    of tokens contribute and combine additively:
    1. ``_TEST_RUNNER_TOKENS`` — built-in defaults
    2. ``LOOP_CUSTOM_TEST_COMMANDS`` setting — user-declared, persistent
    3. ``state.custom_test_commands`` — agent-declared via the
       ``declare_test_command`` tool, scoped to the current task

  * :func:`test_outcome_fingerprint` — extract a stable, normalised
    string from a runner's stdout (e.g. ``"1 failed, 2 passed"``) so
    different runners producing the same outcome get the same
    fingerprint and identical outcomes can be compared with set
    operations.

Both functions are intentionally permissive (high recall, acceptable
precision). False positives in the fingerprint are harmless because
the downstream detectors require a fingerprint to repeat 3 times in
a row before firing.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState


# ── Built-in runner tokens ───────────────────────────────────────────────

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


# ── Token sources ────────────────────────────────────────────────────────

def _user_test_command_tokens() -> tuple[str, ...]:
    """Read user-declared test-command tokens from settings.

    The setting ``LOOP_CUSTOM_TEST_COMMANDS`` accepts a comma-separated
    list of substrings (e.g. ``"bash test.sh,make integration"``).
    """
    try:
        from infinidev.config.settings import settings
        raw = getattr(settings, "LOOP_CUSTOM_TEST_COMMANDS", "") or ""
    except Exception:
        return ()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def _runtime_test_command_tokens(state: "LoopState | None" = None) -> tuple[str, ...]:
    """Read agent-declared test-command tokens from the running LoopState.

    Populated by the ``declare_test_command`` meta tool.
    """
    if state is None:
        return ()
    tokens = getattr(state, "custom_test_commands", None) or []
    return tuple(t.lower() for t in tokens if t)


def is_test_command(args_str: str, state: "LoopState | None" = None) -> bool:
    """True iff the execute_command arguments look like a test runner call.

    Sources are additive — declaring ``bash test.sh`` does not disable
    pytest detection. A project that uses both pytest and a custom
    shell wrapper gets both recognised.
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


# ── Outcome fingerprint ──────────────────────────────────────────────────

# Keywords that, when paired with a number nearby, signal a test
# outcome. The detector collects all (number, keyword) pairs from
# the output and uses their multiset as the fingerprint.
_OUTCOME_KEYWORDS: tuple[str, ...] = (
    "passed", "failed", "errors", "error",
    "passing", "failing", "skipped", "pending",
    "ok", "fail", "pass",
    "total",          # jest summary line
    "tests run", "failures",  # mvn / surefire
    "successes", "ignored",   # cargo test
)

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

    Returns a normalised string like ``"1 failed, 2 passed"`` or
    ``"3 passed"`` — identical outcomes on different runs produce the
    same string. Returns None when the content doesn't look like
    test output at all.
    """
    if not content:
        return None
    lower = content.lower()
    # Cheap pre-filter: skip clearly-not-test content.
    if not any(k in lower for k in ("passed", "failed", "test", "ok", "fail", "error")):
        return None

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
        # Last occurrence per keyword wins (runners often print
        # progress totals followed by the final value).
        pairs[kw_norm] = n

    if not pairs:
        return None
    keep_order = ("failed", "error", "passed", "skipped", "pending", "ignored", "total")
    parts = [f"{pairs[k]} {k}" for k in keep_order if k in pairs]
    if not parts:
        return None
    return ", ".join(parts)


__all__ = ["is_test_command", "test_outcome_fingerprint"]
