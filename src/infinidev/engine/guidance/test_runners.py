"""Multi-language test-runner detection and outcome fingerprinting.

Three primitives used by the stuck-pattern detectors:

  * :func:`is_test_command` — was this ``execute_command`` invocation
    a test runner call? Built-in coverage comes from each
    :class:`~infinidev.engine.test_parsers.base.TestParser` subclass:
    every parser declares its own ``command_tokens``, so adding a new
    runner is one new file in ``test_parsers/`` and this module never
    needs to know. Two user-overridable sources contribute additively:
    1. ``LOOP_CUSTOM_TEST_COMMANDS`` setting — user-declared, persistent
    2. ``state.custom_test_commands`` — agent-declared via the
       ``declare_test_command`` tool, scoped to the current task

  * :func:`test_outcome_fingerprint` — extract a stable, normalised
    string from a runner's stdout (e.g. ``"1 failed, 2 passed"``) so
    different runners producing the same outcome get the same
    fingerprint and identical outcomes can be compared.

  * :func:`normalize_test_command` — strip the runner's cosmetic flags
    while keeping the positional targets, so two runs of the same test
    set produce the same key. Delegates to the parser whose
    ``command_tokens`` matches — every parser knows its own flag table.

All three are intentionally permissive (high recall, acceptable
precision). False positives in the fingerprint are harmless because
the downstream detectors require additional structural conditions
before they fire.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from infinidev.engine.test_parsers import _PARSERS

if TYPE_CHECKING:
    from infinidev.engine.loop.models import LoopState
    from infinidev.engine.test_parsers.base import TestParser


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
    if not args_str:
        return False
    s = args_str.lower()
    # 1. Built-in runners — each TestParser subclass owns its tokens.
    for parser in _PARSERS:
        if parser.matches_command(args_str):
            return True
    # 2. User-declared via setting.
    for token in _user_test_command_tokens():
        if token in s:
            return True
    # 3. Agent-declared via tool.
    for token in _runtime_test_command_tokens(state):
        if token in s:
            return True
    return False


def _parser_for_command(cmd: str) -> "TestParser | None":
    """Return the registered TestParser whose command_tokens match *cmd*.

    Used by ``normalize_test_command`` to delegate flag stripping to
    the right runner. Returns the FIRST match — parser order in the
    ``test_parsers`` registry decides ties.
    """
    if not cmd:
        return None
    for parser in _PARSERS:
        if parser.matches_command(cmd):
            return parser
    return None


def normalize_test_command(cmd: str) -> str:
    """Reduce a test command to its 'what is being tested' essence.

    Two commands that test the same set of things produce the same
    normalised key::

      pytest test_x.py::test_a -v        → "pytest test_x.py::test_a"
      pytest test_x.py::test_a --tb=long → "pytest test_x.py::test_a"
      cd /tmp && pytest test_x.py::test_a → "pytest test_x.py::test_a"

    Two commands that test DIFFERENT things stay distinct::

      pytest test_x.py → "pytest test_x.py"
      pytest test_y.py → "pytest test_y.py"

    Flag-stripping logic lives on each runner's TestParser subclass —
    this function just identifies which parser owns the command via
    ``command_tokens`` and delegates. Falls back to the lowercased
    command (with the cd prelude stripped) when no registered parser
    matches.
    """
    if not cmd:
        return ""
    s = cmd.strip()
    # Drop a leading "cd ... &&" prelude — the working directory
    # doesn't change which tests run.
    if s.lower().startswith("cd ") and "&&" in s:
        s = s.split("&&", 1)[1].strip()

    parser = _parser_for_command(s)
    if parser is not None:
        return parser.normalize_command(s)
    # Unknown runner (probably custom test command from setting/tool):
    # just lowercase + collapse whitespace, no flag stripping.
    return " ".join(s.split()).lower()


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


__all__ = ["is_test_command", "test_outcome_fingerprint", "normalize_test_command"]
