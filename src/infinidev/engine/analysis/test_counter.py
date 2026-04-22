"""Language-agnostic test-case counter for review signals.

The reviewer LLM is unreliable at counting tests in a diff — it routinely
claims "added 3 tests" when 9 were added. This module gives the judge a
deterministic count to anchor its reasoning against.

We use per-language regexes rather than full parsers because:

- The signal is a count, not an AST analysis; a few false positives from
  e.g. a commented-out ``it(`` line are acceptable noise.
- Tree-sitter would force a language-detection hop on every test file,
  and we already do that elsewhere; keeping this module self-contained
  lets the reviewer call it without pulling in the code-intel stack.

If the count disagrees with what the developer's report claims, the
judge now has concrete grounds to flag a discrepancy.
"""

from __future__ import annotations

import os
import re
from typing import Iterable


# ─────────────────────────────────────────────────────────────────────────
# Language patterns
# ─────────────────────────────────────────────────────────────────────────
# Each entry: extension (lowercase, with dot) → compiled regex over the
# whole file in MULTILINE mode. We match the declaration style; describe
# blocks are NOT counted as tests (they're groupings) — we only count
# leaf cases so "3 describe blocks with 5 it each" reports as 15, the
# number the developer would see in a test report.

_JS_TS_TEST_RE = re.compile(
    r"^\s*(?:it|test)(?:\.\w+)?\s*\(",
    re.MULTILINE,
)

_PY_TEST_RE = re.compile(
    r"^\s*(?:async\s+)?def\s+test_\w+\s*\(",
    re.MULTILINE,
)

_RUST_TEST_RE = re.compile(
    r"^\s*#\[\s*test\s*\]",
    re.MULTILINE,
)

_GO_TEST_RE = re.compile(
    r"^\s*func\s+Test[A-Z]\w*\s*\(",
    re.MULTILINE,
)


_LANG_PATTERNS: dict[str, re.Pattern[str]] = {
    ".js":   _JS_TS_TEST_RE,
    ".jsx":  _JS_TS_TEST_RE,
    ".ts":   _JS_TS_TEST_RE,
    ".tsx":  _JS_TS_TEST_RE,
    ".mjs":  _JS_TS_TEST_RE,
    ".cjs":  _JS_TS_TEST_RE,
    ".py":   _PY_TEST_RE,
    ".rs":   _RUST_TEST_RE,
    ".go":   _GO_TEST_RE,
}


# Path fragments that indicate a file is part of a test suite. We are
# permissive here — better to count tests in an uncommon layout than to
# miss a test file because its path didn't match "tests/".
_TEST_PATH_FRAGMENTS: tuple[str, ...] = (
    "/tests/", "/test/", "/__tests__/", "/spec/",
    "tests/", "test/", "__tests__/", "spec/",
)

_TEST_NAME_FRAGMENTS: tuple[str, ...] = (
    ".test.", ".spec.", "_test.", "_spec.",
)


def looks_like_test_file(path: str) -> bool:
    """Heuristic: is ``path`` a test file we should count?

    Matches either a path prefix (``tests/foo.ts``) or a name suffix
    (``foo.test.ts``, ``bar_test.py``). Python's ``test_foo.py`` is
    matched via the ``test_`` filename prefix.
    """
    if not path:
        return False
    norm = path.replace("\\", "/")
    basename = os.path.basename(norm)

    if any(f in norm for f in _TEST_PATH_FRAGMENTS):
        return True
    if any(f in basename for f in _TEST_NAME_FRAGMENTS):
        return True
    if basename.startswith("test_") and basename.endswith(".py"):
        return True
    if basename.endswith("_test.go"):
        return True
    return False


def count_tests_in_content(path: str, content: str) -> int | None:
    """Count test cases in ``content`` using ``path`` to pick the regex.

    Returns ``None`` for files we don't know how to count (unsupported
    extension). Returns 0 for supported extensions with no test cases —
    that's a meaningful signal distinct from "can't count".
    """
    if not content:
        return 0 if _extension_supported(path) else None

    _, ext = os.path.splitext(path.lower())
    pattern = _LANG_PATTERNS.get(ext)
    if pattern is None:
        return None
    return sum(1 for _ in pattern.finditer(content))


def _extension_supported(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in _LANG_PATTERNS


def count_tests_for_files(
    files: Iterable[tuple[str, str | None, str | None]],
) -> dict[str, dict[str, int | None]]:
    """Compute per-file test counts before/after for a diff.

    ``files`` is an iterable of ``(path, before_content, after_content)``
    tuples. ``None`` content means "file didn't exist at that snapshot"
    and counts as 0 when the other side has content (for diff purposes).

    Only test files are included in the result — non-test files are
    skipped entirely rather than emitted with null counts, keeping the
    judge's automated-checks block small.
    """
    out: dict[str, dict[str, int | None]] = {}
    for path, before, after in files:
        if not looks_like_test_file(path):
            continue
        if not _extension_supported(path):
            continue
        before_n = count_tests_in_content(path, before or "")
        after_n = count_tests_in_content(path, after or "")
        out[path] = {
            "before": before_n if before_n is not None else 0,
            "after": after_n if after_n is not None else 0,
            "delta": (after_n or 0) - (before_n or 0),
        }
    return out
