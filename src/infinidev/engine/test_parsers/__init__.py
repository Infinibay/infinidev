"""Per-runner test failure parsers.

This package gives the rest of the codebase one job: take a test
runner's stdout, return a list of structured :class:`ParsedFailure`
objects describing each failed test (name, file, line, error type,
message). The :func:`parse_test_failures` entry point auto-detects
the runner from the content, or accepts a ``runner_hint``.

Each test runner has its own subclass of :class:`TestParser` in a
dedicated module:

  * :mod:`pytest_parser`     → :class:`PytestParser`     (Python)
  * :mod:`jest_parser`       → :class:`JestParser`       (JavaScript / TypeScript — handles jest + vitest)
  * :mod:`mocha_parser`      → :class:`MochaParser`      (JavaScript)
  * :mod:`node_test_parser`  → :class:`NodeTestParser`   (Node 20+ ``node:test``, TAP format)
  * :mod:`go_parser`         → :class:`GoTestParser`     (Go)
  * :mod:`cargo_parser`      → :class:`CargoTestParser`  (Rust)
  * :mod:`rspec_parser`      → :class:`RSpecParser`      (Ruby)

Adding a new runner is one new file with a TestParser subclass plus
one entry in the ``_PARSERS`` tuple below. No other module needs
changes — the dispatch loop here picks it up automatically.
"""

from __future__ import annotations

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser
from infinidev.engine.test_parsers.pytest_parser import PytestParser
from infinidev.engine.test_parsers.jest_parser import JestParser
from infinidev.engine.test_parsers.go_parser import GoTestParser
from infinidev.engine.test_parsers.cargo_parser import CargoTestParser
from infinidev.engine.test_parsers.mocha_parser import MochaParser
from infinidev.engine.test_parsers.rspec_parser import RSpecParser
from infinidev.engine.test_parsers.node_test_parser import NodeTestParser


# Order matters slightly: pytest is the most common and has the
# cheapest detect(), so it goes first to short-circuit. The rest are
# in arbitrary order — detect() functions are mutually exclusive
# enough that the order between non-pytest parsers doesn't change
# results in practice. node-test (TAP format) goes before mocha so
# the very-specific "TAP version" / "not ok" markers win over
# mocha's looser "N failing" detection.
_PARSERS: tuple[TestParser, ...] = (
    PytestParser(),
    JestParser(),
    NodeTestParser(),
    MochaParser(),
    GoTestParser(),
    CargoTestParser(),
    RSpecParser(),
)

# Lookup by runner name for callers who pass an explicit hint.
_BY_NAME: dict[str, TestParser] = {p.runner_name: p for p in _PARSERS}


def parse_test_failures(
    content: str,
    runner_hint: str | None = None,
) -> list[ParsedFailure]:
    """Parse a test runner's output into structured failure entries.

    When *runner_hint* is one of ``"pytest"``, ``"jest"``, ``"go"``,
    ``"cargo"``, ``"mocha"``, that parser is used directly. Otherwise
    the function asks each registered parser whether it recognises
    the content (via :meth:`TestParser.detect`) and uses the first
    match.

    Returns ``[]`` when nothing failed or when no parser recognises
    the format. Never raises — a parser failure returns ``[]``.
    """
    if not content:
        return []
    if runner_hint:
        parser = _BY_NAME.get(runner_hint.strip().lower())
        if parser is None:
            return []
        return parser.parse(content)
    for parser in _PARSERS:
        try:
            if parser.detect(content):
                return parser.parse(content)
        except Exception:
            continue
    return []


__all__ = [
    "ParsedFailure",
    "TestParser",
    "parse_test_failures",
    "PytestParser",
    "JestParser",
    "GoTestParser",
    "CargoTestParser",
    "MochaParser",
    "RSpecParser",
    "NodeTestParser",
]
