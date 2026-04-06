"""Base contracts for runner-specific test failure parsers.

The package follows a strict separation: each test runner has its
own subclass of :class:`TestParser` in a dedicated module. The
package's :func:`__init__.parse_test_failures` walks the registered
parsers, asks each one whether it recognises the content, and
delegates parsing to the first match.

Adding a new runner is one new file with a subclass + one entry in
``test_parsers/__init__.py``'s ``_PARSERS`` tuple. No other module
needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ParsedFailure:
    """One structured failure extracted from a test runner's output."""

    runner: str          # "pytest", "jest", "go", "cargo", "mocha", ...
    test_name: str       # e.g. "TestCreateTable.test_create_with_types"
    file: str = ""       # source file where the failure was raised
    line: int = 0        # line number (1-based), 0 if unknown
    error_type: str = "" # e.g. "KeyError", "AssertionError", "TypeError"
    message: str = ""    # short failure message (≤200 chars)

    def to_dict(self) -> dict:
        return asdict(self)


class TestParser(ABC):
    """Base class for one test runner.

    Subclasses must set :attr:`runner_name` and implement
    :meth:`detect` (cheap heuristic over the raw content) and
    :meth:`parse` (return the structured failures). Both methods
    must never raise — failures are returned as empty lists / False.
    """

    #: The runner identifier (lowercase, no spaces) — used as the
    #: ``runner`` field on every :class:`ParsedFailure` produced.
    runner_name: str = ""

    @abstractmethod
    def detect(self, content: str) -> bool:
        """Return True if *content* looks like output from this runner."""

    @abstractmethod
    def parse(self, content: str) -> list[ParsedFailure]:
        """Return the list of failures parsed from *content*.

        Returns an empty list when nothing failed or when the parser
        can't extract a useful structure. Must never raise — wrap
        risky regex work in try/except and return ``[]`` on error.
        """


__all__ = ["ParsedFailure", "TestParser"]
