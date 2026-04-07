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

    A subclass owns *everything* runner-specific in one place:

      * **runner_name**       — identifier used in ParsedFailure
      * **command_tokens**    — substrings that identify a CLI invocation
                                of this runner (e.g. ``("pytest", "py.test")``)
      * **flags_with_arg**    — flag names that consume the next token
                                (e.g. pytest ``-k EXPR``, go ``-run NAME``)
      * **flags_no_arg**      — boolean flags (e.g. ``-v``, ``--watch``)
      * **detect()**          — does this look like output from this runner?
      * **parse()**           — extract structured failures
      * **normalize_command**(default impl in this base class) — strip
        the runner's own flags from a CLI invocation so two runs of the
        same target set produce the same key.

    Adding a new runner is one new file with a TestParser subclass +
    one entry in ``test_parsers/__init__.py``'s ``_PARSERS`` tuple.
    No other module ever needs to know about its flags.

    Both ``detect()`` and ``parse()`` must never raise — failures are
    returned as ``False`` / ``[]``.
    """

    #: The runner identifier (lowercase, no spaces) — used as the
    #: ``runner`` field on every :class:`ParsedFailure` produced.
    runner_name: str = ""

    #: Substrings the runner CLI uses. Matched anywhere in the
    #: ``execute_command`` argument string. The first parser whose
    #: token appears in the command "owns" that command for the
    #: purpose of flag normalisation.
    command_tokens: tuple[str, ...] = ()

    #: Flags that consume the next positional token. Used by the
    #: command-normalisation pass: when one of these is seen, both
    #: the flag and the value following it are dropped. May appear
    #: as ``--flag value`` (whitespace-separated) or ``--flag=value``
    #: (joined with ``=``). Lowercase, no values inside.
    flags_with_arg: tuple[str, ...] = ()

    #: Boolean flags with no value. Just dropped during normalisation.
    flags_no_arg: tuple[str, ...] = ()

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

    def matches_command(self, args_str: str) -> bool:
        """Return True if *args_str* invokes this runner's CLI.

        Tokenises *args_str* on whitespace and checks whether any of
        the parser's ``command_tokens`` appears as a consecutive
        token subsequence. This avoids the substring trap where
        ``"cargo test"`` would naively contain the substring
        ``"go test"`` (because cargo ends with ``go``) and falsely
        match the Go parser.
        """
        if not args_str:
            return False
        haystack = args_str.lower().split()
        if not haystack:
            return False
        for token in self.command_tokens:
            needle = token.strip().split()
            if not needle:
                continue
            # Sliding window: does the needle appear as a contiguous
            # subsequence inside the haystack tokens?
            n = len(needle)
            for i in range(len(haystack) - n + 1):
                if haystack[i:i + n] == needle:
                    return True
        return False

    def normalize_command(self, cmd: str) -> str:
        """Strip this runner's flags from *cmd*, return the rest.

        Default implementation walks tokens, drops anything in
        ``flags_no_arg`` or ``flags_with_arg`` (along with the
        following value for the latter), drops shell noise (``2>&1``,
        pipes), and returns the lowercased result. Subclasses can
        override for runners with truly weird grammars but the
        default covers pytest/jest/cargo/go/mocha just fine.
        """
        if not cmd:
            return ""
        parts = cmd.strip().split()
        if not parts:
            return ""
        out: list[str] = []
        skip_next = False
        for tok in parts:
            if skip_next:
                skip_next = False
                continue
            low = tok.lower()
            if low in self.flags_no_arg:
                continue
            if any(low.startswith(f + "=") for f in self.flags_with_arg):
                continue
            if low in self.flags_with_arg:
                skip_next = True
                continue
            if tok in ("2>&1", "|", ">>", ">"):
                break
            out.append(tok)
        return " ".join(out).lower()


__all__ = ["ParsedFailure", "TestParser"]
