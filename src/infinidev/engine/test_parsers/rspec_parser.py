"""Parser for Ruby RSpec output.

RSpec failures look like::

    Failures:

      1) Calculator add adds two numbers
         Failure/Error: expect(calc.add(1, 1)).to eq(3)

           expected: 3
                got: 2

           (compared using ==)
         # ./spec/calculator_spec.rb:6:in `block (3 levels) in <top (required)>'

    Finished in 0.00234 seconds (files took 0.06212 seconds to load)
    1 example, 1 failure

    Failed examples:

    rspec ./spec/calculator_spec.rb:5 # Calculator add adds two numbers

The most stable shape is the bottom ``rspec <file>:<line> # <name>``
list, so we extract that first and enrich each entry with the
``Failure/Error:`` line and ``expected/got`` values from the body.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


# Bottom summary line: "rspec ./spec/calculator_spec.rb:5 # Calculator add adds two numbers"
_SUMMARY_RE = re.compile(
    r"^rspec\s+(\S+):(\d+)\s+#\s+(.+?)$",
    re.MULTILINE,
)
# Numbered failure header: "  1) Calculator add adds two numbers"
_HEADER_RE = re.compile(r"^\s*\d+\)\s+(.+?)$", re.MULTILINE)
# Failure/Error line tells us which assertion blew up.
_FAILURE_ERROR_RE = re.compile(r"^\s*Failure/Error:\s*(.+)$", re.MULTILINE)
# expected/got values (with various indentations)
_EXPECTED_RE = re.compile(r"^\s*expected:?\s*(.+?)$", re.MULTILINE | re.IGNORECASE)
_GOT_RE = re.compile(r"^\s*got:?\s*(.+?)$", re.MULTILINE | re.IGNORECASE)
# Source line: "# ./spec/foo_spec.rb:6:in `block ...'"
_SOURCE_LINE_RE = re.compile(r"^\s*#\s*(\S+\.rb):(\d+)", re.MULTILINE)
# Stable error type names that show up in RSpec output
_ERROR_TYPE_RE = re.compile(
    r"\b(RSpec::Expectations::ExpectationNotMetError|"
    r"NoMethodError|NameError|ArgumentError|TypeError|RuntimeError|StandardError)\b"
)


class RSpecParser(TestParser):
    runner_name = "rspec"

    command_tokens = ("rspec", "rake test", "minitest", "bundle exec rspec")

    flags_with_arg = (
        "-e", "--example",
        "-t", "--tag",
        "-f", "--format",
        "-o", "--out",
        "-p", "--profile",
        "--seed", "--order",
        "-r", "--require",
        "-c", "--color",
    )

    flags_no_arg = (
        "--fail-fast", "--no-fail-fast",
        "--dry-run",
        "--bisect",
        "--backtrace", "-b",
        "--color",
        "--no-profile",
    )

    def detect(self, content: str) -> bool:
        if not content:
            return False
        # The most reliable markers across RSpec versions:
        if re.search(r"^\s*\d+\s+examples?\b.*\bfailures?\b", content, re.MULTILINE):
            return True
        if "Failed examples:" in content and "rspec " in content:
            return True
        if "Failure/Error:" in content:
            return True
        return False

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        # Strategy 1: parse the bottom "rspec <file>:<line> # <name>" list
        # because it's the most stable shape and has file+line built in.
        summary_failures = self._parse_summary(content)
        if summary_failures:
            return self._enrich_with_blocks(summary_failures, content)

        # Strategy 2: fall back to numbered "1) name" blocks.
        return self._parse_blocks(content)

    def _parse_summary(self, content: str) -> list[ParsedFailure]:
        out: list[ParsedFailure] = []
        seen: set[str] = set()
        for m in _SUMMARY_RE.finditer(content):
            file_, line_str, name = m.group(1), m.group(2), m.group(3).strip()
            key = f"{file_}:{line_str}::{name}"
            if key in seen:
                continue
            seen.add(key)
            out.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=file_,
                line=int(line_str),
            ))
        return out

    def _enrich_with_blocks(
        self,
        summary: list[ParsedFailure],
        content: str,
    ) -> list[ParsedFailure]:
        # For each summary entry, find the matching numbered block above
        # and pull error_type + message out of it.
        enriched: list[ParsedFailure] = []
        for f in summary:
            block = self._find_block_for(f.test_name, content)
            if not block:
                enriched.append(f)
                continue
            err_match = _ERROR_TYPE_RE.search(block)
            failure_match = _FAILURE_ERROR_RE.search(block)
            expected = _EXPECTED_RE.search(block)
            got = _GOT_RE.search(block)

            err_type = err_match.group(1) if err_match else ""
            if err_type == "RSpec::Expectations::ExpectationNotMetError":
                err_type = "ExpectationNotMet"

            msg_parts: list[str] = []
            if failure_match:
                msg_parts.append(failure_match.group(1).strip())
            if expected:
                msg_parts.append(f"expected={expected.group(1).strip()}")
            if got:
                msg_parts.append(f"got={got.group(1).strip()}")
            message = " | ".join(msg_parts)[:200]

            enriched.append(ParsedFailure(
                runner=f.runner,
                test_name=f.test_name,
                file=f.file,
                line=f.line,
                error_type=err_type,
                message=message,
            ))
        return enriched

    def _find_block_for(self, name: str, content: str) -> str:
        """Return the body of the numbered failure block for *name*, or ''."""
        headers = list(_HEADER_RE.finditer(content))
        for i, h in enumerate(headers):
            if h.group(1).strip() == name:
                end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
                return content[h.end():end]
        return ""

    def _parse_blocks(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        headers = list(_HEADER_RE.finditer(content))
        for i, h in enumerate(headers):
            name = h.group(1).strip()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            block = content[h.end():end]

            err_match = _ERROR_TYPE_RE.search(block)
            failure_match = _FAILURE_ERROR_RE.search(block)
            source = _SOURCE_LINE_RE.search(block)
            expected = _EXPECTED_RE.search(block)
            got = _GOT_RE.search(block)

            err_type = err_match.group(1) if err_match else ""
            if err_type == "RSpec::Expectations::ExpectationNotMetError":
                err_type = "ExpectationNotMet"

            msg_parts: list[str] = []
            if failure_match:
                msg_parts.append(failure_match.group(1).strip())
            if expected:
                msg_parts.append(f"expected={expected.group(1).strip()}")
            if got:
                msg_parts.append(f"got={got.group(1).strip()}")

            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=source.group(1) if source else "",
                line=int(source.group(2)) if source else 0,
                error_type=err_type,
                message=" | ".join(msg_parts)[:200],
            ))
        return failures


__all__ = ["RSpecParser"]
