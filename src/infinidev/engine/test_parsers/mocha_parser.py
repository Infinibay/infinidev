"""Parser for mocha output.

mocha failures look like::

      1) Suite name
          subtest name:
            AssertionError: expected 5 to equal 3
            at Context.<anonymous> (test/foo.spec.js:12:14)

      1 passing
      1 failing
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


_NUMBERED_RE = re.compile(r"^\s*\d+\)\s+(.+?)$", re.MULTILINE)
_AT_LOCATION_RE = re.compile(
    r"at\s+\S+\s*\(([^:)]+):(\d+):\d+\)|at\s+(\S+\.(?:js|ts)):(\d+):\d+",
)
_ERROR_RE = re.compile(r"^\s*([A-Z]\w*Error)\s*:\s*(.+)$", re.MULTILINE)


class MochaParser(TestParser):
    runner_name = "mocha"

    command_tokens = ("mocha", "npx mocha")

    flags_with_arg = (
        "-g", "--grep",
        "-f", "--fgrep",
        "-r", "--require",
        "--reporter", "-R",
        "--timeout", "-t",
        "--retries",
        "--config",
    )

    flags_no_arg = (
        "--watch", "--no-watch",
        "--bail", "--no-bail",
        "--inspect", "--inspect-brk",
        "--exit", "--no-exit",
        "--invert", "--check-leaks",
    )

    def detect(self, content: str) -> bool:
        return bool(content) and bool(re.search(r"^\s*\d+\s+failing", content, re.MULTILINE))

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        headers = list(_NUMBERED_RE.finditer(content))
        for i, header in enumerate(headers):
            name = header.group(1).strip()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            block = content[header.end():end]

            err_m = _ERROR_RE.search(block)
            loc = _AT_LOCATION_RE.search(block)

            file_ = ""
            line_ = 0
            if loc:
                file_ = loc.group(1) or loc.group(3) or ""
                line_str = loc.group(2) or loc.group(4) or "0"
                try:
                    line_ = int(line_str)
                except ValueError:
                    line_ = 0

            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=file_,
                line=line_,
                error_type=err_m.group(1) if err_m else "",
                message=(err_m.group(2).strip()[:200] if err_m else ""),
            ))
        return failures


__all__ = ["MochaParser"]
