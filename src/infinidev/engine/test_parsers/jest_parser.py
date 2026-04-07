"""Parser for jest / vitest output (also handles plain JS/TS test runners
that emit the same ``● Suite › name`` failure header style).

Both jest and vitest produce structurally identical failure blocks::

    ● TestSuite › subtest name

      Expected: 5
      Received: 3

      14 |   it('adds negative numbers', () => {
    > 15 |     expect(add(-1, -1)).toBe(-2);
         |                          ^

      at Object.<anonymous> (src/calc.test.js:15:25)

The parser handles ``.js``, ``.ts``, ``.jsx``, ``.tsx``, ``.mjs``,
``.cjs`` source paths uniformly. The ``runner`` field on the
returned :class:`ParsedFailure` is set to ``"jest"`` regardless of
which one produced the output, since they share a format.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


_HEADER_RE = re.compile(r"^\s*●\s+(.+?)$", re.MULTILINE)
_EXPECTED_RE = re.compile(r"^\s*Expected:?\s*(.+)$", re.MULTILINE)
_RECEIVED_RE = re.compile(r"^\s*Received:?\s*(.+)$", re.MULTILINE)
_AT_LOCATION_RE = re.compile(r"at\s+\S+\s*\(([^:)]+):(\d+):\d+\)")
_FILE_LOCATION_RE = re.compile(
    r"^\s*at\s+(.+?\.(?:js|ts|jsx|tsx)):(\d+):\d+", re.MULTILINE
)


class JestParser(TestParser):
    runner_name = "jest"

    command_tokens = (
        "jest", "vitest",
        "npx jest", "npx vitest",
        "npm test", "npm run test", "yarn test", "pnpm test", "bun test",
    )

    flags_with_arg = (
        "--testnamepattern", "--testpathpattern", "-t",
        "--config", "--rootdir", "--projects",
        "--reporters", "--coverage-reporters",
        "--max-workers", "-c",
    )

    flags_no_arg = (
        "--watch", "--watchall", "--no-watch",
        "--coverage", "--no-coverage",
        "--verbose", "--silent",
        "--ci", "--bail",
        "--no-cache",
        "--listTests",
        "--passwithnotests",
    )

    def detect(self, content: str) -> bool:
        if not content:
            return False
        # jest / vitest "FAIL <path>.ts" header (any common JS/TS extension)
        if re.search(r"^\s*FAIL\s+\S+\.(?:js|ts|jsx|tsx|mjs|cjs)", content, re.MULTILINE):
            return True
        # vitest writes "Test Files" and "Tests" summary lines
        if "Test Files" in content and "Tests" in content and "●" in content:
            return True
        # Generic jest summary form
        if "Tests:" in content and ("passed" in content or "failed" in content) and "●" in content:
            return True
        return False

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        headers = list(_HEADER_RE.finditer(content))
        for i, header in enumerate(headers):
            name = header.group(1).strip()
            if not name or "console" in name.lower():
                continue
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            block = content[header.end():end]

            expected = _EXPECTED_RE.search(block)
            received = _RECEIVED_RE.search(block)
            loc = _AT_LOCATION_RE.search(block) or _FILE_LOCATION_RE.search(block)

            msg_parts: list[str] = []
            if expected:
                msg_parts.append(f"expected={expected.group(1).strip()}")
            if received:
                msg_parts.append(f"received={received.group(1).strip()}")
            message = ", ".join(msg_parts)[:200]

            file_ = loc.group(1) if loc else ""
            line_ = int(loc.group(2)) if loc else 0

            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=file_,
                line=line_,
                error_type="AssertionError" if expected and received else "",
                message=message,
            ))
        return failures


__all__ = ["JestParser"]
