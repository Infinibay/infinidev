"""Parser for ``go test`` output.

go test failures look like::

    === RUN   TestAdd
    --- FAIL: TestAdd (0.00s)
        calc_test.go:12: expected 5, got 3
    === RUN   TestSub
    --- PASS: TestSub (0.00s)
    FAIL
    exit status 1
    FAIL    github.com/x/calc       0.123s

Each ``--- FAIL: TestName`` block is followed by indented file:line
messages until the next test header or section boundary.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


_FAIL_HEADER_RE = re.compile(r"^---\s+FAIL:\s+(\S+)", re.MULTILINE)
_FILE_LINE_RE = re.compile(r"^\s*([\w./-]+\.go):(\d+):\s*(.+)$", re.MULTILINE)
_NEXT_HEADER_RE = re.compile(
    r"^(?:---\s+(?:FAIL|PASS|SKIP):|===\s+RUN|FAIL\b|PASS\b|ok\b|exit\s+status)",
    re.MULTILINE,
)


class GoTestParser(TestParser):
    runner_name = "go"

    def detect(self, content: str) -> bool:
        return bool(content) and "--- FAIL:" in content

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        for header in _FAIL_HEADER_RE.finditer(content):
            name = header.group(1)
            tail_start = header.end()
            next_match = _NEXT_HEADER_RE.search(content, tail_start)
            body_end = next_match.start() if next_match else len(content)
            body = content[tail_start:body_end]

            loc = _FILE_LINE_RE.search(body)
            if loc:
                failures.append(ParsedFailure(
                    runner=self.runner_name,
                    test_name=name,
                    file=loc.group(1),
                    line=int(loc.group(2)),
                    message=loc.group(3).strip()[:200],
                ))
            else:
                failures.append(ParsedFailure(
                    runner=self.runner_name,
                    test_name=name,
                    message=body.strip()[:200],
                ))
        return failures


__all__ = ["GoTestParser"]
