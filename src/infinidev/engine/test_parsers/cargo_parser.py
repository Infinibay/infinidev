"""Parser for Rust ``cargo test`` output.

cargo test failures look like::

    failures:

    ---- tests::test_div stdout ----
    thread 'tests::test_div' panicked at 'attempt to divide by zero', src/lib.rs:42:5

    failures:
        tests::test_div

    test result: FAILED. 1 passed; 1 failed; 0 ignored

The most stable shape is the ``thread '...' panicked at ...`` line
plus its file:line suffix. Newer rustc puts the message on a
separate line so we accept both flavours.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


_PANIC_INLINE_RE = re.compile(
    r"thread\s+'([^']+)'\s+panicked\s+at\s+'?([^'\n,]+)'?,?\s*([^\s:]+):(\d+)",
)
_PANIC_MULTILINE_RE = re.compile(
    r"thread\s+'([^']+)'\s+panicked\s+at\s+([^\n]+)\n(?:[^\n]*\n)?\s*([^\s:]+):(\d+)",
)


class CargoTestParser(TestParser):
    runner_name = "cargo"

    def detect(self, content: str) -> bool:
        if not content:
            return False
        return "panicked at" in content or bool(re.search(r"test result:\s*FAILED", content))

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        seen: set[str] = set()
        for pattern in (_PANIC_INLINE_RE, _PANIC_MULTILINE_RE):
            for m in pattern.finditer(content):
                test_name = m.group(1)
                if test_name in seen:
                    continue
                seen.add(test_name)
                failures.append(ParsedFailure(
                    runner=self.runner_name,
                    test_name=test_name,
                    file=m.group(3),
                    line=int(m.group(4)),
                    error_type="panic",
                    message=m.group(2).strip()[:200],
                ))
        return failures


__all__ = ["CargoTestParser"]
