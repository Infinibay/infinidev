"""Parser for pytest output.

Strategy:
  1. The bottom ``FAILED <test_id> - <ErrorType>: <message>`` summary
     lines are the most stable shape, so we extract them first.
  2. We then walk the per-test traceback blocks (``___ test_name ___``
     headers) and pull the ``file.py:lineno`` reference out of each
     to enrich the summary entries with location info.
  3. Fallback: if no summary lines exist, we extract directly from
     the traceback blocks.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


# Header style for the per-test traceback block:
#     _________________________ test_create_with_types ___________________________
_HEADER_RE = re.compile(r"^_+\s+([\w:.\[\]<>-]+)\s+_+\s*$", re.MULTILINE)
# E lines have leading spaces, "E   ", then the error.
_E_LINE_RE = re.compile(r"^E\s{2,}(\w+(?:Error|Exception)?)?\s*:?\s*(.+)$", re.MULTILINE)
# file:line where the failure happened.
_FILE_LINE_RE = re.compile(r"^(\S+\.py):(\d+):", re.MULTILINE)
# Bottom summary line: FAILED tests/test_minidb.py::TestX::test_y - KeyError: 'id'
_SUMMARY_RE = re.compile(r"^FAILED\s+(\S+::\S+)\s*-\s*(.+)$", re.MULTILINE)


class PytestParser(TestParser):
    runner_name = "pytest"

    def detect(self, content: str) -> bool:
        if not content:
            return False
        return "test session starts" in content or _SUMMARY_RE.search(content) is not None

    def parse(self, content: str) -> list[ParsedFailure]:
        try:
            return self._parse_unsafe(content)
        except Exception:
            return []

    # ── private ──────────────────────────────────────────────────────

    def _parse_unsafe(self, content: str) -> list[ParsedFailure]:
        # 1. Summary lines first (one per failed test).
        failures: list[ParsedFailure] = []
        seen: set[str] = set()
        for m in _SUMMARY_RE.finditer(content):
            test_id = m.group(1)
            if test_id in seen:
                continue
            seen.add(test_id)
            err_type, message = self._split_error_message(m.group(2).strip())
            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=test_id,
                error_type=err_type,
                message=message[:200],
            ))

        # 2. Enrich with file:line from the traceback blocks above.
        if failures:
            block_locations = self._collect_block_locations(content)
            return [self._enrich(f, block_locations) for f in failures]

        # 3. Fallback: extract directly from traceback blocks.
        return self._parse_blocks(content)

    @staticmethod
    def _split_error_message(text: str) -> tuple[str, str]:
        """Split 'KeyError: id' into ('KeyError', "'id'")."""
        if ":" not in text:
            return ("", text)
        head, _, tail = text.partition(":")
        head = head.strip()
        if head and head[0].isupper() and " " not in head:
            return (head, tail.strip())
        return ("", text)

    @staticmethod
    def _collect_block_locations(content: str) -> dict[str, tuple[str, int]]:
        """Map each per-test header name to the first ``file.py:lineno``
        reference inside its block."""
        locations: dict[str, tuple[str, int]] = {}
        for header in _HEADER_RE.finditer(content):
            name = header.group(1)
            tail = content[header.end():header.end() + 4000]
            file_m = _FILE_LINE_RE.search(tail)
            if file_m:
                locations[name] = (file_m.group(1), int(file_m.group(2)))
        return locations

    def _enrich(self, f: ParsedFailure, locations: dict[str, tuple[str, int]]) -> ParsedFailure:
        short_name = f.test_name.split("::")[-1].split("[")[0]
        if short_name not in locations:
            return f
        file_, line_ = locations[short_name]
        return ParsedFailure(
            runner=f.runner,
            test_name=f.test_name,
            file=file_,
            line=line_,
            error_type=f.error_type,
            message=f.message,
        )

    def _parse_blocks(self, content: str) -> list[ParsedFailure]:
        failures: list[ParsedFailure] = []
        for header in _HEADER_RE.finditer(content):
            name = header.group(1)
            tail = content[header.end():header.end() + 4000]
            e_match = _E_LINE_RE.search(tail)
            file_match = _FILE_LINE_RE.search(tail)
            err_type = e_match.group(1) if e_match and e_match.group(1) else ""
            message = e_match.group(2).strip() if e_match else ""
            file_ = file_match.group(1) if file_match else ""
            line_ = int(file_match.group(2)) if file_match else 0
            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=file_,
                line=line_,
                error_type=err_type,
                message=message[:200],
            ))
        return failures


__all__ = ["PytestParser"]
