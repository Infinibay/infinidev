"""Parser for Node.js built-in test runner output (TAP format).

The Node 20+ ``node --test`` and ``node:test`` API emits TAP::

    TAP version 13
    # Subtest: hello world
    not ok 1 - hello world
      ---
      duration_ms: 0.5
      failureType: 'testCodeFailure'
      error: 'Expected 1 to equal 2'
      code: 'ERR_ASSERTION'
      stack: |-
          TestContext.<anonymous> (test.js:3:5)
      ...
    1..1
    # tests 1
    # fail 1

Each ``not ok N - <name>`` line starts a YAML-ish block with
``error``, ``code`` and ``stack`` fields. The stack tells us the
file:line. We pull all three out per failure.
"""

from __future__ import annotations

import re

from infinidev.engine.test_parsers.base import ParsedFailure, TestParser


# "not ok 1 - hello world"  or  "not ok 1 - hello world # SKIP reason"
_NOT_OK_RE = re.compile(r"^\s*not ok\s+\d+\s+-\s+(.+?)(?:\s+#.*)?$", re.MULTILINE)
# YAML-block fields (everything after the "---" indent until next "...")
_ERROR_FIELD_RE = re.compile(r"^\s*error:\s*['\"]?(.+?)['\"]?$", re.MULTILINE)
_CODE_FIELD_RE = re.compile(r"^\s*code:\s*['\"]?(\w+)['\"]?$", re.MULTILINE)
# Stack lines like "    at TestContext.<anonymous> (test.js:3:5)"
_STACK_LOCATION_RE = re.compile(
    r"\(([^:)]+\.(?:js|mjs|cjs|ts)):(\d+):\d+\)"
)
_STACK_LOCATION_BARE_RE = re.compile(
    r"^\s*at\s+(\S+\.(?:js|mjs|cjs|ts)):(\d+):\d+", re.MULTILINE
)


class NodeTestParser(TestParser):
    runner_name = "node-test"

    command_tokens = ("node --test", "node:test")

    flags_with_arg = (
        "--test-name-pattern",
        "--test-reporter",
        "--test-reporter-destination",
        "--test-concurrency",
        "--test-timeout",
    )

    flags_no_arg = (
        "--test-only",
        "--test-skip-pattern",
        "--watch",
    )

    def detect(self, content: str) -> bool:
        if not content:
            return False
        # Specific to TAP-format node:test output
        if "TAP version" in content and "not ok" in content:
            return True
        # node:test summary marker without TAP header (some versions)
        if re.search(r"^#\s+(?:tests|fail|pass)\b", content, re.MULTILINE) and "not ok" in content:
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
        headers = list(_NOT_OK_RE.finditer(content))
        for i, header in enumerate(headers):
            name = header.group(1).strip()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            block = content[header.end():end]

            err_field = _ERROR_FIELD_RE.search(block)
            code_field = _CODE_FIELD_RE.search(block)
            stack_loc = (
                _STACK_LOCATION_RE.search(block)
                or _STACK_LOCATION_BARE_RE.search(block)
            )

            file_ = stack_loc.group(1) if stack_loc else ""
            line_ = int(stack_loc.group(2)) if stack_loc else 0

            err_type = code_field.group(1) if code_field else ""
            # Normalise the most common assertion code so it reads like
            # the other parsers.
            if err_type == "ERR_ASSERTION":
                err_type = "AssertionError"

            message = err_field.group(1).strip() if err_field else ""
            failures.append(ParsedFailure(
                runner=self.runner_name,
                test_name=name,
                file=file_,
                line=line_,
                error_type=err_type,
                message=message[:200],
            ))
        return failures


__all__ = ["NodeTestParser"]
