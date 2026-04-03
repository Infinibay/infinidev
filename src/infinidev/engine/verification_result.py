"""Post-development verification engine.

Runs actual tests and import checks against changed files to produce
a PASS/FAIL verdict BEFORE the textual code review.  This catches
real breakage that a text-only reviewer would miss.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # seconds per command


@dataclass
class VerificationResult:
    """Result of the verification phase."""

    passed: bool
    summary: str = ""
    commands_run: list[dict[str, str]] = field(default_factory=list)
    # Each entry: {"command": "...", "exit_code": int, "output": "..."}

    @property
    def verdict(self) -> str:
        return "PASS" if self.passed else "FAIL"

    def format_for_developer(self) -> str:
        """Format failures as feedback for a rework iteration."""
        if self.passed:
            return ""
        parts = [
            "## Verification FAILED",
            "",
            f"**Summary:** {self.summary}",
            "",
        ]
        for cmd in self.commands_run:
            if cmd.get("exit_code", 0) != 0:
                parts.append(f"### `{cmd['command']}` (exit {cmd['exit_code']})")
                output = cmd.get("output", "")
                if output:
                    # Truncate long output
                    lines = output.splitlines()
                    if len(lines) > 30:
                        output = "\n".join(lines[-30:])
                    parts.append(f"```\n{output}\n```")
                parts.append("")
        return "\n".join(parts)


