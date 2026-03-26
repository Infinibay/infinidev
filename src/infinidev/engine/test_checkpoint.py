"""Test checkpoint system for tracking test progress during execution.

Runs tests after code changes, tracks pass/fail counts, detects regressions,
and provides progress strings for injection into step prompts.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess

logger = logging.getLogger(__name__)


class TestCheckpoint:
    """Track test progress across execution steps."""

    def __init__(self, test_command: str | None = None, workdir: str | None = None):
        self.command = test_command or self._detect_test_command(workdir)
        self.workdir = workdir or os.getcwd()
        self.baseline: int = 0
        self.current: int = 0
        self.previous: int = 0
        self.high_water: int = 0
        self.total: int = 0
        self.last_output: str = ""
        self._initialized = False

    def _detect_test_command(self, workdir: str | None) -> str:
        """Auto-detect the test command based on project files."""
        wd = workdir or os.getcwd()

        # Python: pytest
        if any(
            os.path.exists(os.path.join(wd, f))
            for f in ("pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini")
        ):
            return "python -m pytest --tb=no -q"

        # Node.js: npm test
        if os.path.exists(os.path.join(wd, "package.json")):
            return "npm test -- --silent 2>&1"

        # Makefile with test target
        makefile = os.path.join(wd, "Makefile")
        if os.path.exists(makefile):
            try:
                with open(makefile) as f:
                    if "test:" in f.read():
                        return "make test"
            except Exception:
                pass

        # Default: try pytest
        return "python -m pytest --tb=no -q"

    def run(self) -> tuple[int, int]:
        """Run tests and return (passed, total).

        Updates internal counters: current, previous, high_water, total.
        """
        if not self.command:
            return (0, 0)

        try:
            result = subprocess.run(
                self.command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            output = result.stdout + "\n" + result.stderr
            self.last_output = output

            passed, total = self._parse_output(output)

            self.previous = self.current
            self.current = passed
            self.total = total
            if passed > self.high_water:
                self.high_water = passed

            if not self._initialized:
                self.baseline = passed
                self._initialized = True

            return (passed, total)

        except subprocess.TimeoutExpired:
            logger.warning("Test command timed out after 120s")
            self.last_output = "TIMEOUT"
            return (self.current, self.total)
        except Exception as exc:
            logger.warning("Failed to run tests: %s", str(exc)[:200])
            self.last_output = str(exc)
            return (self.current, self.total)

    def _parse_output(self, output: str) -> tuple[int, int]:
        """Parse test runner output to extract pass/total counts.

        Supports pytest and jest output formats.
        """
        # pytest: "15 passed, 46 failed in 0.3s" or "15 passed in 0.3s"
        pytest_match = re.search(
            r"(\d+)\s+passed(?:.*?(\d+)\s+failed)?(?:.*?(\d+)\s+error)?",
            output,
        )
        if pytest_match:
            passed = int(pytest_match.group(1))
            failed = int(pytest_match.group(2) or 0)
            errors = int(pytest_match.group(3) or 0)
            total = passed + failed + errors
            return (passed, total)

        # pytest short: "15 passed" on its own line
        short_match = re.search(r"^(\d+) passed", output, re.MULTILINE)
        if short_match:
            passed = int(short_match.group(1))
            # Look for total in collection line: "collected 61 items"
            collected = re.search(r"collected (\d+) items?", output)
            total = int(collected.group(1)) if collected else passed
            return (passed, total)

        # jest: "Tests: 5 passed, 3 failed, 8 total"
        jest_match = re.search(
            r"Tests:\s+(?:(\d+)\s+passed)?.*?(?:(\d+)\s+failed)?.*?(\d+)\s+total",
            output,
        )
        if jest_match:
            passed = int(jest_match.group(1) or 0)
            total = int(jest_match.group(3))
            return (passed, total)

        # Fallback: count PASSED/FAILED lines
        passed_lines = len(re.findall(r"PASSED|✓|✔", output))
        failed_lines = len(re.findall(r"FAILED|✗|✘|FAIL", output))
        if passed_lines or failed_lines:
            return (passed_lines, passed_lines + failed_lines)

        return (0, 0)

    def has_regression(self) -> bool:
        """True if current pass count dropped below previous."""
        return self.current < self.previous

    def progress_str(self) -> str:
        """Human-readable progress string for injection into prompts.

        Examples:
          "Tests: 15/61 passing (↑5 from last step)"
          "Tests: 10/61 passing (↓3 REGRESSION from last step)"
          "Tests: 15/61 passing (first run)"
        """
        if self.total == 0:
            return "Tests: not yet run"

        delta = self.current - self.previous
        if not self._initialized or self.previous == 0 and self.current == self.baseline:
            delta_str = "(baseline)"
        elif delta > 0:
            delta_str = f"(↑{delta} from last step)"
        elif delta < 0:
            delta_str = f"(↓{abs(delta)} REGRESSION from last step)"
        else:
            delta_str = "(unchanged)"

        return f"Tests: {self.current}/{self.total} passing {delta_str}"

    def regression_warning(self) -> str:
        """Warning message to inject when regression detected. Empty if no regression."""
        if not self.has_regression():
            return ""
        return (
            f"⚠ REGRESSION DETECTED: Tests dropped from {self.previous} to {self.current} "
            f"passing (high water mark: {self.high_water}). "
            f"Your last change broke {self.previous - self.current} test(s). "
            f"Consider reverting your last edit or fixing the regression before continuing."
        )
