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


class VerificationEngine:
    """Run tests and import checks against changed files.

    Detects the project's test runner and executes it.  Falls back to
    import-checking changed Python files if no test runner is found.
    """

    def __init__(self, workspace: str | None = None) -> None:
        self._workspace = workspace or os.getcwd()

    def verify(
        self,
        changed_files: list[str] | None = None,
    ) -> VerificationResult:
        """Run verification checks.

        Args:
            changed_files: List of changed file paths (absolute or relative).

        Returns:
            VerificationResult with pass/fail and command outputs.
        """
        commands_run: list[dict[str, Any]] = []
        all_passed = True

        # 1. Detect and run test suite
        test_cmd = self._detect_test_command()
        if test_cmd:
            result = self._run(test_cmd)
            commands_run.append(result)
            if result["exit_code"] != 0:
                all_passed = False

        # 2. Import-check changed Python files (catches syntax errors)
        py_files = [
            f for f in (changed_files or [])
            if f.endswith(".py") and os.path.isfile(
                os.path.join(self._workspace, f) if not os.path.isabs(f) else f
            )
        ]
        if py_files and not test_cmd:
            # Only do import checks if we didn't already run tests
            for py_file in py_files[:5]:  # Limit to 5 files
                module = self._file_to_module(py_file)
                if module:
                    result = self._run(f"python -c 'import {module}'")
                    commands_run.append(result)
                    if result["exit_code"] != 0:
                        all_passed = False

        if not commands_run:
            return VerificationResult(
                passed=True,
                summary="No verification commands to run",
                commands_run=[],
            )

        # Build summary
        total = len(commands_run)
        failed = sum(1 for c in commands_run if c["exit_code"] != 0)
        if all_passed:
            summary = f"All {total} verification command(s) passed"
        else:
            summary = f"{failed}/{total} verification command(s) failed"

        return VerificationResult(
            passed=all_passed,
            summary=summary,
            commands_run=commands_run,
        )

    def _detect_test_command(self) -> str | None:
        """Detect the project's test runner."""
        ws = self._workspace

        # pytest (Python)
        if os.path.isfile(os.path.join(ws, "pyproject.toml")) or \
           os.path.isfile(os.path.join(ws, "setup.py")) or \
           os.path.isdir(os.path.join(ws, "tests")):
            # Check if pytest is available
            check = self._run("python -m pytest --version", timeout=10)
            if check["exit_code"] == 0:
                return "python -m pytest --tb=short -q 2>&1 | tail -20"

        # npm test (JavaScript/TypeScript)
        pkg_json = os.path.join(ws, "package.json")
        if os.path.isfile(pkg_json):
            try:
                import json
                with open(pkg_json) as f:
                    pkg = json.load(f)
                if "test" in pkg.get("scripts", {}):
                    return "npm test 2>&1 | tail -20"
            except (json.JSONDecodeError, OSError):
                pass

        # cargo test (Rust)
        if os.path.isfile(os.path.join(ws, "Cargo.toml")):
            return "cargo test 2>&1 | tail -20"

        # go test (Go)
        if os.path.isfile(os.path.join(ws, "go.mod")):
            return "go test ./... 2>&1 | tail -20"

        return None

    def _run(self, command: str, timeout: int | None = None) -> dict[str, Any]:
        """Execute a shell command and capture output."""
        timeout = timeout or _TIMEOUT
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._workspace,
            )
            output = (proc.stdout + proc.stderr).strip()
            # Truncate very long output
            if len(output) > 3000:
                output = output[-3000:]
            return {
                "command": command,
                "exit_code": proc.returncode,
                "output": output,
            }
        except subprocess.TimeoutExpired:
            return {
                "command": command,
                "exit_code": -1,
                "output": f"Command timed out after {timeout}s",
            }
        except Exception as exc:
            return {
                "command": command,
                "exit_code": -1,
                "output": f"Error: {exc}",
            }

    def _file_to_module(self, filepath: str) -> str | None:
        """Convert a file path to a Python module name for import checking."""
        # Strip .py extension
        if not filepath.endswith(".py"):
            return None
        path = filepath[:-3]
        # Strip leading src/ if present
        if path.startswith("src/"):
            path = path[4:]
        # Convert path separators to dots
        module = path.replace(os.sep, ".").replace("/", ".")
        # Skip __init__, test files, conftest
        basename = os.path.basename(filepath)
        if basename in ("__init__.py", "conftest.py") or basename.startswith("test_"):
            return None
        return module
