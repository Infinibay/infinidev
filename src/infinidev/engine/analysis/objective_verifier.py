"""ObjectiveVerifier — run a single step's StepVerification deterministically.

This is the per-step counterpart to ``VerificationEngine`` (which runs the
whole detected test suite post-loop). Given one ``StepVerification`` it
dispatches on ``kind`` and produces a ``VerificationResult`` whose
``passed`` flag is decided by an exit code / substring / grep hit — never
by an LLM reading the agent's own claim.

Kept separate from ``VerificationEngine`` so the per-objective check stays
small, side-effect-scoped, and unit-testable without exercising the whole
review-rework loop.
"""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
import sys

from infinidev.engine.analysis.step_verification import StepVerification
from infinidev.engine.analysis.verification_result import VerificationResult

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # seconds per command


class ObjectiveVerifier:
    """Execute a single step's verification check and return PASS/FAIL."""

    def __init__(self, workspace: str | None = None, timeout: int | None = None) -> None:
        self._workspace = workspace or os.getcwd()
        self._timeout = timeout or _TIMEOUT

    def verify(self, check: StepVerification) -> VerificationResult:
        """Run ``check`` and return a deterministic VerificationResult.

        A non-executable check (kind 'none' or empty spec) is treated as a
        PASS with no commands — the gate is expected to skip those before
        calling here, but this stays safe if it doesn't.
        """
        if not check.is_executable:
            return VerificationResult(
                passed=True,
                summary="No executable verification for this step",
                commands_run=[],
            )

        if check.kind == "command":
            return self._verify_command(check.spec, check.observable)
        if check.kind == "test_id":
            return self._verify_test_id(check.spec, check.observable)
        if check.kind == "file_contains":
            return self._verify_file_contains(check.spec, check.observable)
        if check.kind == "symbol_exists":
            return self._verify_symbol_exists(check.spec)

        # Unknown kind — fail closed but explain, so it surfaces rather than
        # silently passing.
        return VerificationResult(
            passed=False,
            summary=f"Unknown verification kind: {check.kind!r}",
            commands_run=[],
        )

    # ── per-kind handlers ────────────────────────────────────────────────

    def _verify_command(self, command: str, observable: str) -> VerificationResult:
        run = self._run(command)
        passed = run["exit_code"] == 0 and self._observable_ok(observable, run["output"])
        return self._result_from_run(run, passed, observable)

    def _interpreter(self) -> str:
        """The python that can import this workspace's test dependencies.

        A bare ``python`` is whatever the shell resolves, which outside a
        venv has no pytest and fails a check the code would have passed.
        The workspace's own venv wins — the target project may be neither
        Infinidev nor share its environment — and Infinidev's interpreter is
        the fallback, since that one at least exists.
        """
        for candidate in (".venv/bin/python", "venv/bin/python"):
            path = os.path.join(self._workspace, candidate)
            if os.path.exists(path):
                return shlex.quote(path)
        return shlex.quote(sys.executable)

    def _verify_test_id(self, node_id: str, observable: str) -> VerificationResult:
        # Treat the spec as a pytest node id (the dominant case). Quote it so
        # paths with '::' and special chars survive the shell.
        command = f"{self._interpreter()} -m pytest {shlex.quote(node_id)} -q"
        run = self._run(command)
        passed = run["exit_code"] == 0 and self._observable_ok(observable, run["output"])
        return self._result_from_run(run, passed, observable)

    def _verify_file_contains(self, path: str, needle: str) -> VerificationResult:
        abs_path = path if os.path.isabs(path) else os.path.join(self._workspace, path)
        # Spelled out rather than set-notation: this string reaches the
        # developer inside the BLOCKED message (verification_result.py:52).
        entry = {
            "command": f"(file_contains {path!r}: required text {needle!r})",
            "exit_code": 0,
            "output": "",
        }
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as fh:
                content = fh.read()
        except OSError as exc:
            entry["exit_code"] = 1
            entry["output"] = f"Could not read {path}: {exc}"
            return VerificationResult(passed=False, summary=f"file_contains: {path} unreadable", commands_run=[entry])
        present = needle in content
        entry["exit_code"] = 0 if present else 1
        entry["output"] = (
            f"Found required text in {path}" if present
            else f"Required text NOT found in {path}: {needle!r}"
        )
        return VerificationResult(
            passed=present,
            summary=("file_contains passed" if present else "file_contains failed"),
            commands_run=[entry],
        )

    def _verify_symbol_exists(self, symbol: str) -> VerificationResult:
        # Cheap, language-agnostic existence check via grep. -r recursive,
        # -I skip binaries, -F fixed string, -q quiet (exit code only),
        # -- end-of-options so a symbol starting with '-' is safe.
        command = f"grep -rIqF -- {shlex.quote(symbol)} ."
        run = self._run(command)
        passed = run["exit_code"] == 0
        run["output"] = (
            f"Symbol/text present in workspace: {symbol!r}" if passed
            else f"Symbol/text NOT found anywhere in workspace: {symbol!r}"
        )
        return VerificationResult(
            passed=passed,
            summary=("symbol_exists passed" if passed else "symbol_exists failed"),
            commands_run=[run],
        )

    # ── helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _observable_ok(observable: str, output: str) -> bool:
        """A required-output fragment, if given, must appear in the output."""
        observable = (observable or "").strip()
        return (not observable) or (observable in output)

    def _result_from_run(self, run: dict, passed: bool, observable: str) -> VerificationResult:
        if passed:
            summary = "verification passed"
        elif run["exit_code"] != 0:
            summary = f"command exited {run['exit_code']}"
        else:
            summary = f"required output not found: {observable!r}"
        return VerificationResult(passed=passed, summary=summary, commands_run=[run])

    def _run(self, command: str) -> dict:
        """Execute a shell command and capture truncated output."""
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self._timeout,
                cwd=self._workspace,
            )
            output = (proc.stdout + proc.stderr).strip()
            if len(output) > 3000:
                output = output[-3000:]
            return {"command": command, "exit_code": proc.returncode, "output": output}
        except subprocess.TimeoutExpired:
            return {"command": command, "exit_code": -1, "output": f"Command timed out after {self._timeout}s"}
        except Exception as exc:  # pragma: no cover - defensive
            return {"command": command, "exit_code": -1, "output": f"Error: {exc}"}
