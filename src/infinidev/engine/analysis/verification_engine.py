"""Post-development verification engine.

Runs actual tests and import checks against changed files to produce
a PASS/FAIL verdict BEFORE the textual code review.  This catches
real breakage that a text-only reviewer would miss.
"""

from __future__ import annotations

import importlib.util
import keyword
import logging
import os
import shlex
import sys
from typing import Any

from infinidev.engine.analysis.verification_result import VerificationResult
from infinidev.engine.subprocess_runner import run_captured

logger = logging.getLogger(__name__)

_TIMEOUT = 120  # seconds per command


class VerificationEngine:
    """Run tests and import checks against changed files.

    Detects the project's test runner and executes it.  Falls back to
    import-checking changed Python files if no test runner is found.
    """

    def __init__(
        self,
        workspace: str | None = None,
        preferred_test_command: str | None = None,
    ) -> None:
        self._workspace = workspace or os.getcwd()
        self._preferred_test_command = (preferred_test_command or "").strip()

    def verify(
        self,
        changed_files: list[str] | None = None,
        file_tracker: Any = None,
    ) -> VerificationResult:
        """Run verification checks.

        Args:
            changed_files: List of changed file paths (absolute or relative).
            file_tracker: Optional FileChangeTracker with deleted symbol info.
                When provided and it contains removed symbols, runs an
                orphaned-references check to detect "disconnected code"
                (symbols that were removed but are still referenced).

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
                    result = self._run([sys.executable, "-c", f"import {module}"])
                    commands_run.append(result)
                    if result["exit_code"] != 0:
                        all_passed = False

        # 3. Orphaned references check: symbols removed but still referenced elsewhere
        if file_tracker is not None:
            orphaned = self._check_orphaned_references(file_tracker)
            if orphaned:
                commands_run.append({
                    "command": "(orphaned-references check)",
                    "exit_code": 1,
                    "output": self._format_orphaned_warning(orphaned),
                })
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

    def _check_orphaned_references(self, file_tracker: Any) -> list[dict[str, Any]]:
        """Check for symbols that were deleted but are still referenced elsewhere.

        Ensures each modified file is re-indexed first (so the deleted symbol
        is gone from ci_symbols) before querying references.
        """
        from infinidev.code_intel.analyzer import check_orphaned_references
        from infinidev.code_intel.smart_index import ensure_indexed

        deleted_by_file = file_tracker.get_deleted_symbols()
        if not deleted_by_file:
            return []

        project_id = 1  # matches the default used throughout the CLI

        for source_file in deleted_by_file.keys():
            try:
                ensure_indexed(project_id, source_file)
            except Exception as exc:
                logger.debug("Reindex before orphan check failed for %s: %s",
                             source_file, exc)

        try:
            diags = check_orphaned_references(project_id, deleted_by_file)
        except Exception as exc:
            logger.debug("Orphaned references check failed: %s", exc)
            return []

        return [
            {
                "file": d.file_path,
                "line": d.line,
                "message": d.message,
                "fix": d.fix_suggestion or "",
            }
            for d in diags
        ]

    def _format_orphaned_warning(self, orphaned: list[dict[str, Any]]) -> str:
        """Format orphaned reference warnings for developer output."""
        lines = ["ORPHANED REFERENCES (deleted symbols still in use):"]
        for item in orphaned:
            lines.append(f"  - {item['message']}")
            if item.get("fix"):
                lines.append(f"    Fix: {item['fix']}")
        return "\n".join(lines)

    def _detect_test_command(self) -> str | list[str] | None:
        """Detect the project's test runner."""
        ws = self._workspace

        # The developer already ran this exact command through the ordinary
        # permission boundary. Reusing it preserves project-specific env,
        # targets, wrappers, and flags; replacing it with a guessed full-suite
        # command can manufacture unrelated failures during review.
        if self._preferred_test_command:
            return self._preferred_test_command

        # pytest (Python)
        if os.path.isfile(os.path.join(ws, "pyproject.toml")) or \
           os.path.isfile(os.path.join(ws, "setup.py")) or \
           os.path.isdir(os.path.join(ws, "tests")):
            if importlib.util.find_spec("pytest") is not None:
                return [sys.executable, "-m", "pytest", "--tb=short", "-q"]

        # npm test (JavaScript/TypeScript)
        pkg_json = os.path.join(ws, "package.json")
        if os.path.isfile(pkg_json):
            try:
                import json
                with open(pkg_json) as f:
                    pkg = json.load(f)
                if "test" in pkg.get("scripts", {}):
                    return ["npm", "test"]
            except (json.JSONDecodeError, OSError):
                pass

        # cargo test (Rust)
        if os.path.isfile(os.path.join(ws, "Cargo.toml")):
            return ["cargo", "test"]

        # go test (Go)
        if os.path.isfile(os.path.join(ws, "go.mod")):
            return ["go", "test", "./..."]

        return None

    def _run(
        self,
        command: str | list[str],
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute an approved argv sequence and capture output."""
        timeout = timeout or _TIMEOUT
        is_shell_command = isinstance(command, str)
        display_command = command if is_shell_command else shlex.join(command)

        from infinidev.tools.shell.execute_command_tool import check_command_permission

        permission_error = check_command_permission(
            display_command,
            description="Run post-development verification command",
        )
        if permission_error:
            return {
                "command": display_command,
                "exit_code": -1,
                "output": permission_error,
            }

        try:
            proc = run_captured(
                command,
                cwd=self._workspace,
                timeout=timeout,
                shell=is_shell_command,
            )
            output = (proc.stdout + proc.stderr).strip()
            # Truncate very long output
            if len(output) > 3000:
                output = output[-3000:]
            if proc.timed_out:
                return {
                    "command": display_command,
                    "exit_code": -1,
                    "output": f"Command timed out after {timeout}s",
                }
            return {
                "command": display_command,
                "exit_code": proc.exit_code,
                "output": output,
            }
        except Exception as exc:
            return {
                "command": display_command,
                "exit_code": -1,
                "output": f"Error: {exc}",
            }

    def _file_to_module(self, filepath: str) -> str | None:
        """Convert a file path to a Python module name for import checking.

        Absolute paths are made relative to the workspace FIRST — otherwise
        ``/tmp/x/calc.py`` becomes the garbage module ``.tmp.x.calc`` and the
        ``python -c 'import ...'`` check fails spuriously (which triggered an
        unnecessary developer re-run). Files outside the workspace, and
        ``__init__``/``conftest``/``test_`` files, are skipped.
        """
        if not filepath.endswith(".py"):
            return None
        path = filepath
        # Normalise absolute paths to workspace-relative so the dotted module
        # is a real module, not a leading-dot path fragment.
        if os.path.isabs(path):
            try:
                path = os.path.relpath(path, self._workspace)
            except ValueError:
                return None  # e.g. a different drive on Windows
        # Outside the workspace (``../...``) → not importable from here.
        if path == ".." or path.startswith(".." + os.sep):
            return None
        # Skip __init__, test files, conftest
        basename = os.path.basename(filepath)
        if basename in ("__init__.py", "conftest.py") or basename.startswith("test_"):
            return None
        # Strip .py extension
        path = path[:-3]
        # Strip leading src/ if present
        if path.startswith("src" + os.sep) or path.startswith("src/"):
            path = path[4:]
        # Convert path separators to dots; strip any residual leading/trailing
        # dots defensively so we never emit ``.foo`` or ``foo.``.
        module = path.replace(os.sep, ".").replace("/", ".").strip(".")
        if not module or any(
            not part.isidentifier() or keyword.iskeyword(part)
            for part in module.split(".")
        ):
            return None
        return module or None
