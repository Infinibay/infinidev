"""Tool for executing Python code in a sandboxed interpreter."""

import logging
import os
import subprocess
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)
from infinidev.tools.shell.code_interpreter_input import CodeInterpreterInput


class CodeInterpreterTool(InfinibayBaseTool):
    name: str = "code_interpreter"
    description: str = (
        "Execute Python code for data analysis, computation, validation, "
        "or prototyping. Code runs in a sandboxed subprocess and "
        "returns stdout, stderr, and exit code. For most queries about "
        "the codebase, prefer the dedicated tools (iter_symbols, "
        "find_references, find_similar_methods, search_symbols, etc.) — "
        "they are simpler than writing Python. Use code_interpreter "
        "only when you need a custom computation that no single tool "
        "can express, e.g. multi-step set arithmetic over query "
        "results. The parameter is named 'code' (NOT 'command' or "
        "'script')."
    )
    args_schema: Type[BaseModel] = CodeInterpreterInput

    def _check_permission(self, code: str) -> str | None:
        """Check code execution permission. Returns error string or None if allowed."""
        mode = settings.EXECUTE_COMMANDS_PERMISSION

        if mode == "auto_approve":
            return None

        if mode == "allowed_list":
            # Code interpreter is never in an allowed_list — deny
            return "Code interpreter denied: not in allowed commands list"

        if mode == "ask":
            from infinidev.tools.permission import request_permission
            # Show first 200 chars as description, full code as details
            preview = code[:200] + ("..." if len(code) > 200 else "")
            approved = request_permission(
                tool_name="code_interpreter",
                description=f"Execute Python code ({len(code)} chars): {preview}",
                details=code,
            )
            if not approved:
                return "Code execution denied by user"
            return None

        return None  # Unknown mode — allow

    def _run(
        self,
        code: str,
        libraries_used: list[str] | None = None,
        timeout: int = 120,
    ) -> str:
        # Check permissions first
        perm_error = self._check_permission(code)
        if perm_error:
            return self._error(perm_error)

        timeout = min(timeout, settings.CODE_INTERPRETER_TIMEOUT)
        max_output = settings.CODE_INTERPRETER_MAX_OUTPUT

        # Write code to a temporary file. We prepend a small bootstrap
        # header that pre-imports the read-only code-intelligence API
        # into the script's global namespace so the model can call
        # ``find_references(...)``, ``find_symbols(...)``, etc.
        # directly without knowing the import path. If the user's
        # script already imports these names, Python's wildcard import
        # rules mean the user imports shadow ours — no conflict.
        #
        # The ``try/except ImportError`` guards against the rare case
        # where the sandboxed subprocess can't reach the infinidev
        # package (misconfigured venv). In that case the bootstrap is
        # a silent no-op and the model's script runs without the
        # bridge, same as before this commit.
        # The bootstrap header is prepended to every script the model
        # submits. It imports the read-only bridge functions into the
        # global namespace AND prints a visible proof-of-life marker
        # to STDOUT as its first line. The marker serves two purposes:
        #
        #   1. Model confidence: on the last experiment we saw the
        #      model ignore the bootstrap and DEFINE iter_symbols
        #      itself — because it had no empirical evidence that the
        #      import worked. With the marker, the model sees in its
        #      own tool result "[infinidev bridge: 13 functions live]"
        #      before any of its own output, so it knows the functions
        #      are real and ready to call.
        #   2. Operator debugging: if a future environment breaks the
        #      import (misconfigured venv, missing dep), the marker
        #      switches to "[infinidev bridge: FAILED — <exc>]" and
        #      the failure is loud instead of silent.
        #
        # Why stdout and not stderr: models attend most to stdout
        # (that's where their own print() output goes), so placing
        # the marker there guarantees it's on the main stream the
        # model reads first. A clear [BRIDGE: ...] prefix keeps it
        # visually distinct from the script's own data.
        _BOOTSTRAP_HEADER = (
            "# ── infinidev code_interpreter bootstrap ─────────────────\n"
            "# The 13 functions listed below are ALREADY DEFINED in\n"
            "# your script's global namespace. Call them directly — do\n"
            "# NOT import them, do NOT redefine them, do NOT wrap them.\n"
            "# They read the project_id and workspace from env vars set\n"
            "# by the tool; you never need to pass them.\n"
            "try:\n"
            "    from infinidev.code_intel.interpreter_api import (\n"
            "        find_symbols, find_definitions, find_references,\n"
            "        list_file_symbols, iter_symbols, get_source,\n"
            "        find_similar, search_by_intent, extract_skeleton,\n"
            "        list_files, find_files, code_search, project_stats,\n"
            "    )\n"
            "    print(\n"
            "        '[BRIDGE: 13 functions live — project_stats, '\n"
            "        'iter_symbols, find_symbols, find_definitions, '\n"
            "        'find_references, list_file_symbols, get_source, '\n"
            "        'find_similar, search_by_intent, extract_skeleton, '\n"
            "        'list_files, find_files, code_search. '\n"
            "        'CALL THEM DIRECTLY — they are already in globals, '\n"
            "        'do NOT define them yourself]'\n"
            "    )\n"
            "except ImportError as _bridge_err:\n"
            "    print(\n"
            "        f'[BRIDGE: FAILED to import — {_bridge_err}. '\n"
            "        f'Code-intelligence queries unavailable this script.]'\n"
            "    )\n"
            "# ── end bootstrap ─────────────────────────────────────────\n"
        )

        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="infinidev_code_",
                dir="/tmp",
                delete=False,
            ) as tmp:
                tmp.write(_BOOTSTRAP_HEADER)
                tmp.write(code)
                tmp_path = tmp.name

            result = self._run_direct(tmp_path, timeout)

            stdout = result["stdout"]
            stderr = result["stderr"]
            if len(stdout) > max_output:
                stdout = stdout[:max_output] + "\n... [output truncated]"
            if len(stderr) > max_output:
                stderr = stderr[:max_output] + "\n... [output truncated]"

            self._log_tool_usage(
                f"Code interpreter: {len(code)} chars, exit={result['exit_code']}"
            )

            return self._success({
                "exit_code": result["exit_code"],
                "stdout": stdout,
                "stderr": stderr,
                "success": result["exit_code"] == 0,
            })

        except Exception as e:
            return self._error(f"Code execution failed: {e}")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _run_direct(self, script_path: str, timeout: int) -> dict:
        """Execute directly via subprocess.

        Uses ``sys.executable`` — the exact Python interpreter running
        infinidev — instead of a bare ``python3``. This guarantees the
        subprocess has the same site-packages as the main process, so
        the bootstrap header's ``from infinidev.code_intel.interpreter_api
        import ...`` actually works. Without this, a system ``python3``
        would be missing the infinidev package entirely.

        Also injects two env vars for the read-only API shim:

          * ``INFINIDEV_PROJECT_ID`` — the active project_id resolved
            from the tool context (defaults to 1).
          * ``INFINIDEV_WORKSPACE_PATH`` — the active workspace path
            resolved from the tool context (defaults to cwd).

        These give the bridge functions in ``interpreter_api.py``
        their ambient scope without the model having to pass them.
        """
        import sys
        env = os.environ.copy()
        # Resolve project context from the tool's bound context if
        # available; fall back to sensible defaults otherwise.
        try:
            pid = self.project_id
            if pid:
                env["INFINIDEV_PROJECT_ID"] = str(pid)
        except Exception:
            pass
        try:
            wp = self.workspace_path
            if wp:
                env["INFINIDEV_WORKSPACE_PATH"] = str(wp)
        except Exception:
            pass

        try:
            result = subprocess.run(
                [sys.executable, script_path],
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout}s",
            }

        return {
            "exit_code": result.returncode,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }

