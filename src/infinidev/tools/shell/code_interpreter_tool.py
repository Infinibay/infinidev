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
        "or prototyping. The code runs in a sandboxed environment. "
        "Returns stdout, stderr, and exit code. Use this for tasks that "
        "require computation — data processing, statistical analysis, "
        "generating charts, validating hypotheses with code, etc."
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

        # Write code to a temporary file
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="infinidev_code_",
                dir="/tmp",
                delete=False,
            ) as tmp:
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
        """Execute directly via subprocess."""
        try:
            result = subprocess.run(
                ["python3", script_path],
                shell=False,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=os.environ.copy(),
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

