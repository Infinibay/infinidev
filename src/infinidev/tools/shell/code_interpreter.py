"""Tool for executing Python code in a sandboxed interpreter."""

import logging
import os
import subprocess
import tempfile
from typing import Type

from pydantic import BaseModel, Field

from infinidev.config.settings import settings
from infinidev.security.container_runtime import runtime_available
from infinidev.security.sandbox import sandbox_executor
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)


class CodeInterpreterInput(BaseModel):
    code: str = Field(..., description="Python code to execute")
    libraries_used: list[str] | None = Field(
        default=None,
        description="List of libraries the code uses (informational only)",
    )
    timeout: int = Field(
        default=120,
        ge=1,
        le=600,
        description="Max execution time in seconds",
    )


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

    def _run(
        self,
        code: str,
        libraries_used: list[str] | None = None,
        timeout: int = 120,
    ) -> str:
        timeout = min(timeout, settings.CODE_INTERPRETER_TIMEOUT)
        max_output = settings.CODE_INTERPRETER_MAX_OUTPUT

        # ── Pod mode: pass code via stdin to python3 inside pod ──
        if self._is_pod_mode():
            return self._run_in_pod(code, timeout, max_output)

        # Write code to a temporary file
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                prefix="infinibay_code_",
                dir="/tmp",
                delete=False,
            ) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            # Execute via sandbox if available, else direct subprocess
            if settings.SANDBOX_ENABLED and runtime_available():
                result = self._run_sandboxed(tmp_path, timeout)
            else:
                result = self._run_direct(tmp_path, timeout)

            # Truncate output if needed
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

    def _run_in_pod(self, code: str, timeout: int, max_output: int) -> str:
        """Execute Python code inside the agent's pod via stdin."""
        try:
            result = self._exec_in_pod(
                ["python3", "-"],
                timeout=timeout,
                stdin_data=code,
            )
        except RuntimeError as e:
            return self._error(f"Pod execution failed: {e}")

        stdout = result.stdout or ""
        stderr = result.stderr or ""

        if result.timed_out:
            return self._success({
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout}s",
                "success": False,
            })

        if len(stdout) > max_output:
            stdout = stdout[:max_output] + "\n... [output truncated]"
        if len(stderr) > max_output:
            stderr = stderr[:max_output] + "\n... [output truncated]"

        self._log_tool_usage(
            f"Code interpreter (pod): {len(code)} chars, exit={result.exit_code}"
        )

        return self._success({
            "exit_code": result.exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "success": result.exit_code == 0,
        })

    def _run_sandboxed(self, script_path: str, timeout: int) -> dict:
        """Execute via container sandbox."""
        agent_id = self.agent_id or "unknown"
        role = self._get_agent_role(agent_id)

        result = sandbox_executor.execute(
            command=["python3", script_path],
            agent_id=agent_id,
            role=role,
            timeout=timeout,
        )

        if result.timed_out:
            return {
                "exit_code": -1,
                "stdout": "",
                "stderr": f"Code execution timed out after {timeout}s",
            }

        return {
            "exit_code": result.exit_code,
            "stdout": result.stdout or "",
            "stderr": result.stderr or "",
        }

    def _run_direct(self, script_path: str, timeout: int) -> dict:
        """Execute directly via subprocess (dev mode / fallback)."""
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

    @staticmethod
    def _get_agent_role(agent_id: str) -> str:
        """Look up the agent's role from the roster table."""
        import sqlite3
        from infinidev.tools.base.db import execute_with_retry

        role_result = {"role": "default"}

        def _query(conn: sqlite3.Connection):
            row = conn.execute(
                "SELECT role FROM roster WHERE agent_id = ?", (agent_id,)
            ).fetchone()
            if row:
                role_result["role"] = row[0]

        try:
            execute_with_retry(_query)
        except Exception:
            pass
        return role_result["role"]
