"""Tool for executing shell commands in Infinidev CLI."""

import logging
import os
import shlex
import subprocess
from typing import Type
from pydantic import BaseModel, Field
from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool

logger = logging.getLogger(__name__)

class ExecuteCommandInput(BaseModel):
    command: str = Field(..., description="Command to execute")
    timeout: int = Field(
        default=60, ge=1, le=600, description="Max execution time in seconds"
    )
    cwd: str | None = Field(
        default=None, description="Working directory for the command"
    )
    env: dict[str, str] | None = Field(
        default=None, description="Additional environment variables"
    )

class ExecuteCommandTool(InfinibayBaseTool):
    name: str = "execute_command"
    description: str = (
        "Execute a shell command in the current environment. "
        "Returns stdout, stderr, and exit code."
    )
    args_schema: Type[BaseModel] = ExecuteCommandInput

    def _run(
        self,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        if not command or not command.strip():
            return self._error("Empty command")

        # Use shell=True to allow piping and other shell features,
        # since this is a local CLI for the user's own machine.
        run_env = os.environ.copy()
        if env:
            run_env.update(env)

        if not cwd:
            cwd = self.workspace_path or os.getcwd()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
                env=run_env,
            )
            
            return self._success({
                "exit_code": result.returncode,
                "stdout": result.stdout[-10000:] if result.stdout else "",
                "stderr": result.stderr[-5000:] if result.stderr else "",
                "success": result.returncode == 0,
            })
            
        except subprocess.TimeoutExpired:
            return self._error(f"Command timed out after {timeout}s")
        except Exception as e:
            return self._error(f"Execution failed: {e}")
