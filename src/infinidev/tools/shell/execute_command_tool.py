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
from infinidev.tools.shell.execute_command_input import ExecuteCommandInput


class ExecuteCommandTool(InfinibayBaseTool):
    name: str = "execute_command"
    description: str = (
        "Execute a shell command in the current environment. "
        "Returns stdout, stderr, and exit code."
    )
    args_schema: Type[BaseModel] = ExecuteCommandInput

    def _check_permission(self, command: str) -> str | None:
        """Check command execution permission. Returns error string or None if allowed."""
        mode = settings.EXECUTE_COMMANDS_PERMISSION

        if mode == "auto_approve":
            return None

        if mode == "allowed_list":
            allowed = settings.ALLOWED_COMMANDS_LIST
            if not allowed:
                return f"Command denied: no commands in allowed list"
            # Check if the command's base executable is in the allowed list
            try:
                base_cmd = shlex.split(command)[0]
            except ValueError:
                base_cmd = command.split()[0] if command.split() else command
            if base_cmd not in allowed and command not in allowed:
                return f"Command denied: '{base_cmd}' not in allowed list"
            return None

        if mode == "ask":
            from infinidev.tools.permission import request_permission
            approved = request_permission(
                tool_name="execute_command",
                description=f"Execute shell command",
                details=command,
            )
            if not approved:
                return f"Command denied by user: {command}"
            return None

        return None  # Unknown mode — allow

    def _run(
        self,
        command: str,
        timeout: int = 60,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
    ) -> str:
        if not isinstance(command, str):
            command = str(command) if command else ""
        if not command or not command.strip():
            return self._error("Empty command")

        # Check permissions
        perm_error = self._check_permission(command)
        if perm_error:
            return self._error(perm_error)

        # Use shell=True to allow piping and other shell features,
        # since this is a local CLI for the user's own machine.
        run_env = os.environ.copy()
        if env:
            # LLMs sometimes send non-string values (ints, bools) in env dicts.
            # subprocess requires all env values to be strings.
            if isinstance(env, dict):
                run_env.update({str(k): str(v) for k, v in env.items()})

        if not cwd or not isinstance(cwd, str):
            cwd = self.workspace_path or os.getcwd()

        effective_timeout = timeout if timeout > 0 else None

        try:
            from infinidev.engine.static_analysis_timer import measure as _sa_measure
            with _sa_measure("subprocess_exec"):
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=effective_timeout,
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

