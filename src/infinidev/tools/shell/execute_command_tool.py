"""Tool for executing shell commands in Infinidev CLI."""

import logging
import os
import re
import select
import shlex
import subprocess
import time
from typing import Type
from pydantic import BaseModel, Field
from infinidev.config.settings import settings
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.stdin_prompt import (
    has_stdin_input_handler,
    request_stdin_input,
)

logger = logging.getLogger(__name__)
from infinidev.tools.shell.execute_command_input import ExecuteCommandInput


# Patterns that indicate a subprocess is prompting for a stdin value.
# Matched against recent stderr output. Case-insensitive. The exact
# matched substring is passed to the handler as ``prompt_text`` so the
# user sees what the process asked for.
_PROMPT_PATTERNS: tuple[re.Pattern[bytes], ...] = (
    re.compile(rb"\[sudo\]\s+password\s+for\s+\S+\s*:", re.IGNORECASE),
    re.compile(rb"(?m)^\s*password\s*:\s*$", re.IGNORECASE),
    re.compile(rb"password\s+for\s+\S+\s*:", re.IGNORECASE),
    re.compile(rb"enter\s+passphrase\b[^\n]*:", re.IGNORECASE),
    re.compile(rb"passphrase\s*:\s*$", re.IGNORECASE | re.MULTILINE),
    re.compile(rb"(?m)^\s*username\s*:\s*$", re.IGNORECASE),
    re.compile(rb"(?m)^\s*login\s*:\s*$", re.IGNORECASE),
    re.compile(rb"enter\s+pin\b[^\n]*:", re.IGNORECASE),
    re.compile(rb"one[- ]?time\s+code\s*:", re.IGNORECASE),
    re.compile(rb"\(yes/no(/\[[^\]]+\])?\)\?\s*$", re.IGNORECASE),
)

# After a match we remember the substring so we don't re-fire on the
# same bytes still lingering in the stderr buffer.
_PROMPT_SCAN_WINDOW = 512  # bytes from the end of stderr to scan


def _detect_prompt(recent: bytes, already_seen: set[str]) -> str | None:
    """Return the prompt text if stderr looks like an input prompt."""
    for r in _PROMPT_PATTERNS:
        m = r.search(recent)
        if not m:
            continue
        text = m.group(0).decode(errors="replace").strip()
        if text not in already_seen:
            return text
    return None


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
        rationale: str = "",
    ) -> str:
        # rationale is enforced as required at the schema level (≥30
        # chars). Consumed by the critic via the tool_call args, not
        # by the executor — so we accept it here for kwarg parity and
        # discard it.
        del rationale
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
                if has_stdin_input_handler():
                    result = self._run_with_stdin_detection(
                        command, cwd, run_env, effective_timeout,
                    )
                else:
                    # No UI handler registered (classic mode / no TUI).
                    # Close stdin so interactive prompts fail fast
                    # instead of hijacking the parent terminal.
                    result = self._run_sealed(
                        command, cwd, run_env, effective_timeout,
                    )
            return self._success(result)

        except subprocess.TimeoutExpired:
            return self._error(f"Command timed out after {timeout}s")
        except Exception as e:
            return self._error(f"Execution failed: {e}")

    # ── Execution strategies ─────────────────────────────────────────

    def _run_sealed(
        self,
        command: str,
        cwd: str,
        run_env: dict[str, str],
        timeout: int | None,
    ) -> dict:
        """Run with stdin=DEVNULL so interactive prompts fail fast.

        Used when no UI stdin handler is registered. Commands like
        ``sudo`` exit with "no tty present" instead of hanging the
        parent terminal.
        """
        result = subprocess.run(
            command,
            shell=True,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
            env=run_env,
        )
        return {
            "exit_code": result.returncode,
            "stdout": (result.stdout or "")[-10000:],
            "stderr": (result.stderr or "")[-5000:],
            "success": result.returncode == 0,
        }

    def _run_with_stdin_detection(
        self,
        command: str,
        cwd: str,
        run_env: dict[str, str],
        timeout: int | None,
    ) -> dict:
        """Spawn with stdin=PIPE and watch stderr for prompt patterns.

        When a prompt is detected, ask the UI via
        ``request_stdin_input`` for the reply (or a kill signal),
        then write it to the child's stdin and continue streaming
        output. Loops until the process exits or the overall timeout
        is hit.
        """
        proc = subprocess.Popen(
            command,
            shell=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=run_env,
            bufsize=0,
        )

        stdout_buf = bytearray()
        stderr_buf = bytearray()
        seen_prompts: set[str] = set()
        # Only scan stderr bytes emitted AFTER the last handled prompt,
        # so overlapping regexes on the same prompt text don't re-fire.
        stderr_scan_from = 0
        start = time.monotonic()
        killed_reason: str | None = None

        try:
            while proc.poll() is None:
                fds = [f for f in (proc.stdout, proc.stderr) if f is not None]
                if not fds:
                    break
                try:
                    ready, _, _ = select.select(fds, [], [], 0.25)
                except (ValueError, OSError):
                    break

                for r in ready:
                    try:
                        data = os.read(r.fileno(), 4096)
                    except OSError:
                        data = b""
                    if not data:
                        continue
                    if r is proc.stdout:
                        stdout_buf.extend(data)
                    else:
                        stderr_buf.extend(data)

                # Check for prompt AFTER reading fresh stderr, since
                # the fragment may have landed just now.
                if len(stderr_buf) > stderr_scan_from:
                    fresh_start = max(
                        stderr_scan_from,
                        len(stderr_buf) - _PROMPT_SCAN_WINDOW,
                    )
                    window = bytes(stderr_buf[fresh_start:])
                    prompt = _detect_prompt(window, seen_prompts)
                    if prompt is not None:
                        seen_prompts.add(prompt)
                        reply = request_stdin_input(
                            command,
                            prompt,
                            stdout_buf.decode(errors="replace"),
                            stderr_buf.decode(errors="replace"),
                        )
                        # Advance the scan cursor so the same bytes
                        # don't re-fire a prompt, regardless of
                        # whether the user sent a reply or a kill.
                        stderr_scan_from = len(stderr_buf)
                        if reply is None:
                            killed_reason = (
                                f"User chose to kill the process at "
                                f"prompt: {prompt!r}"
                            )
                            break
                        try:
                            proc.stdin.write((reply + "\n").encode())
                            proc.stdin.flush()
                        except (BrokenPipeError, OSError) as exc:
                            logger.warning(
                                "Could not write stdin reply: %s", exc,
                            )

                # Overall timeout check.
                if timeout and (time.monotonic() - start) > timeout:
                    killed_reason = f"Command timed out after {timeout}s"
                    break

            if killed_reason is not None:
                self._terminate(proc)
                return {
                    "exit_code": -1,
                    "stdout": stdout_buf.decode(errors="replace")[-10000:],
                    "stderr": stderr_buf.decode(errors="replace")[-5000:],
                    "killed_reason": killed_reason,
                    "success": False,
                }

            # Drain any remaining output after the process exits.
            try:
                rest_out, rest_err = proc.communicate(timeout=2)
                if rest_out:
                    stdout_buf.extend(rest_out)
                if rest_err:
                    stderr_buf.extend(rest_err)
            except subprocess.TimeoutExpired:
                self._terminate(proc)

            return {
                "exit_code": proc.returncode if proc.returncode is not None else -1,
                "stdout": stdout_buf.decode(errors="replace")[-10000:],
                "stderr": stderr_buf.decode(errors="replace")[-5000:],
                "success": proc.returncode == 0,
            }
        finally:
            # Ensure file descriptors are closed in every branch.
            for f in (proc.stdin, proc.stdout, proc.stderr):
                try:
                    if f is not None:
                        f.close()
                except Exception:
                    pass

    @staticmethod
    def _terminate(proc: subprocess.Popen) -> None:
        """Terminate a subprocess gracefully, escalating to kill."""
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except Exception:
                pass
