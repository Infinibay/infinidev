"""Tests for ExecuteCommandTool."""

import json
import signal
import sys
from unittest.mock import MagicMock, patch

import pytest

from infinidev.config.settings import settings
from infinidev.engine.test_environment import prepare_test_environment
from infinidev.tools.shell.execute_command import ExecuteCommandTool
from infinidev.tools.shell.execute_command_input import ExecuteCommandInput
from infinidev.tools.shell.execute_command_tool import (
    _effective_command_timeout,
)


class TestExecuteCommand:
    """Tests for shell command execution."""

    def test_execute_simple_command(self, bound_tool, auto_approve_permissions):
        """echo hello returns stdout."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="echo hello")
        data = json.loads(result)
        assert data["success"] is True
        assert "hello" in data["stdout"]

    def test_schema_explains_that_cwd_is_per_command(self):
        description = ExecuteCommandInput.model_fields["cwd"].description
        assert description is not None
        assert "prior shell `cd` does not persist" in description

    def test_short_specific_rationale_is_valid(self):
        parsed = ExecuteCommandInput(command="pytest -q", rationale="Verify focused regression")

        assert parsed.rationale == "Verify focused regression"

    def test_missing_rationale_is_derived_from_the_command(self):
        parsed = ExecuteCommandInput(command="pytest tests/test_auth.py -q")

        assert "pytest tests/test_auth.py -q" in parsed.rationale
        assert "observable result" in parsed.rationale

    def test_description_explains_that_shell_state_does_not_persist(self):
        description = ExecuteCommandTool.model_fields["description"].default
        assert "cd from an earlier call does not persist" in description

    def test_execute_empty_command(self, bound_tool, auto_approve_permissions):
        """Empty string returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="")
        data = json.loads(result)
        assert "error" in data
        assert "empty" in data["error"].lower()

    def test_execute_whitespace_only(self, bound_tool, auto_approve_permissions):
        """Whitespace-only command returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="   ")
        data = json.loads(result)
        assert "error" in data

    def test_execute_nonzero_exit_code(self, bound_tool, auto_approve_permissions):
        """exit 1 returns success=false with exit_code=1."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="exit 1")
        data = json.loads(result)
        assert data["success"] is False
        assert data["exit_code"] == 1

    def test_pipeline_reports_failure_from_an_earlier_stage(
        self, bound_tool, auto_approve_permissions
    ):
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="sh -c 'exit 7' | tail -1")
        data = json.loads(result)

        assert data["success"] is False
        assert data["exit_code"] == 7

    def test_successful_pipeline_still_returns_its_output(
        self, bound_tool, auto_approve_permissions
    ):
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="printf 'hello\\n' | tail -1")
        data = json.loads(result)

        assert data["success"] is True
        assert data["exit_code"] == 0
        assert data["stdout"] == "hello\n"

    def test_execute_timeout(self, bound_tool, auto_approve_permissions):
        """Command that exceeds timeout returns error."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="sleep 10", timeout=1)
        data = json.loads(result)
        assert "error" in data
        assert "timed out" in data["error"].lower()

    def test_configured_timeout_is_a_hard_ceiling(self, monkeypatch):
        monkeypatch.setattr(settings, "COMMAND_TIMEOUT", 7)

        assert _effective_command_timeout(None) == 7
        assert _effective_command_timeout(0) == 7
        assert _effective_command_timeout(-1) == 7
        assert _effective_command_timeout(3) == 3
        assert _effective_command_timeout(300) == 7

    def test_terminate_signals_the_entire_process_group(self):
        proc = MagicMock()
        proc.pid = 4321
        proc.wait.return_value = 0

        with patch(
            "infinidev.tools.shell.execute_command_tool.os.killpg",
        ) as killpg:
            ExecuteCommandTool._terminate(proc)

        assert killpg.call_args_list == [
            ((4321, signal.SIGTERM),),
            ((4321, 0),),
            ((4321, signal.SIGKILL),),
        ]

    def test_execute_stdout_truncation(self, bound_tool, auto_approve_permissions):
        """Output longer than 10K is truncated to last 10K chars."""
        tool = bound_tool(ExecuteCommandTool)
        # Generate >10K of output
        result = tool._run(command="python3 -c \"print('x' * 15000)\"")
        data = json.loads(result)
        assert len(data["stdout"]) <= 10001  # 10K + possible newline

    def test_capture_disabled_preserves_exact_truncated_shape(
        self, bound_tool, auto_approve_permissions, monkeypatch
    ):
        tool = bound_tool(ExecuteCommandTool)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", False)
        stdout = "head-" + "x" * 12_000
        stderr = "error-" + "y" * 6_000

        data = tool._capture_before_truncation(
            exit_code=9, stdout=stdout, stderr=stderr, success=False
        )

        assert data == {
            "exit_code": 9,
            "stdout": stdout[-10000:],
            "stderr": stderr[-5000:],
            "success": False,
        }

    def test_sealed_capture_reconstructs_exact_precut_text(
        self, bound_tool, auto_approve_permissions, monkeypatch, tmp_path
    ):
        tool = bound_tool(ExecuteCommandTool)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
        stdout = "START-SECRET\n" + "π" * 10_100
        command = (
            "python3 -c \"import sys; "
            f"sys.stdout.write({stdout!r})\""
        )

        data = json.loads(tool._run(command=command))

        assert data["stdout"] == stdout[-10000:]
        raw = data["command_output_handles"]["stdout"]
        from infinidev.engine.command_output_store import (
            CommandOutputHandle,
            CommandOutputStore,
        )
        handle = CommandOutputHandle(
            artifact_id=raw["artifact_id"],
            artifact_type=raw["type"],
            stream=raw["stream"],
            char_count=raw["char_count"],
            byte_count=raw["byte_count"],
        )
        assert CommandOutputStore().read_text(
            handle, project_id=tool.project_id, session_id=tool.session_id
        ) == stdout

    def test_capture_failure_keeps_legacy_result_and_announces_no_handle(
        self, bound_tool, auto_approve_permissions, monkeypatch
    ):
        tool = bound_tool(ExecuteCommandTool)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_MAX_ARTIFACT_BYTES", 3)
        stdout = "x" * 10_001

        data = tool._capture_before_truncation(
            exit_code=0, stdout=stdout, stderr="", success=True
        )

        assert data == {
            "exit_code": 0,
            "stdout": stdout[-10000:],
            "stderr": "",
            "success": True,
        }

    def test_invalid_capture_limits_soft_disable_without_handle(
        self, bound_tool, auto_approve_permissions, monkeypatch
    ):
        tool = bound_tool(ExecuteCommandTool)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_CAPTURE_ENABLED", True)
        monkeypatch.setattr(settings, "COMMAND_OUTPUT_STORE_TIMEOUT_SECONDS", 0)
        stdout = "x" * 10_001

        data = tool._capture_before_truncation(
            exit_code=0, stdout=stdout, stderr="", success=True
        )

        assert "command_output_handles" not in data
        assert data["stdout"] == stdout[-10000:]

    def test_execute_custom_cwd(self, bound_tool, auto_approve_permissions, workspace_dir):
        """cwd parameter is respected."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="pwd", cwd=str(workspace_dir))
        data = json.loads(result)
        assert str(workspace_dir) in data["stdout"]

    def test_execute_custom_env(self, bound_tool, auto_approve_permissions):
        """env parameter adds environment variables."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(
            command="echo $MY_TEST_VAR",
            env={"MY_TEST_VAR": "test_value_123"},
        )
        data = json.loads(result)
        assert "test_value_123" in data["stdout"]

    def test_pytest_uses_checked_out_src_package_without_installing(
        self, bound_tool, auto_approve_permissions, tmp_path
    ):
        package = tmp_path / "src" / "demo_package"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("VALUE = 'checkout'\n")
        (tmp_path / "test_demo.py").write_text(
            "import demo_package\n\n"
            "def test_checkout_import():\n"
            "    assert demo_package.VALUE == 'checkout'\n"
        )

        result = json.loads(bound_tool(ExecuteCommandTool)._run(
            command="python -m pytest test_demo.py -q",
            cwd=str(tmp_path),
        ))

        assert result["exit_code"] == 0
        assert "1 passed" in result["stdout"]
        assert "Prepended" in result["environment_adjustment"]

    def test_explicit_pythonpath_is_never_overridden(self, tmp_path):
        (tmp_path / "src" / "demo").mkdir(parents=True)
        (tmp_path / "src" / "demo" / "__init__.py").write_text("")
        run_env = {"PYTHONPATH": "inherited"}

        run_env, adjustment = prepare_test_environment(
            "python -m pytest -q",
            str(tmp_path),
            explicit_env={"PYTHONPATH": "explicit"},
            base_env=run_env,
        )

        assert adjustment is None
        assert run_env["PYTHONPATH"] == "explicit"

    def test_absolute_python_pytest_command_gets_src_checkout(self, tmp_path):
        (tmp_path / "src" / "demo").mkdir(parents=True)
        (tmp_path / "src" / "demo" / "__init__.py").write_text("")

        run_env, adjustment = prepare_test_environment(
            [sys.executable, "-m", "pytest", "-q"],
            str(tmp_path),
            base_env={},
        )

        assert adjustment is not None
        assert run_env["PYTHONPATH"] == str(tmp_path / "src")

    def test_permission_auto_approve(self, bound_tool):
        """auto_approve mode allows any command."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "auto_approve"
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo allowed")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_permission_allowed_list_allows(self, bound_tool):
        """Command in allowed list runs."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = ["echo", "ls"]
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo ok")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_allowed_list_blocks(self, bound_tool):
        """Command not in allowed list is denied."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = ["echo"]
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="rm -rf /")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_allowed_list_empty(self, bound_tool):
        """Empty allowed list denies everything."""
        orig_mode = settings.EXECUTE_COMMANDS_PERMISSION
        orig_list = settings.ALLOWED_COMMANDS_LIST
        settings.EXECUTE_COMMANDS_PERMISSION = "allowed_list"
        settings.ALLOWED_COMMANDS_LIST = []
        try:
            tool = bound_tool(ExecuteCommandTool)
            result = tool._run(command="echo blocked")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig_mode
            settings.ALLOWED_COMMANDS_LIST = orig_list

    def test_permission_ask_approved(self, bound_tool):
        """When ask mode and permission granted, command runs."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "ask"
        try:
            tool = bound_tool(ExecuteCommandTool)
            with patch("infinidev.tools.permission.request_permission", return_value=True):
                result = tool._run(command="echo approved")
            data = json.loads(result)
            assert data["success"] is True
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_permission_ask_denied(self, bound_tool):
        """When ask mode and permission denied, command blocked."""
        orig = settings.EXECUTE_COMMANDS_PERMISSION
        settings.EXECUTE_COMMANDS_PERMISSION = "ask"
        try:
            tool = bound_tool(ExecuteCommandTool)
            with patch("infinidev.tools.permission.request_permission", return_value=False):
                result = tool._run(command="echo denied")
            data = json.loads(result)
            assert "error" in data
            assert "denied" in data["error"].lower()
        finally:
            settings.EXECUTE_COMMANDS_PERMISSION = orig

    def test_cwd_defaults_to_workspace(self, bound_tool, auto_approve_permissions):
        """When no cwd given, uses workspace_path."""
        tool = bound_tool(ExecuteCommandTool)
        result = tool._run(command="pwd")
        data = json.loads(result)
        ws = tool.workspace_path
        assert ws in data["stdout"]


class TestManualBackgroundingDetection:
    """The executor must bounce manual backgrounding to run_in_background."""

    @pytest.mark.parametrize("cmd", [
        "python server.py & echo PID=$!",
        "npm run dev &",
        "sleep 100 &",
        "nohup ./run.sh &",
        "setsid mytask",
        "node app.js & disown",
        "./server & echo started",
        "python -m http.server 8000 &",
    ])
    def test_backgrounding_is_rejected(self, cmd, bound_tool, auto_approve_permissions):
        tool = bound_tool(ExecuteCommandTool)
        data = json.loads(tool._run(command=cmd))
        assert "error" in data
        assert "run_in_background" in data["error"]

    @pytest.mark.parametrize("cmd", [
        "ls -la",
        "grep foo bar.txt && echo done",
        "make build 2>&1 | tee log",
        'echo "a & b"',
        'curl -s "http://x/?a=1&b=2"',
        "cat file > out 2>&1",
        "echo done >&2",
        "git log --oneline && git status",
    ])
    def test_legitimate_commands_not_flagged(self, cmd):
        from infinidev.tools.shell.execute_command_tool import (
            detect_manual_backgrounding,
        )
        assert detect_manual_backgrounding(cmd) is None
