"""Tests for the SWE-bench subprocess boundary."""

from pathlib import Path

from bench.run_swebench import _benchmark_environment


def test_disposable_benchmark_checkout_has_noninteractive_tool_authority(
    monkeypatch,
) -> None:
    monkeypatch.setenv("INFINIDEV_EXECUTE_COMMANDS_PERMISSION", "ask")
    monkeypatch.setenv("INFINIDEV_FILE_OPERATIONS_PERMISSION", "ask")
    monkeypatch.setenv("INFINIDEV_TOOL_EFFECTS_PERMISSION", "ask")

    env = _benchmark_environment(Path("/tmp/swe-instance"))

    assert env["INFINIDEV_WORKSPACE"] == "/tmp/swe-instance"
    assert env["INFINIDEV_EXECUTE_COMMANDS_PERMISSION"] == "auto_approve"
    assert env["INFINIDEV_FILE_OPERATIONS_PERMISSION"] == "auto_approve"
    assert env["INFINIDEV_TOOL_EFFECTS_PERMISSION"] == "auto_approve"
