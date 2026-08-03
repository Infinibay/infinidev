"""Regression tests for bounded, permission-aware test checkpoints."""

from __future__ import annotations

from unittest.mock import patch

from infinidev.engine.test_checkpoint import TestCheckpoint as Checkpoint


def test_permission_denial_prevents_test_command(tmp_path):
    checkpoint = Checkpoint("touch should-not-exist", str(tmp_path))

    with (
        patch(
            "infinidev.tools.shell.execute_command_tool.check_command_permission",
            return_value="Command denied by user",
        ),
        patch("infinidev.engine.test_checkpoint.run_captured") as run_captured,
    ):
        assert checkpoint.run() == (0, 0)

    run_captured.assert_not_called()
    assert not (tmp_path / "should-not-exist").exists()
    assert "denied" in checkpoint.last_output.lower()


def test_checkpoint_parses_captured_test_output(tmp_path, auto_approve_permissions):
    checkpoint = Checkpoint("printf '2 passed in 0.01s'", str(tmp_path))

    assert checkpoint.run() == (2, 2)
    assert checkpoint.baseline == 2
    assert checkpoint.high_water == 2
