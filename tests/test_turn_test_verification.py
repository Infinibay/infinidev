"""Workspace evidence for checks performed in the principal turn."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from infinidev.config.settings import settings
from infinidev.engine.turns import runtime as turn_runtime
from infinidev.tools.base.base_tool import BaseTool
from infinidev.tools.file.edit_file_tool import EditFileTool
from tests.test_turn_runtime import reply, runtime, tc


class CommandInput(BaseModel):
    command: str


@pytest.mark.parametrize("scenario,expected", [
    ("current", "current"),
    ("stale_same_batch", "stale"),
    ("rechecked", "current"),
    ("failed", "unconfirmed"),
    ("conflicting", "unconfirmed"),
    ("no_snapshot", "unconfirmed"),
    ("cancelled", "unconfirmed"),
    ("late_change", "stale"),
    ("disabled", None),
])
def test_test_receipt_tracks_the_observed_workspace(
    scenario, expected, runtime, monkeypatch, workspace_dir,
    auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    monkeypatch.setattr(settings, "TASK_TEST_REVISION_FEEDBACK_ENABLED", scenario != "disabled")
    path = workspace_dir / "checked.py"
    path.write_text("value = 1\n")
    commands = []

    class CommandTool(BaseTool):
        name: str = "execute_command"
        description: str = "Return the scripted check result."
        args_schema: type[BaseModel] = CommandInput

        def _run(self, command: str) -> str:
            commands.append(command)
            if scenario == "cancelled":
                engine.cancel()
            payload = {"exit_code": 1 if scenario == "failed" else 0, "stdout": "check output"}
            if scenario == "conflicting":
                payload["test_output_conflict"] = True
            return json.dumps(payload)

    snapshot_count = 0
    revision = turn_runtime.FileChangeTracker.workspace_revision

    def observe_snapshot(tracker):
        nonlocal snapshot_count
        snapshot_count += 1
        return None if scenario == "no_snapshot" else revision(tracker)

    monkeypatch.setattr(turn_runtime.FileChangeTracker, "workspace_revision", observe_snapshot)
    if scenario == "stale_same_batch":
        monkeypatch.setattr(
            turn_runtime, "get_model_capabilities",
            lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False),
        )

    agent.tools = [CommandTool(), EditFileTool()]
    check = tc("execute_command", command="pytest -q")
    edit = tc("edit_file", file_path=str(path), old_string="value = 1", new_string="value = 2")
    finish = reply("", tc("finish_turn", message="Done."))
    edit_first_batch = scenario in {
        "stale_same_batch", "rechecked", "failed", "conflicting", "no_snapshot", "disabled",
    }
    calls = [reply("", check, edit) if edit_first_batch else reply("", check)]
    if scenario == "rechecked":
        calls.append(reply("", check))
    if scenario == "late_change":
        def finish_after_external_edit():
            path.write_text("value = 3\n")
            return finish

        calls.append(finish_after_external_edit)
    elif scenario != "cancelled":
        calls.append(finish)
    script(*calls)

    result = engine.run_turn(agent, "Work on the fixture.", max_total_tool_calls=8)

    assert engine._last_turn_receipt.get("test_verification") == expected
    notices = [
        message for messages, _, _ in seen for message in messages
        if (message.get("content") or "").startswith("Host verification:")
    ]
    # A notice remains visible in subsequent requests. It enters the history once.
    assert bool(notices) == (scenario in {"stale_same_batch", "rechecked"})
    if notices:
        assert "test command returned exit 0" in notices[0]["content"]
        assert sum(
            message.get("content") == notices[0]["content"]
            for message in seen[-1][0]
        ) == 1
    assert commands == ["pytest -q"] * (2 if scenario == "rechecked" else 1)
    assert result.status == ("cancelled" if scenario == "cancelled" else "answered")
    if scenario in {"disabled", "cancelled", "no_snapshot"}:
        assert snapshot_count == (0 if scenario == "disabled" else 1)


@pytest.mark.parametrize(("comparison", "recall_granted", "expected"), [
    ("unchanged", True, "current"),
    ("edited", True, "stale"),
    ("unavailable", True, "unconfirmed"),
    ("unchanged", False, "current"),
])
def test_followup_revalidates_checkpointed_test_result(
    comparison, recall_granted, expected, runtime, monkeypatch, workspace_dir, tool_context,
):
    from infinidev.db.service import get_session_runtime_state
    from infinidev.tools.meta.recall_context_tool import RecallContextTool

    engine, agent, script, seen = runtime
    monkeypatch.setattr(settings, "TASK_TEST_REVISION_FEEDBACK_ENABLED", True)
    target = workspace_dir / "checked.py"
    target.write_text("value = 1\n")

    class CommandTool(BaseTool):
        name: str = "execute_command"
        description: str = "Return the scripted exit status."
        args_schema: type[BaseModel] = CommandInput

        def _run(self, command: str) -> str:
            return '{"exit_code": 0, "stdout": "observed output"}'

    def close_first_turn():
        saved = get_session_runtime_state(tool_context.session_id)["ui_state"]["conversation"]
        assert saved["receipt"]["closure"] == "unclosed"
        assert saved["test_observation"]["locator"] in saved["evidence"]
        return reply("", tc("finish_turn", message="Check executed."))

    script(reply("", tc("execute_command", command="pytest -q")), close_first_turn)
    assert engine.run_turn(agent, "Run the check.",
                           task_tools=[CommandTool(), RecallContextTool()]).status == "answered"
    if comparison == "edited":
        target.write_text("value = 2\n")
    if comparison == "unavailable":
        monkeypatch.setattr(turn_runtime.FileChangeTracker, "workspace_revision", lambda _: None)
    engine.begin_turn()
    script(reply("Reviewed the recorded observation."))
    tools = [RecallContextTool()] if recall_granted else []
    assert engine.run_turn(agent, "Do I need another check?", task_tools=tools).status == "answered"

    prior = next(m["content"] for m in seen[-1][0] if
                 m["content"].startswith("<prior_state_context>"))
    evidence = json.loads(prior.splitlines()[1])["previous_test_observation"]
    assert evidence["revision_status"] == expected
    assert evidence["recognized_test_command_exit_code"] == 0
    if recall_granted:
        assert "recall_context reference: wm-archive:" in evidence["locator"]
    else:
        assert "locator" not in evidence
    assert engine._last_turn_receipt["test_verification_origin"] == "prior_turn"
    saved = get_session_runtime_state(tool_context.session_id)["ui_state"]["conversation"]
    assert ("test_observation" in saved) is (expected == "current")
