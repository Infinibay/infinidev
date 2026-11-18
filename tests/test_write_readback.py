"""The native turn can spend one request on observed-write verification."""
from types import SimpleNamespace

import pytest

from infinidev.config.settings import settings
from infinidev.engine.turns import runtime as turn_runtime
from infinidev.tools.file.create_file_tool import CreateFileTool
from infinidev.tools.file.edit_file_tool import EditFileTool
from infinidev.tools.file.read_file_tool import ReadFileTool
from tests.test_turn_runtime import reply, runtime, tc


@pytest.mark.parametrize(
    "scenario,expect_named",
    [
        ("enabled", True),
        ("enabled_auto", True),
        ("edit", True),
        ("off", False),
        ("missing_reader", False),
        ("failed_write", False),
        ("already_read", False),
        ("auto_only", False),
        ("other_provider", False),
        ("user_update", False),
        ("last_call", False),
        ("last_iteration", False),
        ("cancel_after_write", False),
        ("compaction", False),
    ],
)
def test_first_write_readback(
    scenario, expect_named, runtime, monkeypatch, workspace_dir,
    auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    path = workspace_dir / "result.txt"
    if scenario in {"failed_write", "edit"}:
        path.write_text("Pre-existing file.\n")

    monkeypatch.setattr(
        settings, "TASK_FIRST_WRITE_READBACK_ENABLED", scenario != "off",
    )
    post_tool_choice = "auto" if scenario == "enabled_auto" else "required"
    monkeypatch.setattr(settings, "TASK_POST_TOOL_CHOICE", post_tool_choice)
    monkeypatch.setattr(
        settings, "LLM_PROVIDER", "openai" if scenario == "other_provider" else "openai_compatible",
    )
    monkeypatch.setattr(
        turn_runtime, "get_model_capabilities",
        lambda: SimpleNamespace(
            supports_function_calling=True, supports_vision=False,
            restricts_tool_choice_to_auto=scenario == "auto_only",
        ),
    )
    if scenario == "compaction":
        monkeypatch.setattr(turn_runtime, "_get_model_max_context", lambda _: 100)
    agent.tools = [CreateFileTool(), EditFileTool()]
    if scenario != "missing_reader":
        agent.tools.append(ReadFileTool())

    write = tc("create_file", file_path=str(path), content="Expected bytes.\n")
    if scenario == "edit":
        write = tc("edit_file", file_path=str(path), old_string="Pre-existing file.",
                   new_string="Expected bytes.")
    read = tc("read_file", file_path=str(path))
    first = reply("", write, read) if scenario == "already_read" else reply("", write)

    if scenario in {"user_update", "cancel_after_write"}:
        execute = turn_runtime.execute_tool_call

        def with_update(dispatch, name, args, **kwargs):
            body = execute(dispatch, name, args, **kwargs)
            if name == "create_file":
                if scenario == "cancel_after_write":
                    engine.cancel()
                else:
                    engine.inject_message("Stop editing and report the observed result.")
            return body

        monkeypatch.setattr(turn_runtime, "execute_tool_call", with_update)

    another = tc("create_file", file_path=str(workspace_dir / "another.txt"), content="Second.\n")
    finish = reply("", tc("finish_turn", message="Done."))
    close_next = scenario in {"last_call", "last_iteration"}
    if close_next:
        script(first, finish)
    else:
        script(first, reply("", another), finish)
    result = engine.run_turn(agent, "Create the file and report.",
                             max_total_tool_calls=2 if scenario == "last_call" else 8,
                             max_iterations=2 if scenario == "last_iteration" else 8)

    named = {"type": "function", "function": {"name": "read_file"}}
    choices = [row[2]["tool_choice"] for row in seen]
    if scenario == "cancel_after_write":
        assert path.read_bytes() == b"Expected bytes.\n"
        assert result.status == "cancelled"
        assert choices == ["auto"]
        return
    assert result.status == "answered"
    assert choices[:2] == ["auto", named if expect_named else post_tool_choice]
    assert choices[2:] == (
        [] if close_next else [named if scenario == "failed_write" else post_tool_choice]
    )
    if expect_named:
        assert seen[1][2]["named_choice_fallback"] == post_tool_choice
    assert [schema["function"]["name"] for schema in seen[0][1]] == [
        schema["function"]["name"] for schema in seen[1][1]
    ]
