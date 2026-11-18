"""Principal turns bound repeated local reads without discarding new work."""

from __future__ import annotations

import json

import pytest

from infinidev.engine.formats.manual_tool_call import ManualToolCall
from infinidev.engine.turns import runtime as turn_runtime
from infinidev.tools.file.code_search_tool import CodeSearchTool
from infinidev.tools.file.edit_file_tool import EditFileTool
from infinidev.tools.file.read_file_tool import ReadFileTool
from tests.test_turn_runtime import reply, runtime, tc


def search(index, pattern="VALUE_FOR_SEARCH"):
    return ManualToolCall(id=f"search_{index}", name="code_search", arguments=json.dumps({
        "file_path": "cycle.py", "pattern": pattern,
    }))


def has_warning(request):
    return any("Host progress: repeated read-only batches" in (message.get("content") or "")
               for message in request[0])


@pytest.mark.parametrize("alternate,dispatches", [(False, 5), (True, 6)])
def test_principal_warns_and_stops_repeated_searches_with_unset_limits(
    alternate, dispatches, runtime, workspace_dir, auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    (workspace_dir / "cycle.py").write_text("VALUE_FOR_SEARCH = 1\nOTHER_VALUE = 2\n")
    agent.tools = [CodeSearchTool()]
    script(*(reply("", search(i, "OTHER_VALUE" if alternate and i % 2 else "VALUE_FOR_SEARCH"))
             for i in range(40)), reply("", tc("finish_turn", message="Sentinel.")))

    result = engine.run_turn(agent, "Find the values.", max_total_tool_calls=0, max_iterations=0)

    assert result.status == "exhausted"
    assert "repeated read-only" in result.message
    assert len(seen) == engine._last_turn_receipt["dispatches"] == dispatches
    assert engine._last_turn_receipt["closure"] == "interrupted"
    assert all(not has_warning(request) for request in seen[:dispatches - 2])
    assert all(has_warning(request) for request in seen[dispatches - 2:])


def test_distinct_files_with_identical_contents_are_new_evidence(
    runtime, workspace_dir, auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    agent.tools = [ReadFileTool()]
    reads = []
    for i in range(8):
        name = f"file_{i}.txt"
        (workspace_dir / name).write_text("ready\n")
        reads.append(reply("", ManualToolCall(
            id=f"read_{i}", name="read_file", arguments=json.dumps({"file_path": name}),
        )))
    script(*reads, reply("", tc("finish_turn", message="All files were read.")))

    result = engine.run_turn(agent, "Inspect the files.", max_total_tool_calls=0, max_iterations=0)

    assert result.status == "answered"
    assert engine._last_turn_receipt["dispatches"] == 8
    assert not any(has_warning(request) for request in seen)


@pytest.mark.parametrize("progress", ["new_result", "guidance", "write_in_batch"])
def test_principal_read_warning_clears_for_progress(
    progress, runtime, monkeypatch, workspace_dir, auto_approve_permissions, sandbox_disabled,
):
    engine, agent, script, seen = runtime
    source = workspace_dir / "cycle.py"
    source.write_text("VALUE_FOR_SEARCH = 1\n")
    agent.tools = [CodeSearchTool(), EditFileTool()]
    dispatch = turn_runtime.execute_tool_call
    counts = {"code_search": 0, "edit_file": 0}

    def observe(tools, name, args, **kwargs):
        body = dispatch(tools, name, args, **kwargs)
        if name in counts:
            counts[name] += 1
        if progress == "guidance" and name == "code_search" and counts[name] == 5:
            engine.inject_message("You have enough evidence. Please conclude.")
        return body

    def observe_progress():
        assert has_warning(seen[-1])
        if progress == "new_result":
            source.write_text("VALUE_FOR_SEARCH = 2\n")
        if progress == "write_in_batch":
            edit = tc("edit_file", file_path="cycle.py", old_string="= 1", new_string="= 2")
            return reply("", search(4), edit)
        return reply("", search(4))

    monkeypatch.setattr(turn_runtime, "execute_tool_call", observe)
    script(*(reply("", search(i)) for i in range(4)), observe_progress,
           reply("", tc("finish_turn", message="Observed the requested value.")))

    result = engine.run_turn(agent, "Find the value.", max_total_tool_calls=0, max_iterations=0)

    assert result.status == "answered"
    assert engine._last_turn_receipt["closure"] == "finish_turn"
    assert counts == {"code_search": 5, "edit_file": int(progress == "write_in_batch")}
    assert not has_warning(seen[-1])
    if progress == "guidance":
        assert seen[-1][0][-1]["content"] == "You have enough evidence. Please conclude."
    else:
        assert source.read_text() == "VALUE_FOR_SEARCH = 2\n"
