"""Runtime validation at the final tool-dispatch boundary."""

from __future__ import annotations

import json
from types import SimpleNamespace

from pydantic import BaseModel, Field

from infinidev.engine.tool_dispatch import execute_tool_call
from infinidev.tools.base.base_tool import InfinibayBaseTool


class _ConstrainedInput(BaseModel):
    value: str = Field(min_length=5)


class _ConstrainedTool(InfinibayBaseTool):
    name: str = "constrained"
    description: str = "Test a constrained argument."
    args_schema: type[BaseModel] = _ConstrainedInput

    def _run(self, value: str) -> str:
        return value


class _RecallLikeTool(InfinibayBaseTool):
    name: str = "recall_context"
    description: str = "Test recall query aliases."

    def _run(self, query: str) -> str:
        return query


class _ShellLikeTool(InfinibayBaseTool):
    name: str = "execute_command"
    description: str = "Record a shell command without running it."

    def _run(self, command: str, cwd: str | None = None) -> str:
        return json.dumps({"command": command, "cwd": cwd})


class _SearchLikeTool(InfinibayBaseTool):
    name: str = "code_search"
    description: str = "Record normalized search arguments."

    def _run(self, pattern: str, context_lines: int = 0) -> str:
        return json.dumps({"pattern": pattern, "context_lines": context_lines})


class _EditLikeTool(InfinibayBaseTool):
    name: str = "edit_file"
    description: str = "Record normalized edit arguments."

    def _run(self, file_path: str, old_string: str, new_string: str) -> str:
        return json.dumps({
            "file_path": file_path,
            "old_string": old_string,
            "new_string": new_string,
        })


class _DelegateLikeTool(InfinibayBaseTool):
    name: str = "team_delegate"
    description: str = "Stand in for the orchestrator's delegation tool."

    def _run(self, tools: list[str] | None = None) -> str:
        return json.dumps({"tools": tools or []})


class _DescribeLikeTool(InfinibayBaseTool):
    name: str = "describe_tool"
    description: str = "Record the requested help topic."

    def _run(self, context: str | None = None) -> str:
        return json.dumps({"context": context})


class _PatchLikeTool(InfinibayBaseTool):
    name: str = "apply_file_patch"
    description: str = "Record normalized patch arguments."

    def _run(self, file_path: str, replacements: list[dict]) -> str:
        return json.dumps({"file_path": file_path, "replacements": replacements})


class _DeclareTestLikeTool(InfinibayBaseTool):
    name: str = "declare_test_command"
    description: str = "Record a custom test command."

    def _run(self, command_pattern: str) -> str:
        return command_pattern


class _ReadPathTool:
    name = "read_file"

    @staticmethod
    def _resolve_path(path: str) -> str:
        return path

    def _run(self, file_path: str, offset: int = 1) -> str:
        return json.dumps({"read": file_path, "offset": offset})


class _ListPathTool:
    name = "list_directory"

    def _run(self, file_path: str = ".") -> str:
        return json.dumps({"listed": file_path})


def test_dispatch_enforces_pydantic_constraints() -> None:
    tool = _ConstrainedTool()

    result = json.loads(
        execute_tool_call({tool.name: tool}, tool.name, {"value": "no"})
    )

    assert "validation failed" in result["error"]
    assert "at least 5 characters" in result["error"]


def test_dispatch_runs_after_successful_validation() -> None:
    tool = _ConstrainedTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"value": "valid value"}
    )

    assert result == "valid value"


def test_dispatch_maps_recall_context_to_query() -> None:
    tool = _RecallLikeTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"context": "package metadata"}
    )

    assert result == "package metadata"


def test_dispatch_maps_minimax_shell_aliases_to_execute_command() -> None:
    """Live M3 naming misses must execute instead of becoming false blockers."""
    tool = _ShellLikeTool()

    for alias in ("shell_command", "shell_exec", "shell_run", "shell"):
        result = execute_tool_call(
            {tool.name: tool},
            alias,
            {"command": "pwd", "cwd": "/tmp/project"},
        )

        assert json.loads(result) == {"command": "pwd", "cwd": "/tmp/project"}


def test_dispatch_moves_leading_cd_to_execute_command_cwd() -> None:
    tool = _ShellLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {"command": "cd '/tmp/project with spaces' && python -m pytest -q"},
    )

    assert json.loads(result) == {
        "command": "python -m pytest -q",
        "cwd": "/tmp/project with spaces",
    }


def test_dispatch_leaves_unsafe_leading_cd_for_permission_layer() -> None:
    tool = _ShellLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {"command": "cd $(pwd) && python -m pytest -q"},
    )

    assert json.loads(result) == {
        "command": "cd $(pwd) && python -m pytest -q",
        "cwd": None,
    }


def test_dispatch_ignores_false_minimax_background_hint() -> None:
    tool = _ShellLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        "run_command",
        {"command": "ls -la", "is_background": False},
    )

    assert json.loads(result) == {"command": "ls -la", "cwd": None}


def test_dispatch_refuses_true_background_hint() -> None:
    tool = _ShellLikeTool()

    result = json.loads(execute_tool_call(
        {tool.name: tool},
        "run_command",
        {"command": "server", "is_background": True},
    ))

    assert "run_in_background" in result["error"]


def test_dispatch_normalizes_minimax_declared_test_command() -> None:
    tool = _DeclareTestLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {"command": "pytest -q", "cwd": "/tmp/project"},
    )

    assert result == "pytest -q"


def test_dispatch_normalizes_minimax_code_search_context() -> None:
    tool = _SearchLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {"pattern": "Widget", "context": 20},
    )

    assert json.loads(result) == {"pattern": "Widget", "context_lines": 5}


def test_dispatch_expands_minimax_structured_edit_replacement() -> None:
    tool = _EditLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {
            "file_path": "src/widget.py",
            "replace": {"old": "before", "new": "after"},
        },
    )

    assert json.loads(result) == {
        "file_path": "src/widget.py",
        "old_string": "before",
        "new_string": "after",
    }


def test_dispatch_routes_read_file_on_directory_to_list_directory(tmp_path) -> None:
    read = _ReadPathTool()
    listing = _ListPathTool()

    result = execute_tool_call(
        {read.name: read, listing.name: listing},
        "read_file",
        {"file_path": str(tmp_path), "offset": 50},
    )

    assert json.loads(result) == {"listed": str(tmp_path)}


def test_dispatch_recovers_paraphrased_edit_parameters() -> None:
    """MiniMax-M3 writes 'old_text' where the schema says 'old_string'."""
    tool = _EditLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {
            "file_path": "src/widget.py",
            "old_text": "before",
            "new_text": "after",
        },
    )

    assert json.loads(result) == {
        "file_path": "src/widget.py",
        "old_string": "before",
        "new_string": "after",
    }


def test_dispatch_recovers_shell_command_alias() -> None:
    tool = _ShellLikeTool()

    result = execute_tool_call({tool.name: tool}, tool.name, {"exec": "pytest -q"})

    assert json.loads(result)["command"] == "pytest -q"


def test_dispatch_recovers_a_schema_wrapped_parameter_key() -> None:
    """The model emitted the schema's wording as the key, not the key itself."""
    tool = _ShellLikeTool()

    result = execute_tool_call(
        {tool.name: tool},
        tool.name,
        {'parameter name="command"': "pytest -q", "cwd": "/tmp"},
    )

    assert json.loads(result) == {"command": "pytest -q", "cwd": "/tmp"}


def test_dispatch_leaves_a_real_parameter_name_untouched() -> None:
    """The recovery must not rewrite a key that is already correct."""
    tool = _ShellLikeTool()

    result = execute_tool_call({tool.name: tool}, tool.name, {"command": "pytest -q"})

    assert json.loads(result)["command"] == "pytest -q"


def test_dispatch_maps_describe_tool_topic_alias() -> None:
    """MiniMax-M3 asked describe_tool(tool=...) where the schema says context."""
    tool = _DescribeLikeTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"tool": "edit_file"},
    )

    assert json.loads(result) == {"context": "edit_file"}


def test_describe_tool_alias_does_not_leak_to_other_tools() -> None:
    """The rewrite is local: 'tool' is not a global alias."""
    tool = _ShellLikeTool()

    result = execute_tool_call({tool.name: tool}, tool.name, {"tool": "edit_file"})

    assert "error" in json.loads(result)


def test_dispatch_maps_apply_file_patch_replacement_aliases() -> None:
    """MiniMax-M3 passed the patch under 'edits' and 'patch', not 'replacements'."""
    tool = _PatchLikeTool()

    for alias in ("edits", "patch", "changes"):
        result = execute_tool_call(
            {tool.name: tool},
            tool.name,
            {"file_path": "src/app.py", alias: [{"old_string": "a", "new_string": "b"}]},
        )
        payload = json.loads(result)
        assert "error" not in payload, f"{alias}: {payload}"
        assert payload["replacements"] == [{"old_string": "a", "new_string": "b"}]


def test_dispatch_recovers_a_required_parameter_from_an_invented_key() -> None:
    """MiniMax-M3 sent execute_command({"param-1": "ls -la"}).

    One required parameter, one invented key, and the right value: the mapping
    is unambiguous, and a rejection costs a full model round.
    """
    tool = _ShellLikeTool()

    result = execute_tool_call({tool.name: tool}, tool.name, {"param-1": "ls -la"})

    assert json.loads(result) == {"command": "ls -la", "cwd": None}


def test_the_recovery_stays_quiet_when_the_required_parameter_is_present() -> None:
    """Two keys, one valid: the invented one is still an error."""
    tool = _ShellLikeTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"command": "pwd", "param-1": "ls"},
    )

    assert "error" in json.loads(result)


def test_the_recovery_does_not_guess_with_several_required_parameters() -> None:
    """edit_file needs three parameters, so an invented key maps to nothing."""
    tool = _EditLikeTool()

    result = execute_tool_call(
        {tool.name: tool}, tool.name, {"param-0": "a", "param-1": "b"},
    )

    assert "error" in json.loads(result)


def test_a_rejected_call_still_reaches_the_hook_chain() -> None:
    """A call the model made must be observable even when it is rejected.

    ``execute_tool_call`` dispatches POST_TOOL at its end, so every rejection
    returned early was invisible to the UI and to the transcript trace.
    """
    from infinidev.engine.hooks.hooks import HookContext, HookEvent, hook_manager

    seen: list[HookContext] = []

    def _capture(ctx: HookContext) -> None:
        seen.append(ctx)

    hook_manager.register(
        HookEvent.POST_TOOL, _capture, priority=950, name="test-post-tool",
    )
    try:
        tool = _ShellLikeTool()
        # Unknown tool: this path does not even reach PRE_TOOL.
        execute_tool_call({tool.name: tool}, "no_such_tool", {})
        # Wrong parameter: this one does, and used to leave the call open.
        execute_tool_call({tool.name: tool}, tool.name, {"nonsense": 1})
    finally:
        hook_manager.unregister(HookEvent.POST_TOOL, "test-post-tool")

    assert len(seen) == 2, f"rejections never reached POST_TOOL: {seen}"
    assert {ctx.tool_name for ctx in seen} == {"no_such_tool", "execute_command"}
    assert all('"error"' in (ctx.result or "") for ctx in seen)


def test_an_ungranted_but_real_tool_names_the_route() -> None:
    """The orchestrator holds reads and team tools and no write tool at all.

    `edit_file` exists in the registry, so "unknown tool" is wrong and the
    similarity list is misleading: one recorded run spent ten rounds guessing
    at write tools before blocking. The message has to say the tool is real and
    name the route that reaches it.
    """
    tool = _ShellLikeTool()
    delegate = _DelegateLikeTool()

    result = execute_tool_call(
        {tool.name: tool, delegate.name: delegate}, "edit_file", {},
    )

    payload = json.loads(result)
    assert "exists but is not granted" in payload["error"]
    assert "team_delegate" in payload["error"]
    assert "Unknown tool" not in payload["error"]


def test_a_genuinely_unknown_tool_still_gets_the_similarity_list() -> None:
    tool = _ShellLikeTool()

    payload = json.loads(
        execute_tool_call({tool.name: tool}, "execute_comand", {}),
    )

    assert "Unknown tool" in payload["error"]
    assert "execute_command" in payload["error"]


# ── a model-authored argument is not always a string ───────────────────


def test_a_dict_shaped_message_no_longer_crashes_the_turn():
    """`(args.get("message") or "").strip()` raised AttributeError on a dict.

    One `test-selection` run in three failed with "The task engine failed:
    AttributeError: 'dict' object has no attribute 'strip'" — the whole turn
    lost to a shape the dispatcher is supposed to absorb.
    """
    from infinidev.engine.orchestration.chat_agent import _build_respond

    call = SimpleNamespace(
        id="c1",
        function=SimpleNamespace(
            name="respond",
            arguments=json.dumps({"message": {"text": "All four tests pass."}}),
        ),
    )

    result = _build_respond(call, "fix the tags")

    assert result.kind == "respond"
    assert result.reply == "All four tests pass."


def test_a_dict_shaped_understanding_does_not_crash_the_escalation():
    from infinidev.engine.orchestration.chat_agent import _build_escalate

    call = SimpleNamespace(
        id="c2",
        function=SimpleNamespace(
            name="escalate",
            arguments=json.dumps({"understanding": {"text": "Fix the rounding."}}),
        ),
    )

    result = _build_escalate(call, "fix the rounding")

    assert result.kind == "escalate"
    assert result.escalation.understanding == "Fix the rounding."


def test_the_coercion_refuses_what_it_cannot_read():
    """A shape with no text under a known key is "no answer", not a repr."""
    from infinidev.engine.tool_dispatch import text_argument

    assert text_argument({"message": {"text": "nested"}}) == ""
    assert text_argument(["a"]) == ""
    assert text_argument(None) == ""
    assert text_argument({"other": "x"}) == ""
    assert text_argument("  spaced  ") == "spaced"
    assert text_argument(3) == "3"
