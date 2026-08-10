"""Task-level capability routing reduces choice without removing capability."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.tool_routing import select_developer_tools, task_capabilities
from infinidev.engine.loop.context_builder import (
    _filter_plan_free_tools,
    _resolve_tools,
)
from infinidev.tools import get_tools_for_role


def _names(description: str, plan=None) -> set[str]:
    return {
        tool.name
        for tool in select_developer_tools(
            get_tools_for_role("developer"), description, plan
        )
    }


def test_routine_code_task_gets_core_not_every_specialist_tool() -> None:
    names = _names("Fix the token validation bug and run its tests")

    assert {
        "read_file", "edit_file", "execute_command", "git_diff",
        "send_message", "describe_tool",
    } <= names
    assert "web_search" not in names
    assert "git_commit" not in names
    assert "write_report" not in names
    assert "run_in_background" not in names
    assert "request_capability" in names


def test_small_model_keeps_prompt_required_communication_tools() -> None:
    names = {
        tool.name
        for tool in get_tools_for_role("developer", small_model=True)
    }

    assert {"send_message", "describe_tool"} <= names


def test_read_only_task_avoids_unrelated_destructive_and_specialist_tools() -> None:
    names = _names("Read pyproject.toml and report the project name. Do not edit files.")

    assert {"read_file", "code_search", "execute_command"} <= names
    assert "delete_file" not in names
    assert "move_file" not in names
    assert "view_image" not in names
    assert "code_interpreter" not in names


def test_task_protocol_does_not_enable_destructive_tools() -> None:
    description = """<task authority="USER_LITERAL">
Fix nested models and run tests.
Requested result kind: implementation
</task>

<rolling-step-policy authority="SYSTEM">
Add, modify, or remove model-inferred Steps whenever evidence changes.
</rolling-step-policy>"""

    names = _names(description)

    assert {"read_file", "edit_file", "execute_command"} <= names
    assert "delete_file" not in names
    assert "rollback_task_changes" not in names


def test_domain_remove_does_not_enable_file_management() -> None:
    names = _names(
        "Remove a once event handler before invoking it, then test re-entrant emit"
    )

    assert "delete_file" not in names
    assert "move_file" not in names
    assert "rollback_task_changes" not in names


def test_explicit_file_management_still_routes_file_tools() -> None:
    names = _names("Rename the file old_config.py to config.py")

    assert {"delete_file", "move_file", "preview_changes"} <= names


def test_explicit_capabilities_restore_their_tools() -> None:
    names = _names(
        "Research the latest online API documentation, write a report, then git commit the changes"
    )

    assert {"web_search", "find_documentation", "write_report", "git_commit"} <= names


def test_planner_detail_can_enable_a_capability() -> None:
    plan = SimpleNamespace(
        overview="Implement the feature",
        steps=[SimpleNamespace(
            title="Verify upstream behavior",
            detail="Browse the web for the current version API reference",
            expected_output="Source captured",
        )],
    )

    assert "web" in task_capabilities("Implement the feature", plan)


def test_configured_mcp_tools_are_preserved() -> None:
    class Mcp:
        name = "custom_remote_read"
        mcp_server = "custom"

    instance = Mcp()
    assert select_developer_tools([instance], "Fix a local bug") == [instance]


def test_capability_expansion_adds_only_the_requested_group() -> None:
    from infinidev.engine.tool_routing import expand_capability_tools

    available = [
        SimpleNamespace(name="read_file"),
        SimpleNamespace(name="web_search"),
        SimpleNamespace(name="web_fetch"),
        SimpleNamespace(name="git_commit"),
    ]
    expanded = expand_capability_tools([available[0]], available, "web")

    assert [tool.name for tool in expanded] == ["read_file", "web_search", "web_fetch"]


def test_request_capability_does_not_claim_effect_permission(bound_tool) -> None:
    import json

    from infinidev.tools.base.context import set_capability_requester
    from infinidev.tools.meta.request_capability_tool import RequestCapabilityTool

    set_capability_requester(
        "test-agent",
        lambda capability, rationale: json.dumps({
            "capability": capability,
            "rationale": rationale,
            "permission_granted": False,
        }),
    )
    result = json.loads(bound_tool(RequestCapabilityTool).run(
        capability="web",
        rationale="Need current upstream documentation",
    ))

    assert result["capability"] == "web"
    assert result["permission_granted"] is False


def test_empty_agent_toolbox_restores_local_developer_core() -> None:
    agent = SimpleNamespace(agent_id="empty-toolbox", tools=[])

    names = {tool.name for tool in _resolve_tools(agent, None, False)}

    assert {"read_file", "list_directory", "code_search", "execute_command"} <= names


def test_mcp_only_agent_toolbox_keeps_mcp_and_restores_local_core() -> None:
    mcp = SimpleNamespace(name="remote_lookup", mcp_server="remote")
    agent = SimpleNamespace(agent_id="mcp-only", tools=[mcp])

    tools = _resolve_tools(agent, None, False)
    names = {tool.name for tool in tools}

    assert "remote_lookup" in names
    assert {"read_file", "list_directory", "code_search", "execute_command"} <= names


def test_plan_free_toolbox_hides_plan_mutation_tools() -> None:
    tools = [
        SimpleNamespace(name="read_file"),
        SimpleNamespace(name="add_step"),
        SimpleNamespace(name="modify_step"),
        SimpleNamespace(name="remove_step"),
        SimpleNamespace(name="step_complete"),
    ]

    assert [tool.name for tool in _filter_plan_free_tools(tools)] == [
        "read_file",
        "step_complete",
    ]
