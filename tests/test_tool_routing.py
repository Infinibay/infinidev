"""Task-level capability routing reduces choice without removing capability."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.tool_routing import select_developer_tools, task_capabilities
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

    assert {"read_file", "edit_file", "execute_command", "git_diff"} <= names
    assert "web_search" not in names
    assert "git_commit" not in names
    assert "write_report" not in names
    assert "run_in_background" not in names
    assert "request_capability" in names


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
