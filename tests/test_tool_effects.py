"""Effects-based authorization independent of tool names and providers."""

from __future__ import annotations

from infinidev.config.settings import settings
from infinidev.tools.base.tool_effects import ToolEffects, check_effect_permission
from infinidev.tools.permission import set_permission_handler


def test_pure_read_does_not_need_an_approval_ui() -> None:
    set_permission_handler(None)
    assert check_effect_permission(
        "read_something", ToolEffects(reads_workspace=True), {}
    ) is None


def test_external_mutation_fails_closed_headless() -> None:
    original = settings.TOOL_EFFECTS_PERMISSION
    settings.TOOL_EFFECTS_PERMISSION = "auto"
    set_permission_handler(None)
    try:
        error = check_effect_permission(
            "neutral_name",
            ToolEffects(mutates_external_state=True),
            {"target": "production"},
        )
    finally:
        settings.TOOL_EFFECTS_PERMISSION = original

    assert error is not None
    assert "no approval UI" in error


def test_invalid_mode_fails_closed() -> None:
    original = settings.TOOL_EFFECTS_PERMISSION
    settings.TOOL_EFFECTS_PERMISSION = "typo"
    try:
        error = check_effect_permission(
            "git_commit", ToolEffects(mutates_git=True), {}
        )
    finally:
        settings.TOOL_EFFECTS_PERMISSION = original

    assert error is not None
    assert "invalid TOOL_EFFECTS_PERMISSION" in error


def test_every_developer_tool_has_effect_metadata() -> None:
    from infinidev.tools import get_tools_for_role

    missing = [tool.name for tool in get_tools_for_role("developer") if tool.effects.is_empty]

    assert missing == []


def test_every_developer_tool_has_use_constraints() -> None:
    from infinidev.tools import get_tools_for_role

    missing = [
        tool.name
        for tool in get_tools_for_role("developer")
        if tool.use_constraints.is_empty
    ]

    assert missing == []


def test_schema_preserves_effects_and_non_use_guidance() -> None:
    from infinidev.engine.schema_sanitizer import tool_to_openai_schema
    from infinidev.tools.git.git_commit_tool import GitCommitTool
    from infinidev.tools.base.tool_effects import apply_local_effect_defaults

    tool = apply_local_effect_defaults(GitCommitTool())
    description = tool_to_openai_schema(tool)["function"]["description"]

    assert "Effects: writes workspace, mutates Git" in description
    assert "Do not use when:" in description
    assert "user explicitly requested" in description
