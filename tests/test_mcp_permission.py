"""Regression tests for the MCP_PERMISSION setting + the permissions tab row.

The TUI permissions tab gained a third toggle for MCP tool execution with
default = auto_approve (MCP servers are user-trusted, unlike shell commands).
These tests cover the setting default, the UI surface, and the runtime check
inside ``mcp_bridge.build_tool_class()._run``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ── Settings default ──────────────────────────────────────────────────────


def test_mcp_permission_defaults_to_auto_approve() -> None:
    """User-facing default: MCP servers the user opted into are trusted."""
    from infinidev.config.settings import Settings

    assert Settings.model_fields["MCP_PERMISSION"].default == "auto_approve"


def test_settings_module_exposes_mcp_permission(monkeypatch: pytest.MonkeyPatch) -> None:
    """The live settings singleton exposes MCP_PERMISSION."""
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "MCP_PERMISSION", "ask")
    assert settings.MCP_PERMISSION == "ask"


# ── TUI Permissions tab row ───────────────────────────────────────────────


def test_permissions_section_lists_three_rows_in_main_state() -> None:
    from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS

    keys = [k for (k, _, _) in SETTINGS_SECTIONS["Permissions"]]
    assert keys == [
        "EXECUTE_COMMANDS_PERMISSION",
        "FILE_OPERATIONS_PERMISSION",
        "MCP_PERMISSION",
    ]


def test_permissions_section_lists_three_rows_in_dropdown_control() -> None:
    from infinidev.ui.dialogs.dropdown_control import SETTINGS_SECTIONS

    keys = [k for (k, _, _) in SETTINGS_SECTIONS["Permissions"]]
    assert keys[-1] == "MCP_PERMISSION"


def test_permissions_section_lists_three_rows_in_sections_control() -> None:
    from infinidev.ui.dialogs.sections_control import SETTINGS_SECTIONS

    keys = [k for (k, _, _) in SETTINGS_SECTIONS["Permissions"]]
    assert keys[-1] == "MCP_PERMISSION"


def test_mcp_permission_row_uses_select_with_auto_approve_first() -> None:
    """auto_approve must be the first option so it is the default in the UI."""
    from infinidev.ui.dialogs.settings_editor_state import SETTINGS_SECTIONS

    row = next(r for r in SETTINGS_SECTIONS["Permissions"] if r[0] == "MCP_PERMISSION")
    key, _desc, stype = row
    assert stype.startswith("select:")
    opts = stype[len("select:"):].split(",")
    assert opts[0] == "auto_approve"
    assert "ask" in opts
    assert "deny" in opts


# ── Runtime enforcement via mcp_bridge.build_tool_class ───────────────────


def _make_tool_class(*, name: str, server: str = "ken", read_only: bool = True):
    """Build an MCP tool class the same way ``discover_mcp_tool_classes`` does."""
    from infinidev.tools.mcp_bridge import build_tool_class

    fake_tool = MagicMock()
    fake_tool.server = server
    fake_tool.name = name
    fake_tool.description = f"fake {name}"
    fake_tool.input_schema = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
    }
    fake_tool.annotations = {}

    # Patch ``is_read_only`` so we can flip it deterministically per test.
    with patch("infinidev.tools.mcp_bridge.is_read_only", return_value=read_only):
        cls = build_tool_class(fake_tool)
    return cls


def _run_tool(cls, **kwargs: Any) -> str:
    """Run a built MCP tool class with a patched manager.call."""
    instance = cls()
    return instance._run(**kwargs)


def test_mcp_tool_runs_without_prompt_when_auto_approve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default mode must NOT prompt for MCP tools (per user requirement)."""
    from infinidev.config.settings import settings
    from infinidev.tools.mcp_bridge import build_tool_class

    monkeypatch.setattr(settings, "MCP_PERMISSION", "auto_approve")

    cls = _make_tool_class(name="ken_search", read_only=True)

    mock_result = MagicMock()
    mock_result.text = "ok"
    mock_result.is_error = False
    mock_result.data = None

    with patch(
        "infinidev.engine.mcp_client.get_default_mcp_manager"
    ) as manager_factory:
        manager_factory.return_value.call.return_value = mock_result
        result = _run_tool(cls, q="hi")

    assert "ok" in result
    manager_factory.return_value.call.assert_called_once()


def test_mcp_tool_blocked_when_deny(monkeypatch: pytest.MonkeyPatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "MCP_PERMISSION", "deny")

    cls = _make_tool_class(name="ken_search", read_only=False)

    with patch(
        "infinidev.engine.mcp_client.get_default_mcp_manager"
    ) as manager_factory:
        result = _run_tool(cls, q="hi")

    manager_factory.return_value.call.assert_not_called()
    assert "denied" in result.lower()


def test_mcp_read_only_tool_runs_when_auto_without_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In ``auto`` mode, read-only MCP tools run even without a permission UI."""
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "MCP_PERMISSION", "auto")

    cls = _make_tool_class(name="ken_search", read_only=True)

    mock_result = MagicMock()
    mock_result.text = "ok"
    mock_result.is_error = False
    mock_result.data = None

    with patch(
        "infinidev.engine.mcp_client.get_default_mcp_manager"
    ) as manager_factory:
        manager_factory.return_value.call.return_value = mock_result
        result = _run_tool(cls, q="hi")

    assert "ok" in result


def test_mcp_side_effecting_tool_blocked_in_auto_with_no_handler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail closed: side-effecting MCP tools block when no approval UI exists."""
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "MCP_PERMISSION", "auto")

    cls = _make_tool_class(name="ken_write", read_only=False)

    # No permission handler registered (classic / bench mode)
    with patch(
        "infinidev.tools.permission.is_permission_handler_registered",
        return_value=False,
    ), patch(
        "infinidev.engine.mcp_client.get_default_mcp_manager"
    ) as manager_factory:
        result = _run_tool(cls)

    manager_factory.return_value.call.assert_not_called()
    assert "MCP_PERMISSION=auto" in result


def test_mcp_ask_mode_prompts_user(monkeypatch: pytest.MonkeyPatch) -> None:
    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "MCP_PERMISSION", "ask")

    cls = _make_tool_class(name="ken_search", read_only=True)

    with patch(
        "infinidev.tools.permission.request_permission",
        return_value=False,
    ) as request_perm, patch(
        "infinidev.engine.mcp_client.get_default_mcp_manager"
    ) as manager_factory:
        result = _run_tool(cls, q="hi")

    request_perm.assert_called_once()
    manager_factory.return_value.call.assert_not_called()
    assert "denied" in result.lower()