"""Permission regressions for the legacy ``delete_finding`` tool."""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import patch

from infinidev.config.settings import settings
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge import (
    DeleteFindingTool,
    RecordFindingTool,
    RejectFindingTool,
    UpdateFindingTool,
    ValidateFindingTool,
)


def _insert_finding() -> int:
    """Create a finding that the legacy deletion tool can remove."""

    def _insert(conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            """
            INSERT INTO findings (project_id, topic, content, finding_type, status)
            VALUES (1, 'obsolete finding', 'no longer applicable', 'lesson', 'active')
            """
        )
        conn.commit()
        return int(cursor.lastrowid)

    return execute_with_retry(_insert)


def _finding_exists(finding_id: int) -> bool:
    def _exists(conn: sqlite3.Connection) -> bool:
        return conn.execute(
            "SELECT 1 FROM findings WHERE id = ?", (finding_id,)
        ).fetchone() is not None

    return execute_with_retry(_exists)


def test_legacy_finding_tools_remain_importable_but_are_not_registered() -> None:
    """Direct compatibility imports survive without exposing retired schemas."""
    from infinidev.tools import get_tools_for_role

    names = {tool.name for tool in get_tools_for_role("developer", supports_vision=False)}
    legacy_names = {
        RecordFindingTool().name,
        ValidateFindingTool().name,
        RejectFindingTool().name,
        UpdateFindingTool().name,
        DeleteFindingTool().name,
    }
    assert names.isdisjoint(legacy_names)


def test_delete_finding_auto_approve_skips_confirmation(
    temp_db, monkeypatch
) -> None:
    """Edit auto-approval authorizes deletion without consulting the UI."""
    finding_id = _insert_finding()
    monkeypatch.setattr(settings, "FILE_OPERATIONS_PERMISSION", "auto_approve")

    with patch("infinidev.tools.permission.request_permission") as request_permission:
        result = DeleteFindingTool()._run(finding_id)

    assert '"deleted": true' in result
    assert not _finding_exists(finding_id)
    request_permission.assert_not_called()


def test_legacy_finding_mutations_auto_approve_skip_confirmation(
    tool_context, bound_tool, monkeypatch
) -> None:
    """All legacy finding mutations share edit auto-approval without UI prompts."""
    monkeypatch.setattr(settings, "FILE_OPERATIONS_PERMISSION", "auto_approve")

    with patch("infinidev.tools.permission.request_permission") as request_permission:
        recorded = json.loads(
            bound_tool(RecordFindingTool)._run(
                title="permission test",
                content="exercise every legacy finding mutation",
            )
        )
        finding_id = recorded["finding_id"]

        validated = json.loads(bound_tool(ValidateFindingTool)._run(finding_id))
        assert validated["status"] == "active"

        updated = json.loads(
            bound_tool(UpdateFindingTool)._run(finding_id, content="updated")
        )
        assert updated["finding_id"] == finding_id

        rejected = json.loads(
            bound_tool(RejectFindingTool)._run(finding_id, reason="obsolete")
        )
        assert rejected["status"] == "superseded"

        deleted = json.loads(bound_tool(DeleteFindingTool)._run(finding_id))
        assert deleted["deleted"] is True

    request_permission.assert_not_called()


def test_delete_finding_ask_mode_requests_confirmation(
    temp_db, monkeypatch
) -> None:
    """Ask mode retains the legacy confirmation flow before deletion."""
    finding_id = _insert_finding()
    monkeypatch.setattr(settings, "FILE_OPERATIONS_PERMISSION", "ask")

    with patch(
        "infinidev.tools.permission.request_permission", return_value=True
    ) as request_permission:
        result = DeleteFindingTool()._run(finding_id)

    assert '"deleted": true' in result
    assert not _finding_exists(finding_id)
    request_permission.assert_called_once()
