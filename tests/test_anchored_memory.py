"""End-to-end tests for the anchored memory subsystem.

Covers the four new pieces added in commits ``1a398f0`` and ``a78367a``:

  * ``db.service.get_anchored_findings`` — the retrieval query
  * ``tool_executor.annotate_with_memory`` — the result-time injection
  * ``tool_executor._MEMORY_HANDLERS`` — the per-tool anchor extractors
  * ``tools.knowledge.RecordFindingTool`` — the write-side validation
    (anchored types MUST carry at least one anchor)

The tests use the ``temp_db`` fixture from ``conftest.py`` so the
user's real ``~/.infinidev/infinidev.db`` is never touched.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from infinidev.db.service import get_anchored_findings
from infinidev.engine.tool_executor import (
    _MEMORY_HANDLERS,
    annotate_with_memory,
)
from infinidev.tools.base.db import execute_with_retry
from infinidev.tools.knowledge.finding_types import (
    FINDING_TYPES,
    FINDING_TYPE_HELP,
)
from infinidev.tools.knowledge.record_finding_input import RecordFindingInput
from infinidev.tools.knowledge.update_finding_input import UpdateFindingInput


def _insert_finding(
    *,
    topic: str,
    content: str,
    finding_type: str = "lesson",
    confidence: float = 0.9,
    anchor_file: str | None = None,
    anchor_symbol: str | None = None,
    anchor_tool: str | None = None,
    anchor_error: str | None = None,
) -> int:
    """Minimal direct-SQL insert used by the retrieval tests.

    Bypasses ``RecordFindingTool`` on purpose — we want to exercise the
    retrieval and injection sides in isolation from the write side.
    """
    def _do(conn: sqlite3.Connection) -> int:
        cursor = conn.execute(
            """
            INSERT INTO findings (
                project_id, session_id, agent_id, topic, content,
                finding_type, confidence, status,
                anchor_file, anchor_symbol, anchor_tool, anchor_error
            )
            VALUES (1, 'test', 'test', ?, ?, ?, ?, 'active',
                    ?, ?, ?, ?)
            """,
            (
                topic, content, finding_type, confidence,
                anchor_file, anchor_symbol, anchor_tool, anchor_error,
            ),
        )
        conn.commit()
        return cursor.lastrowid

    return execute_with_retry(_do)


# ── get_anchored_findings ────────────────────────────────────────────────


class TestGetAnchoredFindings:
    """Retrieval query semantics."""

    def test_empty_db_returns_empty(self, temp_db):
        """No findings → empty list."""
        result = get_anchored_findings(anchor_file="anything.py")
        assert result == []

    def test_no_anchors_supplied_returns_empty(self, temp_db):
        """Calling with zero anchor kwargs must return [] (not crash)."""
        # Even with findings in the DB, a query with no anchors must
        # return empty — otherwise we'd accidentally load everything.
        _insert_finding(topic="x", content="y", anchor_file="a.py")
        assert get_anchored_findings() == []

    def test_matches_by_file(self, temp_db):
        """Finding with matching anchor_file is returned."""
        fid = _insert_finding(
            topic="known gotcha",
            content="something about this file",
            anchor_file="src/foo.py",
        )
        result = get_anchored_findings(anchor_file="src/foo.py")
        assert len(result) == 1
        assert result[0]["id"] == fid
        assert result[0]["topic"] == "known gotcha"

    def test_matches_by_symbol(self, temp_db):
        """Finding with matching anchor_symbol is returned."""
        _insert_finding(
            topic="verify_token quirk",
            content="...",
            anchor_symbol="AuthModule.verify_token",
        )
        result = get_anchored_findings(anchor_symbol="AuthModule.verify_token")
        assert len(result) == 1

    def test_matches_by_tool(self, temp_db):
        """Finding with matching anchor_tool is returned."""
        _insert_finding(
            topic="pytest quirks",
            content="...",
            anchor_tool="pytest",
        )
        result = get_anchored_findings(anchor_tool="pytest")
        assert len(result) == 1

    def test_matches_by_error(self, temp_db):
        """Finding with matching anchor_error is returned."""
        _insert_finding(
            topic="db lock root cause",
            content="...",
            anchor_error="database is locked",
        )
        result = get_anchored_findings(anchor_error="database is locked")
        assert len(result) == 1

    def test_or_semantics_across_anchors(self, temp_db):
        """Any matching anchor returns the finding — it's OR, not AND."""
        # One finding with only a file anchor, another with only a tool.
        _insert_finding(topic="A", content="...", anchor_file="x.py")
        _insert_finding(topic="B", content="...", anchor_tool="pytest")

        # Querying with both file AND tool should return both rows.
        result = get_anchored_findings(
            anchor_file="x.py", anchor_tool="pytest",
        )
        topics = sorted(r["topic"] for r in result)
        assert topics == ["A", "B"]

    def test_only_active_or_provisional(self, temp_db):
        """Dismissed/rejected findings are not returned."""
        # Insert one active, one rejected (status != active/provisional).
        _insert_finding(topic="active one", content="...", anchor_file="x.py")

        def _insert_rejected(conn):
            conn.execute(
                """
                INSERT INTO findings (
                    project_id, topic, content, finding_type, status, anchor_file
                ) VALUES (1, ?, ?, 'lesson', 'rejected', ?)
                """,
                ("rejected one", "...", "x.py"),
            )
            conn.commit()
        execute_with_retry(_insert_rejected)

        result = get_anchored_findings(anchor_file="x.py")
        assert len(result) == 1
        assert result[0]["topic"] == "active one"

    def test_only_anchored_finding_types(self, temp_db):
        """observation/hypothesis/etc. are NOT returned even with anchors."""
        # An observation with an anchor shouldn't fire — anchoring is
        # only for lesson/rule/landmine by design.
        _insert_finding(
            topic="observation",
            content="...",
            finding_type="observation",
            anchor_file="x.py",
        )
        _insert_finding(
            topic="lesson",
            content="...",
            finding_type="lesson",
            anchor_file="x.py",
        )
        result = get_anchored_findings(anchor_file="x.py")
        topics = sorted(r["topic"] for r in result)
        assert topics == ["lesson"]

    def test_respects_limit(self, temp_db):
        """Returns at most ``limit`` rows, ordered by confidence DESC."""
        for i in range(6):
            _insert_finding(
                topic=f"lesson {i}",
                content="...",
                anchor_file="x.py",
                confidence=0.1 * (i + 1),  # 0.1, 0.2, ..., 0.6
            )
        result = get_anchored_findings(anchor_file="x.py", limit=3)
        assert len(result) == 3
        # Highest confidence first
        confidences = [r["confidence"] for r in result]
        assert confidences == sorted(confidences, reverse=True)


# ── annotate_with_memory ─────────────────────────────────────────────────


class TestAnnotateWithMemory:
    """Result-time injection into tool output."""

    def test_no_handler_returns_result_unchanged(self, temp_db):
        """Tools without a memory handler pass through untouched."""
        out = annotate_with_memory(
            tool_name="web_fetch",  # not in _MEMORY_HANDLERS
            arguments=json.dumps({"url": "https://example.com"}),
            result="some content",
            project_id=1,
        )
        assert out == "some content"

    def test_no_match_returns_result_unchanged(self, temp_db):
        """With a handler but no matching memory, result is unchanged."""
        # DB is empty — no findings exist.
        out = annotate_with_memory(
            tool_name="read_file",
            arguments=json.dumps({"file_path": "never-seen.py"}),
            result="file contents",
            project_id=1,
        )
        assert out == "file contents"

    def test_file_anchor_injects_lesson(self, temp_db):
        """A file-anchored lesson appears below the tool result."""
        _insert_finding(
            topic="pydantic warm-up",
            content="do not remove the warm-up at bootstrap",
            anchor_file="engine.py",
        )
        out = annotate_with_memory(
            tool_name="read_file",
            arguments=json.dumps({"file_path": "engine.py"}),
            result="def execute(): pass",
            project_id=1,
        )
        assert "def execute" in out  # original result preserved
        assert "Known lessons" in out
        assert "pydantic warm-up" in out

    def test_error_result_not_annotated(self, temp_db):
        """Error results are not annotated — lessons next to errors are noise."""
        _insert_finding(
            topic="x", content="y", anchor_file="engine.py",
        )
        out = annotate_with_memory(
            tool_name="read_file",
            arguments=json.dumps({"file_path": "engine.py"}),
            result='{"error": "file not found"}',
            project_id=1,
        )
        assert "Known lessons" not in out
        assert out == '{"error": "file not found"}'

    def test_empty_result_not_annotated(self, temp_db):
        """Empty result gets the same skip treatment as an error."""
        _insert_finding(topic="x", content="y", anchor_file="engine.py")
        out = annotate_with_memory(
            tool_name="read_file",
            arguments=json.dumps({"file_path": "engine.py"}),
            result="",
            project_id=1,
        )
        assert out == ""

    def test_malformed_arguments_returns_result_unchanged(self, temp_db):
        """Bad JSON in arguments doesn't crash — result passes through."""
        out = annotate_with_memory(
            tool_name="read_file",
            arguments="not-json-at-all",
            result="file contents",
            project_id=1,
        )
        assert out == "file contents"

    def test_tool_anchor_from_execute_command(self, temp_db):
        """execute_command extracts the first token as the tool anchor."""
        _insert_finding(
            topic="pytest isolation",
            content="use temp_db",
            anchor_tool="pytest",
        )
        out = annotate_with_memory(
            tool_name="execute_command",
            arguments=json.dumps({"command": "pytest tests/ -q"}),
            result="collected 42 tests",
            project_id=1,
        )
        assert "pytest isolation" in out

    def test_symbol_anchor_on_get_symbol_code(self, temp_db):
        """get_symbol_code matches on the ``name`` argument."""
        _insert_finding(
            topic="verify_token is sensitive",
            content="don't touch without running integration suite",
            anchor_symbol="Auth.verify_token",
        )
        out = annotate_with_memory(
            tool_name="get_symbol_code",
            arguments=json.dumps({"name": "Auth.verify_token"}),
            result="def verify_token(token): ...",
            project_id=1,
        )
        assert "verify_token is sensitive" in out


# ── _MEMORY_HANDLERS coverage ────────────────────────────────────────────


class TestMemoryHandlers:
    """The per-tool anchor-extraction functions."""

    def test_all_expected_tools_have_handlers(self):
        """Every tool that mutates or reads files should have a handler."""
        expected = {
            "read_file", "partial_read", "create_file", "edit_file",
            "replace_lines", "add_content_after_line",
            "add_content_before_line", "list_directory",
            "get_symbol_code", "edit_symbol", "add_symbol",
            "remove_symbol", "search_symbols", "execute_command",
        }
        actual = set(_MEMORY_HANDLERS.keys())
        missing = expected - actual
        assert not missing, f"Handlers missing for: {missing}"

    def test_web_tools_not_in_handlers(self):
        """Web / network tools do NOT get memory annotation."""
        # Memory handlers should only cover things with a local anchor.
        # Adding web_fetch would match URLs as 'anchors' which isn't
        # what we want.
        assert "web_fetch" not in _MEMORY_HANDLERS
        assert "web_search" not in _MEMORY_HANDLERS


# ── RecordFindingTool validation ─────────────────────────────────────────


class TestRecordFindingValidation:
    """The write-side invariant: lesson/rule/landmine REQUIRE an anchor."""

    def _run_tool(self, **kwargs):
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = RecordFindingTool()
        return tool

    def test_lesson_without_anchor_is_rejected(self, bound_tool):
        """A lesson without any anchor_* must fail with a helpful error."""
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="some lesson",
            content="some content",
            finding_type="lesson",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        # Error message must tell the LLM exactly what it did wrong
        # AND suggest an alternative.
        assert "anchor_" in parsed["error"]
        assert "observation" in parsed["error"]

    def test_rule_without_anchor_is_rejected(self, bound_tool):
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="some rule",
            content="...",
            finding_type="rule",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "anchor_" in parsed["error"]

    def test_landmine_without_anchor_is_rejected(self, bound_tool):
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="some landmine",
            content="...",
            finding_type="landmine",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "anchor_" in parsed["error"]

    def test_observation_without_anchor_is_accepted(self, bound_tool):
        """observation is the un-anchored escape hatch."""
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="plain observation",
            content="no anchor needed",
            finding_type="observation",
        )
        parsed = json.loads(result)
        # Should succeed — "error" must not be present
        assert "error" not in parsed
        assert parsed.get("finding_id") is not None

    def test_lesson_with_file_anchor_is_accepted(self, bound_tool):
        """A lesson with a file anchor passes validation."""
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="real lesson",
            content="...",
            finding_type="lesson",
            anchor_file="src/foo.py",
        )
        parsed = json.loads(result)
        assert "error" not in parsed
        assert parsed.get("anchored") is True

    def test_lesson_with_only_error_anchor_is_accepted(self, bound_tool):
        """Any single anchor is enough — not just file/symbol."""
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="error-anchored lesson",
            content="...",
            finding_type="lesson",
            anchor_error="database is locked",
        )
        parsed = json.loads(result)
        assert "error" not in parsed
        assert parsed.get("anchored") is True

    def test_invalid_finding_type_rejected(self, bound_tool):
        """Unknown finding_type is rejected before the anchor check runs."""
        from infinidev.tools.knowledge.record_finding_tool import RecordFindingTool
        tool = bound_tool(RecordFindingTool)
        result = tool._run(
            title="x",
            content="y",
            finding_type="nonsense_type",
        )
        parsed = json.loads(result)
        assert "error" in parsed
        assert "Invalid finding_type" in parsed["error"]


# ── FINDING_TYPES centralization ─────────────────────────────────────────


class TestFindingTypesDedup:
    """Regression guard: ``FINDING_TYPES`` must stay centralized."""

    def test_anchored_types_present(self):
        """The three anchored types are defined."""
        for t in ("lesson", "rule", "landmine"):
            assert t in FINDING_TYPES

    def test_classic_types_preserved(self):
        """The original types still work."""
        for t in ("observation", "hypothesis", "project_context"):
            assert t in FINDING_TYPES

    def test_help_covers_new_types(self):
        """``FINDING_TYPE_HELP`` mentions all three anchored types."""
        help_lower = FINDING_TYPE_HELP.lower()
        assert "lesson" in help_lower
        assert "rule" in help_lower
        assert "landmine" in help_lower

    def test_record_and_update_share_the_same_tuple(self):
        """Both tool modules import from ``finding_types`` — same object."""
        from infinidev.tools.knowledge import record_finding_tool as rft
        from infinidev.tools.knowledge import update_finding_tool as uft
        assert rft.FINDING_TYPES is uft.FINDING_TYPES is FINDING_TYPES

    def test_input_schemas_have_anchor_fields(self):
        """Both input schemas expose the four anchor_* fields."""
        for schema in (RecordFindingInput, UpdateFindingInput):
            fields = schema.model_fields
            for anchor in (
                "anchor_file", "anchor_symbol", "anchor_tool", "anchor_error",
            ):
                assert anchor in fields, f"{anchor} missing from {schema.__name__}"
