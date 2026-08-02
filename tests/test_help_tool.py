"""Tests for HelpTool."""

import pytest

from infinidev.tools.meta.help_tool import HelpTool


class TestHelpTool:
    """Tests for HelpTool."""

    def test_overview(self, bound_tool):
        """No context returns category overview."""
        tool = bound_tool(HelpTool)
        result = tool._run()
        assert "Available help categories" in result
        assert "file" in result
        assert "edit" in result
        assert "code_intel" in result

    def test_category_file(self, bound_tool):
        """File category lists the currently registered file tools."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="file")
        assert "read_file" in result
        assert "list_directory" in result
        assert "code_search" in result
        assert "replace_lines" not in result

    def test_category_edit(self, bound_tool):
        """Edit category lists live edit tools, not retired alternatives."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="edit")
        assert "create_file" in result
        assert "edit_file" in result
        assert "edit_symbol" not in result
        assert "replace_lines" not in result

    def test_retired_tool_points_to_replacement(self, bound_tool):
        """Retired tools return actionable migration help."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="replace_lines")
        assert "was retired" in result
        assert "edit_file" in result
        assert "old_string" in result

    def test_specific_tool_create_file(self, bound_tool):
        """create_file help is rendered from its live schema and hints."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="create_file")
        assert "Fails if the file already exists" in result
        assert "## Parameters" in result
        assert "## Example" in result

    def test_unknown_topic(self, bound_tool):
        """Returns helpful message for unknown topics."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="nonexistent_tool")
        assert "No help found" in result
        assert "Available topics" in result

    def test_case_insensitive(self, bound_tool):
        """Handles case-insensitive lookups."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="FILE")
        assert "read_file" in result

    def test_substring_match(self, bound_tool):
        """Finds help via substring matching.

        Uses ``symbol`` as the query because it appears in several tool
        names (``search_symbols``, ``get_symbol_code``, ``edit_symbol``)
        after the partial_read alias was removed — a distinctive enough
        root that substring matching still has something to land on.
        """
        tool = bound_tool(HelpTool)
        result = tool._run(context="symbol")
        assert "search_symbols" in result or "get_symbol_code" in result
