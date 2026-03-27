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
        """File category lists file tools."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="file")
        assert "read_file" in result
        assert "create_file" in result
        assert "replace_lines" in result

    def test_category_edit(self, bound_tool):
        """Edit category lists edit tools."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="edit")
        assert "edit_symbol" in result
        assert "replace_lines" in result

    def test_specific_tool(self, bound_tool):
        """Specific tool returns detailed help."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="replace_lines")
        assert "PARAMS:" in result
        assert "EXAMPLES:" in result
        assert "start_line" in result

    def test_specific_tool_create_file(self, bound_tool):
        """create_file help includes examples."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="create_file")
        assert "FAILS if the file already exists" in result

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
        """Finds help via substring matching."""
        tool = bound_tool(HelpTool)
        result = tool._run(context="partial")
        assert "partial_read" in result
