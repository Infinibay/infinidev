"""Tests for is_read_only tool classification and chat_agent whitelist.

The chat agent (Commit 2+ of the pipeline redesign) is the default entry
point for every user turn. It must NEVER have a write tool in its
schema — the schema is the security boundary; prompt rules are not
sufficient. These tests lock in that invariant: any future refactor
that accidentally exposes a write tool to the chat agent breaks here.
"""

from infinidev.tools import get_tools_for_role
from infinidev.tools.base.base_tool import InfinibayBaseTool
from infinidev.tools.file import (
    ReadFileTool, ListDirectoryTool, CodeSearchTool, GlobTool,
    CreateFileTool, ReplaceLinesTool, WriteFileTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
)
from infinidev.tools.code_intel import (
    FindReferencesTool, GetSymbolCodeTool, ListSymbolsTool, SearchSymbolsTool,
    ProjectStructureTool, AnalyzeCodeTool, FindSimilarMethodsTool,
    SearchByDocstringTool, IterSymbolsTool, ProjectStatsTool,
    EditSymbolTool, AddSymbolTool, RemoveSymbolTool,
    RenameSymbolTool, MoveSymbolTool,
)
from infinidev.tools.git import GitDiffTool, GitStatusTool, GitCommitTool, GitBranchTool
from infinidev.tools.knowledge import (
    ReadFindingsTool, SearchFindingsTool, RecordFindingTool,
    UpdateFindingTool, DeleteFindingTool,
)
from infinidev.tools.shell import ExecuteCommandTool, CodeInterpreterTool
from infinidev.tools.chat import SendMessageTool


# The canonical 18 tools that MUST be classified as read-only.
EXPECTED_READ_ONLY = {
    ReadFileTool, ListDirectoryTool, CodeSearchTool, GlobTool,
    FindReferencesTool, GetSymbolCodeTool, ListSymbolsTool, SearchSymbolsTool,
    ProjectStructureTool, AnalyzeCodeTool, FindSimilarMethodsTool,
    SearchByDocstringTool, IterSymbolsTool, ProjectStatsTool,
    GitDiffTool, GitStatusTool,
    ReadFindingsTool, SearchFindingsTool,
}

# Tools that MUST remain write-capable — forbidden in chat_agent whitelist.
KNOWN_WRITE_TOOLS = {
    CreateFileTool, ReplaceLinesTool, WriteFileTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
    EditSymbolTool, AddSymbolTool, RemoveSymbolTool,
    RenameSymbolTool, MoveSymbolTool,
    GitCommitTool, GitBranchTool,
    RecordFindingTool, UpdateFindingTool, DeleteFindingTool,
    ExecuteCommandTool, CodeInterpreterTool,
    SendMessageTool,
}


class TestBaseAttributeDefault:
    def test_base_class_default_is_false(self):
        # Every tool inherits the False default; read-only tools override it.
        class _Probe(InfinibayBaseTool):
            name: str = "_probe"
            description: str = "probe"

            def _run(self) -> str:
                return ""

        assert _Probe().is_read_only is False


class TestReadOnlyTools:
    def test_each_read_only_tool_declares_true(self):
        for cls in EXPECTED_READ_ONLY:
            assert cls().is_read_only is True, (
                f"{cls.__name__} must declare is_read_only=True "
                f"(it was on the read-only whitelist)"
            )

    def test_no_write_tool_declares_true(self):
        for cls in KNOWN_WRITE_TOOLS:
            assert cls().is_read_only is False, (
                f"{cls.__name__} has a side effect (writes / runs / "
                f"mutates) and must not claim is_read_only=True — "
                f"that would let the chat agent call it."
            )


class TestChatAgentRole:
    def test_returns_only_read_only_tools(self):
        tools = get_tools_for_role("chat_agent")
        for tool in tools:
            assert tool.is_read_only is True, (
                f"chat_agent role returned non-readonly tool {tool.name}"
            )

    def test_no_known_write_tool_is_returned(self):
        tools = get_tools_for_role("chat_agent")
        names = {t.name for t in tools}
        forbidden = {
            "create_file", "replace_lines", "write_file",
            "add_content_after_line", "add_content_before_line",
            "edit_symbol", "add_symbol", "remove_symbol",
            "rename_symbol", "move_symbol",
            "git_commit", "git_branch",
            "record_finding", "update_finding", "delete_finding",
            "execute_command", "code_interpreter",
            "send_message",
        }
        overlap = names & forbidden
        assert not overlap, f"chat_agent exposes write tools: {overlap}"

    def test_includes_expected_read_tools(self):
        tools = get_tools_for_role("chat_agent")
        names = {t.name for t in tools}
        must_include = {
            "read_file", "list_directory", "code_search", "glob",
            "find_references", "get_symbol_code", "list_symbols",
            "search_symbols", "project_structure", "analyze_code",
            "find_similar_methods", "search_by_docstring",
            "iter_symbols", "project_stats",
            "git_diff", "git_status",
            "read_findings", "search_findings",
        }
        missing = must_include - names
        assert not missing, f"chat_agent missing read-only tools: {missing}"

    def test_default_role_returns_full_toolset(self):
        # Spot-check that an unknown/default role still returns everything
        # — we haven't regressed the existing behavior for the developer.
        tools = get_tools_for_role("developer")
        names = {t.name for t in tools}
        assert "read_file" in names
        assert "create_file" in names
        assert "execute_command" in names
