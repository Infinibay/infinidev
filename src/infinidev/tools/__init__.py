"""Infinidev Tool Registry."""

from infinidev.tools.file import (
    ReadFileTool, WriteFileTool, EditFileTool,
    ListDirectoryTool, CodeSearchTool, GlobTool,
)
from infinidev.tools.git import (
    GitBranchTool, GitCommitTool,
    GitDiffTool, GitStatusTool,
)
from infinidev.tools.shell import ExecuteCommandTool
from infinidev.tools.web import WebSearchTool, WebFetchTool
from infinidev.tools.knowledge import (
    RecordFindingTool, ReadFindingsTool, SearchFindingsTool,
    ValidateFindingTool, RejectFindingTool, UpdateFindingTool, DeleteFindingTool,
    WriteReportTool, ReadReportTool, DeleteReportTool,
    SearchKnowledgeTool, SummarizeFindingsTool,
)

FILE_TOOLS = [ReadFileTool, WriteFileTool, EditFileTool, ListDirectoryTool, CodeSearchTool, GlobTool]
GIT_TOOLS = [GitBranchTool, GitCommitTool, GitDiffTool, GitStatusTool]
SHELL_TOOLS = [ExecuteCommandTool]
WEB_TOOLS = [WebSearchTool, WebFetchTool]
KNOWLEDGE_TOOLS = [
    RecordFindingTool, ReadFindingsTool, SearchFindingsTool,
    ValidateFindingTool, RejectFindingTool, UpdateFindingTool, DeleteFindingTool,
    WriteReportTool, ReadReportTool, DeleteReportTool,
    SearchKnowledgeTool, SummarizeFindingsTool,
]

def get_tools_for_role(role: str) -> list:
    """Simplified tool selection for the CLI."""
    tool_classes = FILE_TOOLS + GIT_TOOLS + SHELL_TOOLS + WEB_TOOLS + KNOWLEDGE_TOOLS
    return [cls() for cls in tool_classes]
