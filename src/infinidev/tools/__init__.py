"""Infinidev Tool Registry."""

from infinidev.tools.file import (
    ReadFileTool, WriteFileTool, EditFileTool, MultiEditFileTool,
    ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool,
)
from infinidev.tools.git import (
    GitBranchTool, GitCommitTool,
    GitDiffTool, GitStatusTool,
)
from infinidev.tools.shell import ExecuteCommandTool, CodeInterpreterTool
from infinidev.tools.web import WebSearchTool, WebFetchTool, CodeSearchWebTool
from infinidev.tools.knowledge import (
    RecordFindingTool, ReadFindingsTool, SearchFindingsTool,
    ValidateFindingTool, RejectFindingTool, UpdateFindingTool, DeleteFindingTool,
    WriteReportTool, ReadReportTool, DeleteReportTool,
    SearchKnowledgeTool, SummarizeFindingsTool,
)
from infinidev.tools.chat import SendMessageTool
from infinidev.tools.docs import DeleteDocumentationTool, FindDocumentationTool, UpdateDocumentationTool

FILE_TOOLS = [ReadFileTool, WriteFileTool, EditFileTool, MultiEditFileTool, ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool]
GIT_TOOLS = [GitBranchTool, GitCommitTool, GitDiffTool, GitStatusTool]
SHELL_TOOLS = [ExecuteCommandTool, CodeInterpreterTool]
WEB_TOOLS = [WebSearchTool, WebFetchTool, CodeSearchWebTool]
KNOWLEDGE_TOOLS = [
    RecordFindingTool, ReadFindingsTool, SearchFindingsTool,
    ValidateFindingTool, RejectFindingTool, UpdateFindingTool, DeleteFindingTool,
    WriteReportTool, ReadReportTool, DeleteReportTool,
    SearchKnowledgeTool, SummarizeFindingsTool,
]
CHAT_TOOLS = [SendMessageTool]
DOCS_TOOLS = [DeleteDocumentationTool, FindDocumentationTool, UpdateDocumentationTool]

def get_tools_for_role(role: str) -> list:
    """Simplified tool selection for the CLI."""
    tool_classes = FILE_TOOLS + GIT_TOOLS + SHELL_TOOLS + WEB_TOOLS + KNOWLEDGE_TOOLS + CHAT_TOOLS + DOCS_TOOLS
    return [cls() for cls in tool_classes]
