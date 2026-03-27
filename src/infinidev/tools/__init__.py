"""Infinidev Tool Registry."""

from infinidev.tools.file import (
    ReadFileTool, WriteFileTool, EditFileTool, MultiEditFileTool,
    ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool,
    CreateFileTool, ReplaceLinesTool, PartialReadTool,
)
from infinidev.tools.meta import HelpTool
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
from infinidev.tools.code_intel import (
    FindDefinitionTool, FindReferencesTool, ListSymbolsTool, SearchSymbolsTool,
    GetSymbolCodeTool, ProjectStructureTool,
    EditSymbolTool, AddSymbolTool, RemoveSymbolTool,
    EditMethodTool, AddMethodTool, RemoveMethodTool,  # backward-compat aliases
)

FILE_TOOLS = [ReadFileTool, PartialReadTool, CreateFileTool, ReplaceLinesTool, ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool]
META_TOOLS = [HelpTool]
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
CODE_INTEL_TOOLS = [FindReferencesTool, ListSymbolsTool, SearchSymbolsTool, GetSymbolCodeTool, ProjectStructureTool, EditSymbolTool, AddSymbolTool, RemoveSymbolTool]

def get_tools_for_role(role: str) -> list:
    """Simplified tool selection for the CLI."""
    tool_classes = FILE_TOOLS + GIT_TOOLS + SHELL_TOOLS + WEB_TOOLS + KNOWLEDGE_TOOLS + CHAT_TOOLS + DOCS_TOOLS + CODE_INTEL_TOOLS + META_TOOLS
    return [cls() for cls in tool_classes]
