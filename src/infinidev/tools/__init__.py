"""Infinidev Tool Registry."""

from infinidev.tools.file import (
    ReadFileTool, WriteFileTool, EditFileTool, MultiEditFileTool,
    ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool,
    CreateFileTool, ReplaceLinesTool, PartialReadTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
)
from infinidev.tools.meta import HelpTool
from infinidev.tools.meta.plan_tools import AddStepTool, ModifyStepTool, RemoveStepTool
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
    AnalyzeCodeTool, RenameSymbolTool, MoveSymbolTool,
)

FILE_TOOLS = [ReadFileTool, PartialReadTool, CreateFileTool, ReplaceLinesTool, AddContentAfterLineTool, AddContentBeforeLineTool, ApplyPatchTool, ListDirectoryTool, CodeSearchTool, GlobTool]
META_TOOLS = [HelpTool, AddStepTool, ModifyStepTool, RemoveStepTool]
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
CODE_INTEL_TOOLS = [FindReferencesTool, ListSymbolsTool, SearchSymbolsTool, GetSymbolCodeTool, ProjectStructureTool, EditSymbolTool, AddSymbolTool, RemoveSymbolTool, AnalyzeCodeTool, RenameSymbolTool, MoveSymbolTool]

# Curated subset for small models (<25B) — tools with simple schemas
SMALL_MODEL_TOOLS = [
    # File I/O (8)
    ReadFileTool, CreateFileTool, ReplaceLinesTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
    ListDirectoryTool, CodeSearchTool, GlobTool,
    # Git (3)
    GitCommitTool, GitDiffTool, GitStatusTool,
    # Shell (1)
    ExecuteCommandTool,
    # Knowledge (2)
    RecordFindingTool, SearchFindingsTool,
    # Code intelligence (4)
    SearchSymbolsTool, GetSymbolCodeTool, EditSymbolTool,
    FindReferencesTool,
    # Plan management (3)
    AddStepTool, ModifyStepTool, RemoveStepTool,
]


def get_tools_for_role(role: str, *, small_model: bool = False) -> list:
    """Simplified tool selection for the CLI."""
    if small_model:
        return [cls() for cls in SMALL_MODEL_TOOLS]
    tool_classes = FILE_TOOLS + GIT_TOOLS + SHELL_TOOLS + WEB_TOOLS + KNOWLEDGE_TOOLS + CHAT_TOOLS + DOCS_TOOLS + CODE_INTEL_TOOLS + META_TOOLS
    return [cls() for cls in tool_classes]
