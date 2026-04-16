from infinidev.tools.file import (
    ReadFileTool, WriteFileTool, MultiEditFileTool,
    ListDirectoryTool, CodeSearchTool, GlobTool,
    CreateFileTool, ReplaceLinesTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
)
from infinidev.tools.meta import HelpTool
from infinidev.tools.meta.plan_tools import AddStepTool, ModifyStepTool, RemoveStepTool
from infinidev.tools.meta.declare_test_command_tool import DeclareTestCommandTool
from infinidev.tools.meta.tail_test_output_tool import TailTestOutputTool
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
    FindSimilarMethodsTool, SearchByDocstringTool,
    IterSymbolsTool, ProjectStatsTool,
)

FILE_TOOLS = [ReadFileTool, CreateFileTool, ReplaceLinesTool, AddContentAfterLineTool, AddContentBeforeLineTool, ListDirectoryTool, CodeSearchTool, GlobTool]
META_TOOLS = [HelpTool, AddStepTool, ModifyStepTool, RemoveStepTool, DeclareTestCommandTool, TailTestOutputTool]
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
CODE_INTEL_TOOLS = [FindReferencesTool, ListSymbolsTool, SearchSymbolsTool, GetSymbolCodeTool, ProjectStructureTool, EditSymbolTool, AddSymbolTool, RemoveSymbolTool, AnalyzeCodeTool, RenameSymbolTool, MoveSymbolTool, FindSimilarMethodsTool, SearchByDocstringTool, IterSymbolsTool, ProjectStatsTool]

# Curated subset for small models (<25B) — tools with simple schemas
SMALL_MODEL_TOOLS = [
    # File I/O (8)
    ReadFileTool, CreateFileTool, ReplaceLinesTool,
    AddContentAfterLineTool, AddContentBeforeLineTool,
    ListDirectoryTool, CodeSearchTool, GlobTool,
    # Git (3)
    GitCommitTool, GitDiffTool, GitStatusTool,
    # Shell (2)
    ExecuteCommandTool, CodeInterpreterTool,
    # Knowledge (2)
    RecordFindingTool, SearchFindingsTool,
    # Code intelligence (8)
    SearchSymbolsTool, GetSymbolCodeTool, EditSymbolTool,
    FindReferencesTool, FindSimilarMethodsTool, SearchByDocstringTool,
    IterSymbolsTool, ProjectStatsTool,
    # Plan management (3)
    AddStepTool, ModifyStepTool, RemoveStepTool,
    # Project introspection (2)
    DeclareTestCommandTool, TailTestOutputTool,
]


def get_tools_for_role(role: str, *, small_model: bool = False) -> list:
    """Simplified tool selection for the CLI.

    role="chat_agent" returns only tools whose class declares
    is_read_only=True. The chat agent is the default entry point of the
    pipeline; the whitelist at the schema level is the security boundary
    — prompt rules alone cannot stop a model from calling a write tool
    if the schema exposes it.
    """
    all_tool_classes = FILE_TOOLS + GIT_TOOLS + SHELL_TOOLS + WEB_TOOLS + KNOWLEDGE_TOOLS + CHAT_TOOLS + DOCS_TOOLS + CODE_INTEL_TOOLS + META_TOOLS
    if role == "chat_agent":
        # Instantiate each tool and keep only the read-only ones. Pydantic
        # moves class-level field defaults into model_fields so getattr on
        # the class returns the descriptor rather than the default value —
        # instantiating is the reliable way to read is_read_only.
        instances = [cls() for cls in all_tool_classes]
        return [t for t in instances if t.is_read_only]
    if small_model:
        return [cls() for cls in SMALL_MODEL_TOOLS]
    return [cls() for cls in all_tool_classes]
