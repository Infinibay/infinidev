from infinidev.tools.file import (
    ReadFileTool,
    WriteFileTool,
    MultiEditFileTool,
    EditFileTool,
    ListDirectoryTool,
    CodeSearchTool,
    GlobTool,
    CreateFileTool,
    ReplaceLinesTool,
    AddContentAfterLineTool,
    AddContentBeforeLineTool,
    ViewImageTool,
)
from infinidev.tools.mcp_bridge import discover_mcp_tool_classes
from infinidev.tools.meta import HelpTool
from infinidev.tools.meta.recall_context_tool import RecallContextTool
from infinidev.tools.meta.plan_tools import (
    AddStepTool,
    ModifyStepTool,
    RemoveStepTool,
    PlanAddTool,
    PlanListTool,
    PlanRemoveTool,
    PlanUpdateTool,
)
from infinidev.tools.meta.declare_test_command_tool import DeclareTestCommandTool
from infinidev.tools.meta.tail_test_output_tool import TailTestOutputTool
from infinidev.tools.git import (
    GitBranchTool,
    GitCommitTool,
    GitDiffTool,
    GitStatusTool,
)
from infinidev.tools.shell import (
    ExecuteCommandTool,
    CodeInterpreterTool,
    RunInBackgroundTool,
    BackgroundStatusTool,
    StopBackgroundTaskTool,
    WaitForBackgroundTaskTool,
)
from infinidev.tools.web import WebSearchTool, WebFetchTool, CodeSearchWebTool
from infinidev.tools.knowledge import (
    RecordFindingTool,
    ReadFindingsTool,
    SearchFindingsTool,
    ValidateFindingTool,
    RejectFindingTool,
    UpdateFindingTool,
    DeleteFindingTool,
    WriteReportTool,
    ReadReportTool,
    ReadCommandOutputTool,
    DeleteReportTool,
    SearchKnowledgeTool,
    SummarizeFindingsTool,
)
from infinidev.tools.chat import SendMessageTool
from infinidev.tools.chat_agent import RespondTool, EscalateTool
from infinidev.tools.planner import EmitPlanTool
from infinidev.tools.council import (
    COUNCIL_MEMBER_TOOLS as _COUNCIL_MEMBER_TOOLS,
    COUNCIL_MODERATOR_TOOLS as _COUNCIL_MODERATOR_TOOLS,
)
from infinidev.tools.docs import (
    DeleteDocumentationTool,
    FindDocumentationTool,
    UpdateDocumentationTool,
)
from infinidev.tools.code_intel import (
    FindDefinitionTool,
    FindReferencesTool,
    ListSymbolsTool,
    SearchSymbolsTool,
    GetSymbolCodeTool,
    ProjectStructureTool,
    EditSymbolTool,
    AddSymbolTool,
    RemoveSymbolTool,
    EditMethodTool,
    AddMethodTool,
    RemoveMethodTool,  # backward-compat aliases
    AnalyzeCodeTool,
    RenameSymbolTool,
    MoveSymbolTool,
    FindSimilarMethodsTool,
    SearchByDocstringTool,
    IterSymbolsTool,
    ProjectStatsTool,
)

FILE_TOOLS = [
    ReadFileTool,
    CreateFileTool,
    EditFileTool,
    # ReplaceLinesTool / AddContentAfterLineTool / AddContentBeforeLineTool are
    # unbound: all three are `edit_file` with a different way of pointing at
    # the text. Line numbers shift as soon as an earlier edit in the same step
    # lands, so an off-by-one writes into the wrong place and reports success;
    # an exact-text match refuses instead. MultiEditFileTool stays unbound for
    # the same reason it always was — one way to edit a file.
    ListDirectoryTool,
    CodeSearchTool,
    GlobTool,
    ViewImageTool,
]
# Tools that only make sense when the model can see images.
# Filtered out in get_tools_for_role when supports_vision is False so they
# never reach the schema the LLM sees.
VISION_ONLY_TOOLS = {ViewImageTool}
META_TOOLS = [
    HelpTool,
    RecallContextTool,
    AddStepTool,
    ModifyStepTool,
    RemoveStepTool,
    # PlanAddTool / PlanListTool / PlanUpdateTool / PlanRemoveTool are
    # deliberately not bound. They back a second, durable plan under
    # `.infinidev/plans` — but nothing in the engine, the prompt or the
    # review ever reads it back, so writing to it is a no-op the model
    # pays ~425 tokens of schema to be tempted by. Worse, two ways to
    # "manage a plan" is exactly the ambiguity that makes tool selection
    # unreliable: `add_step` is the one that steers the run.
    # `plan_store` and `plan_tools` stay in the tree — rebinding them is
    # this list, once something consumes the store.
    DeclareTestCommandTool,
    TailTestOutputTool,
]
GIT_TOOLS = [GitBranchTool, GitCommitTool, GitDiffTool, GitStatusTool]
SHELL_TOOLS = [
    ExecuteCommandTool,
    CodeInterpreterTool,
    RunInBackgroundTool,
    BackgroundStatusTool,
    StopBackgroundTaskTool,
    WaitForBackgroundTaskTool,
]
WEB_TOOLS = [WebSearchTool, WebFetchTool, CodeSearchWebTool]
KNOWLEDGE_TOOLS = [
    RecordFindingTool,
    # ReadFindingsTool is unbound: it and `search_knowledge` were both
    # full-text search over findings, and two tools for one algorithm is the
    # ambiguity that makes tool selection unreliable. `search_knowledge`
    # absorbed its browse mode and its session/type filters; the name still
    # resolves through `_TOOL_ALIASES`. `search_findings` stays — semantic
    # search is a different algorithm, not a different spelling.
    SearchFindingsTool,
    ValidateFindingTool,
    RejectFindingTool,
    UpdateFindingTool,
    DeleteFindingTool,
    WriteReportTool,
    ReadReportTool,
    ReadCommandOutputTool,
    DeleteReportTool,
    SearchKnowledgeTool,
    SummarizeFindingsTool,
]
CHAT_TOOLS = [SendMessageTool]
# Exclusive to the chat agent tier — NOT bound to the developer.
# These are schema-level terminators (like step_complete); the chat
# orchestrator parses their args directly from the LLM response.
CHAT_AGENT_TOOLS = [RespondTool, EscalateTool]
# Exclusive to the planner tier — NOT bound to the chat agent or the
# developer. The planner orchestrator reads its args directly as the
# final artifact of the planning turn.
PLANNER_TOOLS = [EmitPlanTool]
# Exclusive to the council tiers — schema-level terminators read
# directly by the council orchestrator (see engine/council/). Members
# get post/conclude; the moderator gets seed/verdict/synthesize.
COUNCIL_MEMBER_TOOLS = _COUNCIL_MEMBER_TOOLS
COUNCIL_MODERATOR_TOOLS = _COUNCIL_MODERATOR_TOOLS
DOCS_TOOLS = [DeleteDocumentationTool, FindDocumentationTool, UpdateDocumentationTool]
CODE_INTEL_TOOLS = [
    FindReferencesTool,
    ListSymbolsTool,
    SearchSymbolsTool,
    GetSymbolCodeTool,
    ProjectStructureTool,
    # EditSymbolTool / AddSymbolTool / RemoveSymbolTool are unbound: replacing,
    # inserting or deleting a symbol body is `edit_file` addressed by name.
    # RenameSymbolTool and MoveSymbolTool stay — those rewrite every reference
    # and import across the index, which is an algorithm the model cannot
    # reproduce by editing files one at a time.
    AnalyzeCodeTool,
    RenameSymbolTool,
    MoveSymbolTool,
    FindSimilarMethodsTool,
    SearchByDocstringTool,
    IterSymbolsTool,
    ProjectStatsTool,
]

# Curated subset for small models (<25B) — tools with simple schemas
SMALL_MODEL_TOOLS = [
    # File I/O (8)
    ReadFileTool,
    CreateFileTool,
    EditFileTool,
    ListDirectoryTool,
    CodeSearchTool,
    GlobTool,
    # Git (3)
    GitCommitTool,
    GitDiffTool,
    GitStatusTool,
    # Shell (6)
    ExecuteCommandTool,
    CodeInterpreterTool,
    RunInBackgroundTool,
    BackgroundStatusTool,
    StopBackgroundTaskTool,
    WaitForBackgroundTaskTool,
    # Knowledge (2)
    RecordFindingTool,
    SearchFindingsTool,
    # Code intelligence (8)
    SearchSymbolsTool,
    GetSymbolCodeTool,
    FindReferencesTool,
    FindSimilarMethodsTool,
    SearchByDocstringTool,
    IterSymbolsTool,
    ProjectStatsTool,
    # Plan management (3)
    AddStepTool,
    ModifyStepTool,
    RemoveStepTool,
    # Project introspection (2)
    DeclareTestCommandTool,
    TailTestOutputTool,
]


def get_tools_for_role(
    role: str,
    *,
    small_model: bool = False,
    supports_vision: bool | None = None,
) -> list:
    """Simplified tool selection for the CLI.

    role="chat_agent" returns only tools whose class declares
    is_read_only=True. The chat agent is the default entry point of the
    pipeline; the whitelist at the schema level is the security boundary
    — prompt rules alone cannot stop a model from calling a write tool
    if the schema exposes it.

    ``supports_vision`` gates VISION_ONLY_TOOLS. When None, it's looked up
    from the model capability cache at call time so existing callers don't
    have to be updated.
    """
    if supports_vision is None:
        # Use the lightweight vision check directly — it only consults
        # LiteLLM's static metadata table. Going through
        # get_model_capabilities() would trigger a full capability probe
        # (Ollama /api/show or a live litellm.completion for
        # openai_compatible/vllm), which is too heavy a side-effect for a
        # tool-list lookup and breaks tests that patch litellm.completion.
        try:
            from infinidev.config.model_capabilities import _detect_vision_support

            supports_vision = _detect_vision_support()
        except Exception:
            supports_vision = False

    def _vision_filter(classes: list) -> list:
        if supports_vision:
            return classes
        return [c for c in classes if c not in VISION_ONLY_TOOLS]

    # Tools published by the configured MCP servers, under the names those
    # servers gave them (``ken_rank``, ``ken_recall``, …). Discovery is
    # cached and non-blocking, so a cold session simply gets none of them
    # this turn and all of them the next; a local tool always wins a name
    # collision, since a remote server must not be able to shadow read_file.
    mcp_tool_classes = discover_mcp_tool_classes()

    # CHAT_AGENT_TOOLS (respond, escalate) are NOT in the developer
    # toolset — they're exclusive to the chat agent tier. The developer
    # uses step_complete for termination; the chat agent uses respond
    # and escalate.
    local_tool_classes = _vision_filter(
        FILE_TOOLS
        + GIT_TOOLS
        + SHELL_TOOLS
        + WEB_TOOLS
        + KNOWLEDGE_TOOLS
        + CHAT_TOOLS
        + DOCS_TOOLS
        + CODE_INTEL_TOOLS
        + META_TOOLS
    )
    # Keep the disabled-by-default capture feature out of the tool schema and
    # generated prompt entirely. This preserves the pre-feature prompt surface
    # when all flags are off; a handle reader is useful only if this run can
    # produce handles in the first place.
    from infinidev.config.settings import settings

    if not settings.COMMAND_OUTPUT_CAPTURE_ENABLED:
        local_tool_classes = [
            cls for cls in local_tool_classes if cls is not ReadCommandOutputTool
        ]
    def _declared_name(cls) -> str | None:
        field = cls.model_fields.get("name")
        return getattr(field, "default", None) if field is not None else None

    local_names = {_declared_name(c) for c in local_tool_classes}
    all_tool_classes = local_tool_classes + [
        c for c in mcp_tool_classes if _declared_name(c) not in local_names
    ]
    if role == "chat_agent":
        # Instantiate each tool and keep only the read-only ones. Pydantic
        # moves class-level field defaults into model_fields so getattr on
        # the class returns the descriptor rather than the default value —
        # instantiating is the reliable way to read is_read_only.
        read_only = [t for t in (cls() for cls in all_tool_classes) if t.is_read_only]
        return read_only + [cls() for cls in CHAT_AGENT_TOOLS]
    if role == "planner":
        # Planner gets the same read-only exploration tools as the chat
        # agent, plus EmitPlanTool as its terminator. Tight budget
        # (~4 tool calls) enforced by the orchestrator, not the tool
        # list.
        read_only = [t for t in (cls() for cls in all_tool_classes) if t.is_read_only]
        return read_only + [cls() for cls in PLANNER_TOOLS]
    if role == "assistant_critic":
        # The pair-programming critic gets read-only exploration tools
        # so it can verify the principal's claims (read the file the
        # principal says it read, run code_search to confirm a pattern
        # exists, etc.) before emitting a verdict. The critic's
        # terminator is ``emit_verdict`` (registered separately in
        # ``critic.py``), so we don't add chat_agent terminators here.
        # Tight budget enforced by the critic's own sub-loop, not the
        # tool list.
        return [t for t in (cls() for cls in all_tool_classes) if t.is_read_only]
    if role == "council_member":
        # A council subagent: read-only exploration (codebase + web — the
        # web tools are now is_read_only, so they flow in here) plus the
        # channel terminators (channel_post, conclude). It can NEVER
        # write — the council is a design/research phase, enforced at the
        # schema level.
        read_only = [t for t in (cls() for cls in all_tool_classes) if t.is_read_only]
        return read_only + [cls() for cls in COUNCIL_MEMBER_TOOLS]
    if role == "council_moderator":
        # The orchestrator of the council: same read-only exploration
        # plus its three terminators (seed_council, council_verdict,
        # synthesize_brief).
        read_only = [t for t in (cls() for cls in all_tool_classes) if t.is_read_only]
        return read_only + [cls() for cls in COUNCIL_MODERATOR_TOOLS]
    if small_model:
        # A small model's problem is choosing, not capability: the curated
        # list exists because a 90-tool schema wrecks its selection. MCP
        # tools join it only once MCP_TOOL_FILTER names which ones matter,
        # so "expose everything" never silently lands on a 7B model.
        narrowed = str(getattr(settings, "MCP_TOOL_FILTER", "*") or "*").strip()
        extra = mcp_tool_classes if narrowed and narrowed != "*" else []
        small_classes = _vision_filter(SMALL_MODEL_TOOLS)
        if settings.COMMAND_OUTPUT_CAPTURE_ENABLED:
            small_classes.append(ReadCommandOutputTool)
        return [cls() for cls in small_classes + extra]
    return [cls() for cls in all_tool_classes]
